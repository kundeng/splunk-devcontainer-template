import os
if len(os.getenv('SPLUNK_HOME', '')) > 0:
    import exec_anaconda
    exec_anaconda.exec_anaconda()

import sys
from datetime import datetime

import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from splunklib.searchcommands import dispatch, EventingCommand, Configuration, Option

from timeseriesregularitycheck import TimeSeriesRegularityCheck
from saad.utils.parsing_utils import timestamp_to_unix_ms, is_number

from saad.utils import setup_logging
logger = setup_logging.get_logger()


@Configuration()
class InterpolateMissingValues(EventingCommand):
    """
    This command does the following:
    1) Infers the resolution of the dataset (can assume even spacing from timeseriesregularitycheck).
    2) Identifies missing data points in the time series.
    3) Fills missing data points by linearly interpolating between existing values.

    Example usage:
    | inputlookup interpolation_test.csv
    | interpolatemissingvalues value_field=val time_field=timestamp
    """

    value_field = Option(require=True)
    time_field = Option(default='_time')
    resolution = Option(default=None)
    prev_chunk_end_record = None  # Record previous chunk's final record to interpolate missing values between chunks

    @Configuration()
    def transform(self, records):

        # Format and validate the input data
        time_val_pairs = [(r[self.time_field], r[self.value_field]) for r in records]
        series = self.process_input_data(time_val_pairs)
        is_valid = self.validate_series(series)
        if not is_valid:
            return []

        # If there was a previous chunk, prepend its end-timestamp to interpolate values between chunks
        if self.prev_chunk_end_record is not None:
            if self.prev_chunk_end_record[0] > series[0][0]:  # Should never happen, but just in case....
                self._write_error(f'Chunks arrived out-of-order, so interpolation cannot be performed.')
            series = [self.prev_chunk_end_record] + series

        # Run interpolation
        logger.info(f'AnomalyApp Telemetry: Running command on dataset with length {len(series)}')
        filled_series = InterpolateMissingValues.interpolate_missing_values(series, self.resolution)
        logger.info(f'AnomalyApp Telemetry: Num values interpolated = {len(filled_series) - len(series)}')

        # Prepare output data
        filled_records = [
            {
                '_time': t / 1000, # Return in UNIX seconds
                self.value_field: v
            } for t, v in filled_series
        ]

        # If we prepended the previous chunk's timestamp, remove it
        if self.prev_chunk_end_record is not None:
            filled_records = filled_records[1:]
        self.prev_chunk_end_record = series[-1]

        return filled_records

    @staticmethod
    def interpolate_missing_values(time_val_pairs, resolution_str=None):

        if len(time_val_pairs) < 2:
            return time_val_pairs

        # Compute time series resolution; assume min timestamp diff is resolution if none provided
        if resolution_str is None:
            resolution_ms = np.diff(sorted(set(t for t, _ in time_val_pairs))).min()
        else:
            valid_resolutions = TimeSeriesRegularityCheck.TEST_RESOLUTIONS
            if resolution_str not in valid_resolutions:
                raise ValueError(f"Invalid resolution {resolution_str}. Valid options: {valid_resolutions.keys()}")
            resolution_ms = valid_resolutions[resolution_str]
        logger.info(f'Running interpolation with resolution: {resolution_ms}ms')

        # Create a new time series with missing values filled with 'None', and duplicate timestamps removed
        filled_series = []
        for i, (t1, v1) in enumerate(time_val_pairs):

            # If this timestamp is the same as the previous, skip the record
            if (i > 0) and (t1 == time_val_pairs[i - 1][0]):
                continue
            filled_series.append((t1, v1))

            # Fill in any missing timestamps with `None`
            if i < len(time_val_pairs) - 1:
                t2, v2 = time_val_pairs[i + 1]
                num_missing_vals = max(0, int((t2 - t1) / resolution_ms) - 1)
                step_size = (v2 - v1) / (num_missing_vals + 1)
                for fill_i in range(1, num_missing_vals + 1):
                    fill_t = int(t1 + (fill_i * resolution_ms))
                    fill_v = v1 + (fill_i * step_size)
                    filled_series.append((fill_t, fill_v))

        return filled_series

    @staticmethod
    def process_input_data(time_val_pairs):
        series = [
            (timestamp_to_unix_ms(t), float(str(v).replace(',', '')))
            for t, v in time_val_pairs
            if is_number(v)
        ]
        return series

    def validate_series(self, time_val_pairs):

        timestamps_ms = [t for t, v in time_val_pairs]

        # If there's no data after preprocessing, raise an error
        if len(time_val_pairs) == 0:
            self._write_error('Data contains no numeric values.')
            return False

        # Check for duplicate timestamps
        ctr = Counter(timestamps_ms)
        most_common_timestamp, count = ctr.most_common(1)[0]
        if count > 1:
            self._write_warning(f'Found {count} records with unix timestamp: "{most_common_timestamp}". '
                                f'One record will be selected at random to use.')

        # Ensure data is sorted
        for timestamp, next_timestamp in zip(timestamps_ms[:-1], timestamps_ms[1:]):
            if next_timestamp < timestamp:
                self._write_error(f'Interpolation requires sorted data; run "| sort 0 {{time_field}}" before invoking.')
                return False

        # Check if the data is evenly spaced. If not, raise an error
        is_evenly_spaced = InterpolateMissingValues.test_if_evenly_spaced(timestamps_ms)
        if not is_evenly_spaced:
            self._write_error('Interpolation requires evenly spaced timestamps. '
                              'You can use the "timechart" command to aggregate the data and try again.')
            return False

        return True

    @staticmethod
    def test_if_evenly_spaced(timestamps_ms):
        # Check if data is already perfectly spaced with no gaps
        timestamp_diffs_ms = np.diff(sorted(set(timestamps_ms)))
        if len(set(timestamp_diffs_ms)) == 1:
            return True

        # If data is not perfectly spaced, find best fitting resolution.
        # Select the minimum time-step as the candidate resolution
        candidate_resolution_ms = timestamp_diffs_ms.min()

        # If the candidate is  too small (candidate < 10ms), fail.
        # This prevents selecting 1ms resolution, and filling a TON of data
        if candidate_resolution_ms < 10:
            return False

        # Check that each time-step is a multiple of the candidate resolution (allows for missing values)
        timestamp_residuals = [
            timestamp_diff_ms % candidate_resolution_ms
            for timestamp_diff_ms in timestamp_diffs_ms
        ]
        if np.sum(timestamp_residuals) > 0:
            return False

        return True

    def _write_warning(self, message):
        """Writing warnings fails when run locally, so put it in a no-op try/catch"""
        try:
            self.write_warning(message)
        except:
            pass

    def _write_error(self, message):
        """Writing errors fails when run locally, so put it in a no-op try/catch"""
        try:
            self.write_error(message)
        except:
            pass


if len(os.getenv('SPLUNK_HOME', '')) > 0:
    dispatch(InterpolateMissingValues, sys.argv, sys.stdin, sys.stdout, __name__)
