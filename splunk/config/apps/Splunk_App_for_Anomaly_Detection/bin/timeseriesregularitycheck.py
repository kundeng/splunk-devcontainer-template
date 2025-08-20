import os
if len(os.getenv('SPLUNK_HOME', '')) > 0:
    import exec_anaconda
    exec_anaconda.exec_anaconda()

import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from splunklib.searchcommands import dispatch, ReportingCommand, Configuration, Option
from saad.utils.parsing_utils import timestamp_to_unix_ms, is_number
from saad.utils import setup_logging

logger = setup_logging.get_logger()


@Configuration(requires_preop=True)
class TimeSeriesRegularityCheck(ReportingCommand):
    """
    The command checks if the given time series is evenly spaced, and if not, it tests
    different resolutions to find those that result in valid down-sampled times series.

    Example usage:
    | inputlookup kpi.csv
    | timeseriesregularitycheck
    """

    MIN_DATA_POINTS = 10  # The min acceptable number of data points produced when testing a resolution
    MAX_DATA_POINTS = 50000  # The max acceptable number of data points produced when testing a resolution
    # TODO (WD): I'd suggest the threshold be larger than 0.1 (maybe closer to 0.3) for this check, as it seems we'd rather impute than force aggregation
    MAX_MISSING_DATA = (
        0.1  # Max amount of data that can be missing/filled when testing a resolution
    )
    TEST_RESOLUTIONS = {  # Resolutions to test, in milliseconds
        "10ms": 10,  # 10 millisecond
        "100ms": 100,  # 100 millisecond
        "1s": 1e3,  # 1 second (1k MS)
        "5s": 5e3,  # 5 seconds (5k MS)
        "15s": 1.5e4,  # 15 seconds (15k MS)
        "1m": 6e4,  # 1 minute (60k MS)
        "5m": 3e5,  # 5 minutes (300k MS)
        "15m": 9e5,  # 15 minutes (900k MS)
        "1h": 3.6e6,  # 1 hour (3.6m MS)
        "4h": 1.44e7,  # 4 hours (14.4m MS)
        "6h": 2.16e7,  # 6 hours (21.6m MS)
        "12h": 4.32e7,  # 12 hours (43.2m MS)
        "1d": 8.64e7,  # 1 day (86.4m MS)
    }

    time_field = Option(default="_time")

    timestamps = []

    @Configuration()
    def map(self, events):
        return events

    def reduce(self, events):
        chunk_timestamps = [event[self.time_field] for event in events]

        # If there are > 0 records, process and append the timestamps
        if len(chunk_timestamps) > 0 and chunk_timestamps != ['']:
            logger.info(f'{setup_logging.ANOMALY_APP_TELEMETRY} '
                        f'Example timestamps: {chunk_timestamps[0]}, {chunk_timestamps[-1]}')
            self.timestamps.extend(chunk_timestamps)

        # After all chunks have been received, process the complete dataset
        if self._finished:
            preprocessed_timestamps = self.preprocess_time_series()
            logger.info(f'{setup_logging.ANOMALY_APP_TELEMETRY} '
                        f'Running check on time series of length {len(preprocessed_timestamps)}.')

            # Check if the timestamps are already evenly spaced (potentially with some missing values)
            is_evenly_spaced = TimeSeriesRegularityCheck.test_if_evenly_spaced(preprocessed_timestamps)
            if is_evenly_spaced:
                resolutions = []

            # If the timestamps aren't evenly spaced, test different resolutions
            else:
                resolutions = TimeSeriesRegularityCheck.find_valid_resolutions(preprocessed_timestamps)
                logger.info(f"{setup_logging.ANOMALY_APP_TELEMETRY} "
                            f"Time series unevenly-spaced. Valid aggregation spans: {resolutions}")

            yield {"resolutions": resolutions}

    @staticmethod
    def find_valid_resolutions(timestamps_ms):
        """
        Test each hard-coded resolution to see if it is a "valid" aggregation span. If none
        are valid, then return all resolutions that create fewer than MAX_DATA_POINTS points.
        Should never return an empty list, except for datasets longer than 50k days (137 years)
        """
        fully_valid = []
        partially_valid = []
        for resolution_str, resolution_ms in TimeSeriesRegularityCheck.TEST_RESOLUTIONS.items():
            result_state = TimeSeriesRegularityCheck.test_resolution(timestamps_ms, resolution_ms)
            if result_state == 1:
                partially_valid.append(resolution_str)
            if result_state == 2:
                fully_valid.append(resolution_str)

        return fully_valid if len(fully_valid) > 0 else partially_valid

    @staticmethod
    def test_resolution(timestamps_ms, resolution_ms):
        """
        This function tests to see if a time series can be resampled to the specified resolution
        without producing too few or too many data points. A resolution is deemed valid if the following
        conditions are met:
        1) it has a valid number of buckets (MIN_DATA_POINTS <= num_buckets <= MAX_DATA_POINTS)
        2) we don't interpolate too much data (fill amount <= MAX_MISSING_DATA)

        Return values:
        0 = Fails both conditions
        1 = Condition 1 passes, but condition 2 fails
        2 = Both conditions pass
        """
        timestamps_ms = sorted(timestamps_ms)
        resolution_str = f"{int(resolution_ms)}ms"

        # Check how many data points would be produced when using the given resolution
        min_buckets = TimeSeriesRegularityCheck.MIN_DATA_POINTS
        max_buckets = TimeSeriesRegularityCheck.MAX_DATA_POINTS
        num_buckets = int((timestamps_ms[-1] - timestamps_ms[0]) // resolution_ms) + 1
        if (num_buckets < min_buckets) or (num_buckets > max_buckets):
            return 0

        # Group the timestamps
        buckets = [0] * num_buckets
        for ts in timestamps_ms:
            bucket_i = int((ts - timestamps_ms[0]) // resolution_ms)
            buckets[bucket_i] += 1

        # Check how many buckets have no data
        empty_count = sum(int(count == 0) for count in buckets)
        missing_ratio = empty_count / len(buckets)
        if missing_ratio > TimeSeriesRegularityCheck.MAX_MISSING_DATA:
            return 1

        # If it passes both checks, it's a valid resolution
        return 2

    @staticmethod
    def test_if_evenly_spaced(timestamps_ms):
        # Select the minimum time-step as the candidate resolution
        timestamp_diffs_ms = np.diff(timestamps_ms)
        candidate_resolution_ms = timestamp_diffs_ms.min()

        # If there are duplicate timestamps, fail
        if candidate_resolution_ms == 0:
            return False

        # Check that each time-step is a multiple of the candidate resolution (allows for missing values)
        timestamp_residuals = [
            timestamp_diff_ms % candidate_resolution_ms
            for timestamp_diff_ms in timestamp_diffs_ms
        ]
        if np.sum(timestamp_residuals) > 0:
            return False

        # Check that under this resolution, no more than MAX_MISSING_DATA would need to be imputed
        num_missing_values = np.sum([
            (step_size // candidate_resolution_ms) - 1
            for step_size in timestamp_diffs_ms
        ])
        missing_data_frac = num_missing_values / (len(timestamps_ms) + num_missing_values)
        is_valid = missing_data_frac <= TimeSeriesRegularityCheck.MAX_MISSING_DATA

        if is_valid:
            logger.info(f'{setup_logging.ANOMALY_APP_TELEMETRY} '
                        f'Time series evenly-spaced with resolution {candidate_resolution_ms/1000} seconds.')

        return is_valid

    def preprocess_time_series(self):
        # Convert timestamps to Unix MS, remove duplicates, and sort
        series = sorted(set(timestamp_to_unix_ms(t) for t in self.timestamps))

        # If we dropped some values, warn the user
        if len(series) < len(self.timestamps):
            self._write_warning(f'Detected records with duplicate timestamps. '
                                f'One record will be selected at random for each timestamp.')

        return series

    def _write_warning(self, message):
        """Writing warnings fails when run locally, so put it in a no-op try/catch"""
        try:
            self.write_warning(message)
        except:
            pass


if len(os.getenv('SPLUNK_HOME', '')) > 0:
    dispatch(TimeSeriesRegularityCheck, sys.argv, sys.stdin, sys.stdout, __name__)
