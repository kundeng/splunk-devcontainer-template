import os
if len(os.getenv('SPLUNK_HOME', '')) > 0:
    import exec_anaconda
    exec_anaconda.exec_anaconda()

import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from splunklib.searchcommands import dispatch, ReportingCommand, Configuration, Option
from timeseriesregularitycheck import TimeSeriesRegularityCheck
from saad.utils.parsing_utils import timestamp_to_unix_ms, is_number
from saad.utils import setup_logging

logger = setup_logging.get_logger()


@Configuration(requires_preop=True)
class DetectMissingValues(ReportingCommand):
    """
    Computes the maximum number of consecutive "missing events" in the Splunk results. A missing event is either
    (i) a timestamp which is expected in the data based on the resolution but is absent, or (ii) such a timestamp which
    is present but has a missing or non-numeric value in the associated field on which we are performing time series analysis.

    Assumes that the resolution of the data is given by the smallest gap in the timestamps; this is
    assured by the aggregation window enforced by `timeseriesregularitycheck`.

    Example: if the results contain timestamps [11:00, 11:05, 11:10, 11:20, 11:25, 11:40, 11:45], this command will
    return 2, as there are 2 missing timestamps (with respect to the 5 minute resolution) between 11:25 and 11:40. If the value
    associated with the timestamp 11:40 was "-" or "NaN", the command would instead return 3.

    Usage:
        | inputlookup kpi.csv
        | detectmissingvalues value_field=input
    """

    value_field = Option(require=True)
    time_field = Option(default="_time")
    resolution = Option(default=None)
    time_series = []

    @Configuration()
    def map(self, events):
        return events

    def reduce(self, events):

        # If records were provided, append them to the accumulated time series.
        # Note that Splunk can sometimes give a 1-record chunk with no timestamp.
        chunk_ts = [(e[self.time_field], e[self.value_field]) for e in events]
        if len(chunk_ts) > 1 or (len(chunk_ts) == 1 and chunk_ts[0][0] != ['']):
            self.time_series.extend(chunk_ts)

        # Once all data has been received, process it
        if self._finished:
            preprocessed_series = self.preprocess_time_series()
            timestamps = [t for t, _ in preprocessed_series]
            max_consecutive_missing_vals = self.compute_max_consecutive_missing_vals(timestamps, self.resolution)
            logger.info(f"{setup_logging.ANOMALY_APP_TELEMETRY} "
                        f"Max number of consecutive missing values: {max_consecutive_missing_vals}")
            yield {"max_consecutive_missing_vals": max_consecutive_missing_vals}

    @staticmethod
    def compute_max_consecutive_missing_vals(timestamps_ms, resolution_str=None):
        """
        Computes the maximum number of consecutive missing values in the given timestamps.

        Input: `timestamps_ms`, a list of integers representing timestamps in milliseconds
        Returns: `max_consecutive_missing_vals`, integer, the maximum number of consecutive missing values in `timestamps`
        """
        if len(timestamps_ms) < 2:
            return 0
        timestamp_diffs = np.diff(timestamps_ms)

        # Compute time series resolution; assume min timestamp diff is resolution if none provided
        if resolution_str is None:
            resolution_ms = timestamp_diffs.min()
        else:
            valid_resolutions = TimeSeriesRegularityCheck.TEST_RESOLUTIONS
            if resolution_str not in valid_resolutions:
                raise ValueError(
                    f"Invalid resolution {resolution_str}. Valid options: {valid_resolutions.keys()}"
                )
            resolution_ms = valid_resolutions[resolution_str]

        max_timestamp_diff = timestamp_diffs.max()  # widest gap
        max_sequence = int((max_timestamp_diff // resolution_ms) - 1)  # max number of consecutive missing vals
        max_sequence = max(0, max_sequence)  # If there are duplicate timestamps, max_sequence will be -1
        return max_sequence

    def preprocess_time_series(self):
        # Filter non-numeric and duplicate values, and convert timestamps
        filtered_series_dict = {}
        dupe_dropped = False
        for t, v in self.time_series:
            if is_number(v):
                t = timestamp_to_unix_ms(t)
                if t not in filtered_series_dict:
                    filtered_series_dict[t] = v
                else:
                    dupe_dropped = True

        # Convert the dictionary into a sorted list
        filtered_series = sorted(filtered_series_dict.items())

        # If we dropped some values, warn the user
        if dupe_dropped:
            self._write_warning(f'Detected records with duplicate timestamps. '
                                f'One record will be selected at random for each timestamp.')

        return filtered_series

    def _write_warning(self, message):
        """Writing warnings fails when run locally, so put it in a no-op try/catch"""
        try:
            self.write_warning(message)
        except:
            pass


if len(os.getenv('SPLUNK_HOME', '')) > 0:
    dispatch(DetectMissingValues, sys.argv, sys.stdin, sys.stdout, __name__)
