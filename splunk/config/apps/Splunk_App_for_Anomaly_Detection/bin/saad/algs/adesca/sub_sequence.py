from datetime import timedelta
import numpy as np
import scipy.stats as stats

from saad.algs.adesca.data_prepare import COL_VALUE, DAY_OF_WEEK_NAME, resample_with_offset

SKIP_FIRST_SEGMENT = False

class SubSequence:
    def __init__(self, values, start_time, duration) -> None:
        self.values = values
        self.length = len(values)
        self.start_time = start_time
        self.duration = duration

        self.silhouette = float("nan")
        self.has_anomaly = False

    def __str__(self):
        return (
            f"start_time={self.start_time} {DAY_OF_WEEK_NAME[self.start_time.dayofweek]} duration={self.duration}, "
            + f'silhouette={self.silhouette:7.3f} {"has-anomaly" if self.has_anomaly else ""}, '
            + f"values=[({self.length}) min={np.min(self.values):.3f} max={np.max(self.values):.3f} mean={np.mean(self.values):.3f} med={np.median(self.values):.3f}]"
        )

    def hour(self):
        return self.start_time.hour

    def dayofweek(self):
        return self.start_time.dayofweek


def df_to_hour_sequences(df, hourDelta=1, offset=0):
    assert (
        hourDelta == 1 or hourDelta == 2 or hourDelta == 3 or hourDelta == 4
    ), f"hourDelta={hourDelta} is NOT supported"
    assert (
        offset >= 0 and offset < hourDelta
    ), f"Invalid offset({offset}), should be [0, hourDelta({hourDelta}))."

    dfre = resample_with_offset(df, f"{hourDelta}h", offset)
    subsequence_list = [
        SubSequence(
            np.array(x[1][COL_VALUE]),
            duration=timedelta(hours=hourDelta),
            start_time=x[0],
        )
        for x in dfre
    ]

    return subsequence_list[1:] if SKIP_FIRST_SEGMENT else subsequence_list # ignore beginning / ending partial block


def df_to_day_sequences(df, offset=0):
    dfre = resample_with_offset(df, "24h", offset)
    subsequence_list = [
        SubSequence(
            np.array(x[1][COL_VALUE]), duration=timedelta(days=1), start_time=x[0]
        )
        for x in dfre
    ]

    return subsequence_list[1:]  if SKIP_FIRST_SEGMENT else subsequence_list # ignore beginning / ending partial day


def df_to_half_day_sequences(df, start_hour):
    dfre = resample_with_offset(df, "12h", start_hour)
    subsequence_list = [
        SubSequence(
            np.array(x[1][COL_VALUE]), duration=timedelta(hours=12), start_time=x[0]
        )
        for x in dfre
    ]
    
    return subsequence_list[1:]  if SKIP_FIRST_SEGMENT else subsequence_list # ignore beginning / ending partial day TODO: cannot do this for incremental detect


def pack_sub_values_with_filter(subs, percentile_threshold):
    '''
    '''
    sub_len = len(subs[0].values)  if SKIP_FIRST_SEGMENT else len(subs[1].values) # TODO : handle head/tail partial subsequence
    scores = [sub.silhouette for sub in subs]
    percentiles = [stats.percentileofscore(scores, sub.silhouette) for sub in subs]
    subs = [
        sub
        for sub, percent in zip(subs, percentiles)
        if percent >= percentile_threshold
    ]

    cnt_subs = len(subs)
    sub_values = np.empty((cnt_subs, sub_len))
    for i, sub in enumerate(subs):
        if len(sub.values) < sub_len:
            sub_values[i] = np.concatenate(
                (sub.values, np.full(sub_len - len(sub.values), np.median(sub.values)))
            )
        else:
            sub_values[i] = sub.values

    return sub_values, [sub.start_time for sub in subs], percentiles
