import numpy as np
import pandas as pd
import time

from saad.algs.adesca.dev_util import (
    is_verbose,
    disable_multi_resolution,
)
from saad.algs.adesca.data_prepare import (
    COL_VALUE,
    down_sample,
    get_resolution,
    remove_super_spikes,
)
from saad.algs.adesca.pattern_evaluate import (
    THRESHOLD_SILHOUETTE,
    THRESHOLD_SILHOUETTE_HIGH,
    MIN_POINTS_PER_SUBSEQUENCE,
    SILHOUETTE_NEG_INF,
    check_weekly_pattern,
    evaluate_half_day,
    evaluate_hour_block,
)

DAYS_IN_WEEK = [0, 1, 2, 3, 4, 5, 6]

DAY_IN_SECOND = 1440
WEEK_IN_SECOND = 10080

TIMEDELTA_DAY = pd.Timedelta('1d')
TIMESELTA_HALF_DAY = pd.Timedelta('12h')
TIMEDELTA_15_MIN = pd.Timedelta('15min')

class TimePolicy:
    def __init__(
        self,
        hour_block_length,
        offset,
        has_weekend,
        offdays_start=5,
        score=SILHOUETTE_NEG_INF,
        label_method=None,
    ) -> None:
        self.hour_block_length = hour_block_length
        self.offset = offset
        self.has_weekend = has_weekend
        self.offdays_start = offdays_start
        self.score = score
        self.label_method = label_method

    def __str__(self) -> str:
        if self.has_weekend:
            week_clause = f"weekly seasonality"
        else:
            week_clause = ""

        if self.hour_block_length == 0:  # hour pattern score below threshold
            rslt = week_clause + (
                "" if self.offset == 0 else f", offset={self.offset}-hour"
            )
        else:
            rslt = (
                f"daily {int(self.hour_block_length)}-hour blocks"
                + ("" if self.offset == 0 else f", offset={self.offset}-hour")
                + week_clause
            )

        return (rslt if rslt else "no pattern") + f" ({self.score:.2f})"

def recommend_time_policy(df, threshold_silhouette=THRESHOLD_SILHOUETTE):
    df = remove_super_spikes(df)

    (
        hour_block_length,
        offset,
        has_weekend,
        offdays_start,
        score,
        label_method,
    ) = _evaluate_seasonality_patterns_at_multi_resolution(df, threshold_silhouette)

    return TimePolicy(
        hour_block_length,
        offset,
        has_weekend,
        offdays_start,
        score,
        label_method,
    )


def _evaluate_seasonality_patterns_at_multi_resolution(
    df, threshold_silhouette=THRESHOLD_SILHOUETTE
):
    df_resolution = get_resolution(df)
    if disable_multi_resolution():
        resolution = TIMEDELTA_15_MIN
        if df_resolution > resolution:
            resolution_list = [df_resolution]
        else:
            resolution_list = [resolution]

    else:  # enable multi-resolution by default
        resolution_list = ["15min", "30min", "60min"]
        if df_resolution > pd.Timedelta(resolution_list[2]):
            resolution_list = [df_resolution]

    best_score = 0.0
    result = (0, 0, False, 0, SILHOUETTE_NEG_INF, None)
    for resolution in resolution_list:
        if df_resolution > pd.Timedelta(resolution):
            continue
        df = down_sample(df, resolution=resolution)
        (
            hour_block_length,
            offset,
            has_weekend,
            offdays_start,
            score,
            label_method,
        ) = _evaluate_seasonality_patterns(df, threshold_silhouette)
        # time_policy = TimePolicy(hour_block_length, offset, has_weekend, offdays_start, score)
        # logger.debug(f'{str(time_policy)}; (resolution={resolution})')

        if score > best_score:
            best_score = score
            result = (
                hour_block_length,
                offset,
                has_weekend,
                offdays_start,
                score,
                label_method,
            )
        if best_score > THRESHOLD_SILHOUETTE_HIGH:
            break  # TODO

    return result


def _evaluate_seasonality_patterns(df, threshold_silhouette=THRESHOLD_SILHOUETTE):
    df_resolution = get_resolution(df)

    # Possible weekly pattern is evaluated by partitioning the time series into subsequences of 1-day long
    #   if the resolution is lower than 6h, with only a few points per each subsequence, the pattern cannot be reliably evaluated 
    if df_resolution < TIMEDELTA_DAY / MIN_POINTS_PER_SUBSEQUENCE:
        time_0 = time.time()
        best_score_week, offdays_start, offset, label_method = check_weekly_pattern(df)
        if is_verbose():
            print(f"==HB== weekly pattern time spent: {time.time() - time_0:.2f}s")
    else:
        best_score_week, offdays_start, offset, label_method = (SILHOUETTE_NEG_INF, 0, 0, None)

    time_0 = time.time()
    hour_block_candidate = evaluate_hour_block(df, df_resolution)
    if is_verbose():
        print(f"==HB== hour-block pattern time spent: {time.time() - time_0:.2f}s")

    # Possible half-day pattern is evaluated by partitioning the time series into subsequences of half-day long
    #   if the resolution is lower than 3h, with only a few points per each subsequence, the pattern cannot be reliably evaluated 
    if df_resolution < TIMESELTA_HALF_DAY / MIN_POINTS_PER_SUBSEQUENCE:
        time_0 = time.time()
        score_half, workhour_start = evaluate_half_day(df)
        if is_verbose():
            print(f"==HB== half-day pattern time spent: {time.time() - time_0:.2f}s")
    else:
        score_half, workhour_start = (SILHOUETTE_NEG_INF, 0)

    if hour_block_candidate is None:
        hour_block_score = SILHOUETTE_NEG_INF
    else:
        hour_block_score = hour_block_candidate.silhouette

    has_weekend = best_score_week > threshold_silhouette
    # ONLY add hour-block or workhour pattern when it improves the score
    if (
        has_weekend
        and best_score_week > hour_block_score
        and best_score_week > score_half
    ):
        return 0, offset, has_weekend, offdays_start, best_score_week, label_method

    else:
        if hour_block_score >= score_half:
            if hour_block_score >= threshold_silhouette:
                offset = hour_block_candidate.offset
                best_score = hour_block_candidate.silhouette
                hour_block_length = hour_block_candidate.hourdelta
            else:
                offset = 0
                best_score = SILHOUETTE_NEG_INF
                hour_block_length = 0
        else:
            offset = workhour_start
            best_score = score_half
            if best_score >= threshold_silhouette:
                hour_block_length = 12
            else:
                hour_block_length = 0
                offset = 0

        return hour_block_length, offset, False, offdays_start, best_score, None

