"""
Reference: https://github.com/numenta/NAB/blob/master/nab/detectors/earthgecko_skyline/algorithms.py
Adapted for batch mode
"""

import math
import numpy as np
import pandas as pd

from saad.algs.adesca.data_prepare import (
    COL_VALUE,
    COL_ANOMALY_SCORE,
    sigmoid,
    get_resolution,
)

COL_VALUE3 = "value3"

# columns of anoamly for each alg:
ANOMALY_MAD = "anomaly_mad"
ANOMALY_HOUR_AVG = "anomaly_hour_avg"
ANOMALY_ROLLING_AVG = "anomaly_rolling_avg"
ANOMALY_EWM_AVG = "anomaly_ewm_avg"
ANOMALY_ROLLING_AVG1 = "anomaly_rolling_avg1"
ANOMALY_HIST_BIN = "anomaly_hist_bin"

ANOMALY_SCORE_MAD = "anomaly_score_mad"
ANOMALY_SCORE_HOUR_AVG = "anomaly_score_hour_avg"
ANOMALY_SCORE_ROLLING_AVG = "anomaly_score_rolling_avg"
ANOMALY_SCORE_EWM_AVG = "anomaly_score_ewm_avg"
ANOMALY_SCORE_ROLLING_AVG1 = "anomaly_score_rolling_avg1"
ANOMALY_SCORE_HIST_BIN = "anomaly_score_hist_bin"

ANOMALY_ENSEMBLE = "anomaly_ensemble"

CONSENSUS = 5
EXPIRATION_TIME = pd.Timedelta(hours=12)

THRESHOLD_MAD = 6.0
THRESHOLD_HOUR_AVG = 3.0
THRESHOLD_ROLLING_AVG = 3.0
THRESHOLD_EWM_AVG = 3.0
THRESHOLD_ROLLING_AVG_NO_SMOOTH = 3.0
THRESHOLD_HIST_BIN = 20

vector_sigmoid = np.vectorize(sigmoid)


def detect_anomaly(df_in, anomaly_score_thresh):
    df = df_in.copy()

    # smoothing with 3-point average to reduce noise
    df[COL_VALUE3] = df[COL_VALUE].rolling(3, center=True).mean()
    df[COL_VALUE3].iloc[0] = df[COL_VALUE3].iloc[1]
    df[COL_VALUE3].iloc[-1] = df[COL_VALUE3].iloc[-2]

    mad(df)
    same_hour_yesterday_avg(df)
    rolling_avg(df)
    ewm_avg(df)
    rolling_avg_no_smooth(df)
    hist_bins(df)

    # anoamly decision by super majority vote
    cnt_positive = (
        df[ANOMALY_MAD].astype(int)
        + df[ANOMALY_HOUR_AVG].astype(int)
        + df[ANOMALY_ROLLING_AVG].astype(int)
        + df[ANOMALY_EWM_AVG].astype(int)
        + df[ANOMALY_ROLLING_AVG1].astype(int)
        + df[ANOMALY_HIST_BIN].astype(int)
    )
    df[ANOMALY_ENSEMBLE] = (cnt_positive >= CONSENSUS).astype(int)

    df[COL_ANOMALY_SCORE] = df[
        [
            ANOMALY_SCORE_MAD,
            ANOMALY_SCORE_HOUR_AVG,
            ANOMALY_SCORE_ROLLING_AVG,
            ANOMALY_SCORE_EWM_AVG,
            ANOMALY_SCORE_ROLLING_AVG1,
            ANOMALY_SCORE_HIST_BIN,
        ]
    ].mean(axis=1)

    df[ANOMALY_ENSEMBLE] = df[ANOMALY_ENSEMBLE] * (
        df[COL_ANOMALY_SCORE] > anomaly_score_thresh
    ).astype(int)

    # reduce FP by expiration
    expired = True
    for idx, anomaly_label in df[ANOMALY_ENSEMBLE].items():
        if anomaly_label == 0:
            continue
        if expired:
            expired = False
            timestamp_start = idx
            continue
        if idx - timestamp_start < EXPIRATION_TIME:
            df[ANOMALY_ENSEMBLE].loc[idx] = 0  # reset anoamly label to reduce FP
        else:
            expired = False
            timestamp_start = idx

    return df


def mad(df):
    series = df[COL_VALUE]
    median = series.median()
    zero_centered_abs = np.abs(series - median)
    median_deviation = zero_centered_abs.median()
    test_stat = zero_centered_abs / median_deviation
    df[ANOMALY_MAD] = test_stat > THRESHOLD_MAD
    df[ANOMALY_SCORE_MAD] = vector_sigmoid(test_stat / THRESHOLD_MAD)


def same_hour_yesterday_avg(df):
    series = df[COL_VALUE]
    rolling = series.rolling(max(pd.Timedelta("60min"), get_resolution(df) * 4))
    rolling_mean_yesterday = rolling.mean().shift(1, freq="D")
    rolling_std_yesterday = rolling.std().shift(1, freq="D")
    test_stat = np.abs(df[COL_VALUE3] - rolling_mean_yesterday) / rolling_std_yesterday
    test_stat = test_stat.loc[df.index]
    df[ANOMALY_HOUR_AVG] = test_stat > THRESHOLD_HOUR_AVG
    df[ANOMALY_SCORE_HOUR_AVG] = vector_sigmoid(test_stat / THRESHOLD_HOUR_AVG)


def rolling_avg(df):
    series = df[COL_VALUE]
    rolling = series.ewm(alpha=0.01)
    rolling_mean = rolling.mean().shift(1)
    rolling_std = rolling.std().shift(1)
    test_stat = np.abs(df[COL_VALUE3] - rolling_mean) / rolling_std
    df[ANOMALY_ROLLING_AVG] = test_stat > THRESHOLD_ROLLING_AVG
    df[ANOMALY_SCORE_ROLLING_AVG] = vector_sigmoid(test_stat / THRESHOLD_ROLLING_AVG)


def ewm_avg(df):
    series = df[COL_VALUE]
    rolling = series.ewm(ignore_na=False, min_periods=0, adjust=True, com=50)
    rolling_mean = rolling.mean().shift(1)
    rolling_std = rolling.std().shift(1)
    test_stat = np.abs(df[COL_VALUE3] - rolling_mean) / rolling_std
    df[ANOMALY_EWM_AVG] = test_stat > THRESHOLD_EWM_AVG
    df[ANOMALY_SCORE_EWM_AVG] = vector_sigmoid(test_stat / THRESHOLD_EWM_AVG)


def rolling_avg_no_smooth(df):
    series = df[COL_VALUE]
    rolling = series.ewm(alpha=0.01)
    rolling_mean = rolling.mean().shift(1)
    rolling_std = rolling.std().shift(1)
    test_stat = (
        np.abs(df[COL_VALUE] - rolling_mean) / rolling_std
    )  # don't use 3 point avg
    df[ANOMALY_ROLLING_AVG1] = test_stat > THRESHOLD_ROLLING_AVG_NO_SMOOTH
    df[ANOMALY_SCORE_ROLLING_AVG1] = vector_sigmoid(
        test_stat / THRESHOLD_ROLLING_AVG_NO_SMOOTH
    )


def hist_bins(df):
    values = df[COL_VALUE].to_numpy()
    values3 = df[COL_VALUE3].to_numpy()

    hist, bins = np.histogram(values, bins=16)

    is_anomaly = np.zeros(len(values), dtype=int)
    anomaly_scores = np.full(len(values), sigmoid(0))
    for idx, t in enumerate(values3):
        if math.isnan(t):
            continue
        bin_idx = np.searchsorted(bins, t)
        if bin_idx == 0:
            bin_idx += 1
        if bin_idx == len(bins):
            bin_idx -= 1
        cnt_pnt_in_bin = hist[bin_idx - 1]
        anomaly_scores[idx] = sigmoid(THRESHOLD_HIST_BIN / (cnt_pnt_in_bin + 1.0))
        if cnt_pnt_in_bin < THRESHOLD_HIST_BIN:
            is_anomaly[idx] = 1

    df[ANOMALY_HIST_BIN] = is_anomaly
    df[ANOMALY_SCORE_HIST_BIN] = anomaly_scores


