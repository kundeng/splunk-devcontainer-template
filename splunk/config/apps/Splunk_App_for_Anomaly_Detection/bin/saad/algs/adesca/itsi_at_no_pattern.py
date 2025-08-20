from datetime import timedelta
import numpy as np

from saad.algs.adesca.data_prepare import (
    COL_VALUE,
    COL_BND_LOW,
    COL_BND_UP,
    COL_EDGE_MASK,
)
from saad.algs.adesca.data_prepare import calc_iqr, IQR_FACTOR
from saad.algs.adesca.sub_sequence import df_to_hour_sequences, df_to_day_sequences

DEFAULT_Z = 3.3


def itsi_thresholding(df):
    """This function complete the ADESCA detector by covering the no-pattern case with a simple std method
    The input time series is partitioned into 1-day or 1-hour subsequences, based on the time series length
    Default Z score is used to calculate anomaly boundary per each subsequence

    Args:
        df (pandas Dataframe): time series dataframe with value column and timestamp as index

    Returns:
        pandas Dataframe: time series dataframe augmented with anomaly boundary columns
    """
    df_iqr = calc_iqr(df[COL_VALUE].to_numpy())

    if df.index[-1] - df.index[0] <= timedelta(days=3):
        subs = df_to_hour_sequences(df)
    else:
        subs = df_to_day_sequences(df)

    subs_total_len = sum(
        [s.length for s in subs]
    )  # subs may have diffferent length due to possible missing values
    bnd_up = np.empty(subs_total_len)
    bnd_low = np.empty(subs_total_len)

    idx = 0
    for sub in subs:
        threshold = get_thresholds(sub.values, df_iqr=df_iqr)
        bnd_up[idx : idx + sub.length] = threshold[0]
        bnd_low[idx : idx + sub.length] = threshold[1]
        idx += sub.length


    df[COL_BND_UP] = bnd_up
    df[COL_BND_LOW] = bnd_low
    df[COL_EDGE_MASK] = np.ones(df.shape[0], dtype=int)

    return df


def get_thresholds(values, level=DEFAULT_Z, df_iqr=0.0):
    mid = np.mean(values)
    std = max(np.std(values), df_iqr * IQR_FACTOR) 
    variation = std * level

    return mid + variation, mid - variation
