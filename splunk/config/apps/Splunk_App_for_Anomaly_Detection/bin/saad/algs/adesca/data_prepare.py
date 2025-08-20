import csv
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.special import expit
from packaging import version

from saad.algs.adesca.dev_util import is_verbose
from saad.utils.parsing_utils import datetime_is_tz_aware, infer_timestamp_format

pd.options.mode.chained_assignment = None

EPSILON = 1e-7

COL_TIMESTAMP = "timestamp"
COL_DATE = "date"
COL_VALUE = "value"
COL_HOUR = "HourOfDay"
COL_DAY_OF_WEEK = "DayOfWeek"

OUTPUT_ANOMALY_COL = "anomaly_label"

DAY_OF_WEEK_NAME = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

NAB_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ITSI_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.000%z"
ITSI_TIMESTAMP_FORMAT_PD = "%Y-%m-%d %H:%M:%S%z"

COL_BND_LOW = "bnd_low"
COL_BND_UP = "bnd_up"

COL_ANOMALY_LABEL = "anomaly_label"  # binary 0-1 label
COL_ANOMALY_SCORE = "anomaly_score"  # anomaly score express as the distance between a data point to the anomaly boundary

COL_EDGE_MASK = (
    "edge_mask"  # binary 0-1 mask to mitigate false alarms at the edge of segments
)

USE_ROBUST_MEAN_STD = False
ROBUST_MEAN_STD_CLIP_RATIO = 0.15

IQR_FACTOR = 0.05

THRESH_SUPER_SPIKE = 12.0
THRESH_SUPER_SPIKE_REGULARITY = 4


def parse_timestamp(df, col_time=COL_TIMESTAMP):
    sample_ts = str(df[col_time].iloc[0])
    ts_format = infer_timestamp_format(sample_ts)

    if ts_format == 'datetime_str':
        # If the timestamp is timezone-aware, convert to UTC. Otherwise, leave it as is. Specifying the `utc` argument is required to avoid downsteam errors.
        tz_aware = datetime_is_tz_aware(pd.to_datetime(sample_ts, infer_datetime_format=True))
        return pd.to_datetime(df[col_time], utc=tz_aware, infer_datetime_format=True)

    else: # Numeric (unix seconds (`s`) or milliseconds (`ms`))
        return pd.to_datetime(df[col_time], utc=True, unit=ts_format)

def aug_datetime(df, drop_raw_timestamp = True):
    if df.empty:
        return
        
    df[COL_DATE] = parse_timestamp(df)

    if drop_raw_timestamp:
        df.drop(COL_TIMESTAMP, axis=1, inplace=True)
    df[COL_DAY_OF_WEEK] = df[COL_DATE].dt.dayofweek
    df[COL_HOUR] = df[COL_DATE].dt.hour
    df.dropna(inplace=True)  # TODO: try to remove the dropna
    df[COL_DAY_OF_WEEK] = df[COL_DAY_OF_WEEK].astype("uint8")
    df[COL_HOUR] = df[COL_HOUR].astype("uint8")


def get_resolution(df):
    # This function estimates the resolution of a time series (stored as a pandas dataframe)
    # The resolution is estiamted as the most frequent item in the list of timestamp diff
    # 
    # Prerequisite: the index of the dataframe is the timestamps of the time series

    ts_diff = df.index[: min(df.shape[0], 1000)].to_series().diff()
    frequent_timestamp_delta_sorted = ts_diff.value_counts()

    if frequent_timestamp_delta_sorted.index[0] == pd.Timedelta(seconds=0):
        # Handle the unusual case of time series containing many datapoints with SAME timestamps
        #   such that the most frequent item in the list of timestamp-diff is 0
        assert len(frequent_timestamp_delta_sorted) > 1 # assert datapoints have at least 2 different timestamps
        return frequent_timestamp_delta_sorted.index[1] # use the 2nd most frequent timestamp-diff
    else:
        return frequent_timestamp_delta_sorted.index[0]


def down_sample(df, resolution="15min", df_resolution=None):
    """
    The down_sample function only modify the df if the target resolution is NOT finer than the df_resolution.
    For example: When the target resolution is 1h:
        if the df_resolution is 2h, the down_sample does not modify the dataset;
        if the df_resolution is 30min, the dataset will be modified;
        if the df_resolution is also 1h, the pandas resample API will be called; as a result, the dataset may be modified. If the input df contains missing values, those missing values will be filled by linear interpolation.
    
    resolution is 'rule' supported by https://pandas.pydata.org/docs/reference/api/pandas.Series.resample.html
      examples: 3min or 3T : for resolution of 3 minutes
                 30S        : for resolution of 30 seconds
    """

    if not df_resolution:
        df_resolution = get_resolution(df)
    if df_resolution > (
        pd.Timedelta(resolution) if isinstance(resolution, str) else resolution
    ):
        if is_verbose():
            print(
                f"==HB== The downsampling target resolution of {resolution} is finer than the resolution of the input time series of {df_resolution}; no change is made to the resolution of the input time series."
            )
        return df

    df_resampled = df.resample(resolution).mean().interpolate() # Note: non-numeric columns are lost after resample
    return df_resampled


# remove trend on the basis of day median, so that
#   trend from day to day are removed, but differences among hours in a day is kept
def detrend_daily(df, offset=0):
    rslt = df.copy()
    overall_med = df[
        COL_VALUE
    ].median()  # use median to make it robust to extreme values
    dfre = resample_with_offset(df, '24h', offset)
    for _, sub_df in dfre:
        med = sub_df[COL_VALUE].median()
        rslt[COL_VALUE].loc[sub_df.index] = sub_df[COL_VALUE].map(
            lambda x: x - med + overall_med
        )
    return rslt

def resample_with_offset(df, resolution, offset_hours):
    pandas_version = pd.__version__
    if version.parse(pandas_version) < version.parse('1.1.0'):
        dfre = df.resample(resolution, base=offset_hours)
    else:
        dfre = df.resample(resolution, offset=f'{offset_hours}h')
    return dfre

# convert anomaly labels represented as strings to datetime
def anomaly_labels_to_datetime(anomaly_labels, timestamp_format):
    anomaly_datetime = []
    for anomaly in anomaly_labels:
        if isinstance(anomaly, tuple):  # anomaly region
            anomaly_datetime.append(
                [datetime.strptime(endpoint, timestamp_format) for endpoint in anomaly]
            )
        else:
            anomaly_datetime.append(datetime.strptime(anomaly, timestamp_format))
    return anomaly_datetime


# get the anomaly timestamps of a df, where anomaly region is represented by its two end points
def get_detected_anomalies_timestamp(df):
    anomaly_labels_runlength = get_label_run_length(np.array(df[COL_ANOMALY_LABEL]))
    anomalies_timestamp = []
    i = 0
    while i < len(anomaly_labels_runlength):
        if anomaly_labels_runlength[i] == 0:
            i += 1
            continue
        if anomaly_labels_runlength[i] == 1:
            anomalies_timestamp.append(df.index[i])
            i += 1
            continue
        # runlength >1: anomaly region
        anomalies_timestamp.append(
            [df.index[i], df.index[i + anomaly_labels_runlength[i] - 1]]
        )
        i += anomaly_labels_runlength[i] + 1

    return anomalies_timestamp


# convert the 0-1 anomaly label to 0-n, where n is the run length of an anomaly region
def get_label_run_length(anomaly_labels):
    anomaly_label_runlength = np.zeros_like(anomaly_labels)
    idx_pre = -1
    for idx, label in enumerate(anomaly_labels):
        if label == 0:
            if idx_pre >= 0:
                anomaly_label_runlength[idx_pre:idx] = idx - idx_pre
                idx_pre = -1
        else:  # label == 1
            if idx_pre < 0:
                idx_pre = idx
    return anomaly_label_runlength


def mean_std_1d(values, robust=False):
    if robust and USE_ROBUST_MEAN_STD:
        values = sorted(values)
        n = len(values)
        values = values[
            int(n * ROBUST_MEAN_STD_CLIP_RATIO) : int(
                n * (1 - ROBUST_MEAN_STD_CLIP_RATIO)
            )
        ]

    return np.mean(values), np.std(values)


def mean_std_2d(values, robust=False):
    """
    calcualte the mean and std for each row of the input 2d array
    """
    shape = values.shape
    assert len(shape) == 2 and shape[0] > 1 and shape[1] > 1
    if not (robust and USE_ROBUST_MEAN_STD):
        return np.mean(values, axis=1), np.std(values, axis=1)

    n_row = shape[0]
    rslt_mean = np.empty(n_row)
    rslt_std = np.empty(n_row)
    for i in range(n_row):
        rslt_mean[i], rslt_std[i] = mean_std_1d(values[i, :], True)
    return rslt_mean, rslt_std


def calc_iqr(values):
    """
    Calcualte the iqr with two fallbacks, in case majority of the values are equal
    """
    df_iqr = iqr(values)
    if df_iqr == 0.0:
        df_iqr = (np.quantile(values, 0.9) - np.quantile(values, 0.1)) * 0.6
    if df_iqr == 0.0:
        df_iqr = (values.max() - values.min()) * 0.4 + EPSILON
    return df_iqr


def sigmoid(x, trans=1):
    """
    Use the sigmoid function to transfer x to [0,1]
    Use the default value 1 for trans to make sure x=1 map to 0.5
    """
    # Use the expit function from scipy, b/c it is more robust and efficient:
    # https://stackoverflow.com/questions/21106134/numpy-pure-functions-for-performance-caching
    return expit(x - trans)


def remove_super_spikes(df):
    """
    As a pre-processing, remove super high spikes, so that the calculation of thresholds are not distorted by them
    """
    mid = df[COL_VALUE].median()
    std = df[COL_VALUE].std()
    deviation_normalized = np.abs(df[COL_VALUE] - mid) / std

    super_spikes = df[deviation_normalized > THRESH_SUPER_SPIKE]
    half_day = pd.Timedelta(hours=12)

    df_resolution = get_resolution(df)

    def _count_high_spikes_same_hour_other_day(idx_in):
        cnt = 0
        # Here the type of the difference between two timestamp index is pandas.Timedelta.
        # There is no well-defined 'abs' and the negative case is also tricky:
        # https://stackoverflow.com/questions/31836788/subtracting-pandas-timestamps-absolute-value
        other_spikes = super_spikes.loc[
            # select rows at least half-day away and also with same value for hour-of-day
            ((super_spikes.index - idx_in > half_day) | (idx_in - super_spikes.index > half_day)) &
            (super_spikes.index.hour == idx_in.hour)
            ]
        if len(other_spikes) == 0:
            return cnt

        cnt = 1
        other_spikes_idx = other_spikes.index
        idx_pre = other_spikes_idx[0]
        for idx in other_spikes_idx:
            if idx - idx_pre > df_resolution: # consecutive points do NOT count as multiple spikes
                cnt += 1
            idx_pre = idx

        return cnt

    for idx in super_spikes.index:
        if _count_high_spikes_same_hour_other_day(idx) < THRESH_SUPER_SPIKE_REGULARITY:
            # Only remove the super spikes that are unlikely to happen regularly on multiple days
            df[COL_VALUE].loc[idx] = mid

    return df
