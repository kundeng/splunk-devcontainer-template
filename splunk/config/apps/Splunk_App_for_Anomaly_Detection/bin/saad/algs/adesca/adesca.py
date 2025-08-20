from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

from saad.algs.adesca.dev_util import (
    is_verbose,
    find_consecutive_ones,
    show_anomaly_score,
)
from saad.algs.adesca.timepolicy import recommend_time_policy
from saad.algs.adesca.itsi_at_threshold import itsi_thresholding, calc_time_series_normal_behavior
from saad.algs.adesca.itsi_at_no_pattern import (
    itsi_thresholding as itsi_thresholding_np,
)
from saad.algs.adesca.threshold_utils import (
    calc_anomaly_label,
    set_output_anomaly_label,
    eva_post,
    fp_control,
    get_params_by_sensitivity,
    get_subsequence_length,
    SENSITIVITY_MID,
    SENSITIVITY_HIGH,
    SENSITIVITY_LOW,
    PARAM_ENSEMBLE_SWITCH,
    PARAM_EVA_CYCLE,
    PARAM_THRESH_ADESCA,
    PARAM_THRESH_ENSEMBLE,
    HIGH_SILHOUETTE_SCORE,
)
from saad.algs.adesca.data_prepare import (
    aug_datetime,
    get_resolution,
    down_sample,
    sigmoid,
    COL_DATE,
    COL_DAY_OF_WEEK,
    COL_HOUR,
    COL_TIMESTAMP,
    COL_VALUE,
    COL_ANOMALY_LABEL as ADESCA_ANOMALY_LABEL,
    OUTPUT_ANOMALY_COL,
    COL_ANOMALY_SCORE,
)
from saad.utils import setup_logging

logger = setup_logging.get_logger()

class AdescaException(Exception):
    pass

class AdescaWarning(Warning):
    pass

@dataclass
class TimeSeriesInfo:
    name            : str
    datasource      : str
    count_points    : int
    resolution      : pd.Timedelta
    timestamp_first : pd.Timestamp
    timestamp_last  : pd.Timestamp

@dataclass
class TrainInfo:
    train_data  : TimeSeriesInfo
    start_time  : pd.Timestamp
    end_time    : pd.Timestamp
    # TODO: Add result code or result status?

THRESH_NO_PATTERN = 0.2
THRESH_ADESCA_SCORE = 0.8

# THRESH_NO_PATTERN = 0.1
# THRESH_ADESCA_SCORE = 0.626

class ADESCA:
    def __init__(self) -> None:
        self.fitted = False
        self.latest_train = None

    def fit(self, 
            time_series, 
            time_series_name='',
            time_series_datasource='',
            augment_datetime=True,
            threshold_no_pattern=THRESH_NO_PATTERN
    ):
        start_time = pd.Timestamp.now()

        df = time_series.copy()
        if augment_datetime:
            aug_datetime(df)
            df = df.set_index(COL_DATE)
        df_resolution = get_resolution(df)

        def _train_info():
            return TrainInfo(
                TimeSeriesInfo(
                    time_series_name,
                    time_series_datasource,
                    df.shape[0],
                    df_resolution,
                    df.index[0],
                    df.index[-1]
                ),
                start_time,
                pd.Timestamp.now()
            )

        # check the special case of constant time series
        if time_series[COL_VALUE].max() == time_series[COL_VALUE].min():
            self.fitted = True
            self.constant_ts = True
            self.df_resolution = df_resolution
            # Save TrainInfo
            self.latest_train = _train_info()
            return
        
        self.constant_ts = False

        # don' change resolution but fill na, and 
        # sync the timestamps to resolution, and 
        # count-of-points may be slightly different, and
        # non-numerical columns dropped
        # TODO: add bookkeeping for missing values, 
        #       in order to disable reporting anomalies on them later
        df = down_sample(df, resolution=df_resolution, df_resolution=df_resolution) 
        # TODO: improve down_sample, instead of put the following here:
        df[COL_DAY_OF_WEEK] = df[COL_DAY_OF_WEEK].astype("uint8")
        df[COL_HOUR] = df[COL_HOUR].astype("uint8")

        # TODO move pre-processing (remove-super-spikes, and possible log-transform)
        #       here, to make them explicit
        time_policy = recommend_time_policy(df, threshold_no_pattern) 
        history_normal = None
        if time_policy.score > threshold_no_pattern:
            history_normal, _, _ = calc_time_series_normal_behavior(df, time_policy)

        # Save state
        self.fitted = True
        self.time_policy = time_policy
        self.history_normal = history_normal
        self.df_resolution = df_resolution
        # Save TrainInfo
        # The latest_train info will be verified by following calling of the detect() API.
        # Also, latest_train info is useful for model update logic that will be added.
        self.latest_train = _train_info()


    def detect(self,
            time_series, 
            time_series_name='',
            time_series_datasource='',
            augment_datetime=True,
            threshold_no_pattern=THRESH_NO_PATTERN,
            threshold_anomaly_score=THRESH_ADESCA_SCORE,
            detector_anomaly_column = OUTPUT_ANOMALY_COL
    ):
        if not self.fitted:
            raise AdescaException("calling detect() before fit()")
        
        if time_series_name != self.latest_train.train_data.name:
            warnings.warn(
                'time_series_name in detect() is not the same to latest_train', 
                AdescaWarning)

        if time_series_datasource != self.latest_train.train_data.datasource:
            warnings.warn(
                'time_series_datasource in detect() is not the same to latest_train', 
                AdescaWarning)
            
        if self.constant_ts:
            time_series[OUTPUT_ANOMALY_COL] = np.zeros(len(time_series), dtype=int)
            time_series[COL_ANOMALY_SCORE] = np.zeros(len(time_series), dtype=int)
            return time_series, False

        df = time_series.copy()

        if augment_datetime:
            aug_datetime(df)
            df = df.set_index(COL_DATE)

        df = down_sample(df, resolution=self.df_resolution, df_resolution=self.df_resolution) 
        # TODO: improve down_sample, instead of put the following here:
        df[COL_DAY_OF_WEEK] = df[COL_DAY_OF_WEEK].astype("uint8")
        df[COL_HOUR] = df[COL_HOUR].astype("uint8")

        if self.time_policy.score  > threshold_no_pattern:
            sub_length = get_subsequence_length(self.time_policy, self.df_resolution)
            df = itsi_thresholding(
                df, 
                self.time_policy, 
                self.history_normal,
                sub_length,
                )
        else:
            df = itsi_thresholding_np(df)

        df = calc_anomaly_label(
            df,
            threshold_anomaly_score,
            post_region=self.time_policy.score < HIGH_SILHOUETTE_SCORE,
        )

        # if show_anomaly_score():
        #    from eval.notebooks.viz_utils import plot_time_series_with_anomaly_bound_score
        #    plot_time_series_with_anomaly_bound_score(df, time_series_name, self.time_policy, showLegend=False, y_limit=[16, 90])

        return (
            set_output_anomaly_label(
                time_series, df, detector_anomaly_column=detector_anomaly_column
            ),
            True,
        )
    