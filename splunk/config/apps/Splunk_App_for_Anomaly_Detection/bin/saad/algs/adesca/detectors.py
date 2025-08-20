import numpy as np
import pandas as pd
import time

from saad.algs.adesca.dev_util import (
    is_verbose,
    find_consecutive_ones,
    show_anomaly_score,
)
from saad.algs.adesca.timepolicy import recommend_time_policy
from saad.algs.adesca.itsi_at_threshold import (
    itsi_thresholding, 
    calc_time_series_normal_behavior
)
from saad.algs.adesca.itsi_at_no_pattern import (
    itsi_thresholding as itsi_thresholding_np,
)
from saad.algs.adesca.threshold_utils import (
    calc_anomaly_label,
    eva_post,
    fp_control,
    get_params_by_sensitivity,
    set_output_anomaly_label,
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
    COL_TIMESTAMP,
    COL_VALUE,
    COL_ANOMALY_LABEL as ADESCA_ANOMALY_LABEL,
    OUTPUT_ANOMALY_COL,
    COL_ANOMALY_SCORE,
)
from saad.algs.adesca.eg_skyline_utils import (
    detect_anomaly as skyline_detect,
    ANOMALY_ENSEMBLE,
)
from saad.algs.streamhist.streamhist_util import detect_anomaly as streamhist_detect
from saad.utils import setup_logging

logger = setup_logging.get_logger()
import logging

ENSEMBLE_NONE = "no_ensemble"
ENSEMBLE_SKYLINE = "skyline"
ENSEMBLE_STREAMHIST = "streamhist"


def detect_anomaly(
    df_in,
    ensemble=ENSEMBLE_STREAMHIST,
    augment_datetime=True,
    sensitivity=SENSITIVITY_MID,
    df_name="",
    post_processing=True,
):
    # check degrades special cases, and return early
    if df_in[COL_VALUE].max() == df_in[COL_VALUE].min():
        df_in[OUTPUT_ANOMALY_COL] = np.zeros(len(df_in), dtype=int)
        df_in[COL_ANOMALY_SCORE] = np.zeros(len(df_in), dtype=int)
        return df_in, False

    df = df_in.copy()


    if augment_datetime:
        aug_datetime(df)
        df = df.set_index(COL_DATE)

    time_difference = pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])
    if time_difference < pd.Timedelta(hours=1):
        raise Exception("Anomaly detection requires at least one hour of data. Add more data and try again.")

    df_resolution = get_resolution(df)
    df = down_sample(df, resolution=df_resolution, df_resolution=df_resolution) # don' change resolution but fill na

    def _detect_with_sensitivity(sensitivity):
        param_dict = get_params_by_sensitivity(sensitivity)
        ensemble_switch_score = param_dict[PARAM_ENSEMBLE_SWITCH]

        rslt_df, time_policy = time_policy_and_anomaly_bnd(
            df, df_resolution, param_dict[PARAM_THRESH_ADESCA], ensemble_switch_score, df_name
        )
        detector_anomaly_column = OUTPUT_ANOMALY_COL
        use_adesca = True

        if ensemble != ENSEMBLE_NONE and time_policy.score < ensemble_switch_score: # Not using ADESCA
            use_adesca = False

            if ensemble == ENSEMBLE_SKYLINE:
                rslt_df = skyline_detect(df, param_dict[PARAM_THRESH_ENSEMBLE])
                detector_anomaly_column = ANOMALY_ENSEMBLE

            elif ensemble == ENSEMBLE_STREAMHIST:
                rslt_df = streamhist_detect(df, param_dict[PARAM_THRESH_ENSEMBLE])
                detector_anomaly_column = ANOMALY_ENSEMBLE

        if post_processing and time_policy.score < HIGH_SILHOUETTE_SCORE:
            rslt_df = fp_control(rslt_df, detector_anomaly_column)
            for _ in range(param_dict[PARAM_EVA_CYCLE]):
                rslt_df = eva_post(
                    rslt_df,
                    col_anomaly_label=detector_anomaly_column,
                    anomaly_score_thresh=param_dict[PARAM_THRESH_ENSEMBLE],
                    reset_ev_val=True,
                )

        logger.info(
            f"{setup_logging.ANOMALY_APP_TELEMETRY} Time Policy Score: {time_policy.score}"
        )
        logger.info(
            f"{setup_logging.ANOMALY_APP_TELEMETRY} Seasonal Pattern Detection: {str(time_policy)[:-7]}"
        )
        logger.info(f"{setup_logging.ANOMALY_APP_TELEMETRY} Using ADESCA: {use_adesca}")

        return rslt_df, use_adesca, detector_anomaly_column

    rslt_df, use_adesca, detector_anomaly_column = _detect_with_sensitivity(
        SENSITIVITY_MID
    )  # run the mid sensitivity case as the default
    anomaly_est = rslt_df.loc[rslt_df[detector_anomaly_column] == 1].index.tolist()
    cnt_anomaly_est = len(anomaly_est)

    if sensitivity == SENSITIVITY_MID:
        logger.info(
            f"{setup_logging.ANOMALY_APP_TELEMETRY} Number of anomalies detected: {cnt_anomaly_est}"
        )
        cnt_anomaly_non_continous = find_consecutive_ones(
            rslt_df[detector_anomaly_column]
        )
        logger.info(
            f"{setup_logging.ANOMALY_APP_TELEMETRY} Number of non-continuous anomalies detected: {cnt_anomaly_non_continous}"
        )
        return (
            set_output_anomaly_label(
                df_in, rslt_df, detector_anomaly_column=detector_anomaly_column
            ),
            use_adesca,
        )

    rslt_df_s, use_adesca_s, detector_anomaly_column_s = _detect_with_sensitivity(
        sensitivity
    )
    anomaly_est_s = rslt_df_s.loc[
        rslt_df_s[detector_anomaly_column_s] == 1
    ].index.tolist()
    cnt_anomaly_est_s = len(anomaly_est_s)
    logger.info(
        f"{setup_logging.ANOMALY_APP_TELEMETRY} Number of anomalies detected: {cnt_anomaly_est_s}"
    )
    cnt_anomaly_non_continous = find_consecutive_ones(
        rslt_df_s[detector_anomaly_column_s]
    )
    logger.info(
        f"{setup_logging.ANOMALY_APP_TELEMETRY} Number of non-continuous anomalies detected: {cnt_anomaly_non_continous}"
    )

    # Avoid the counterintuitive case of higher sensitivity cases returning less anomaly points
    if (sensitivity == SENSITIVITY_HIGH and cnt_anomaly_est_s > cnt_anomaly_est) or (
        (sensitivity == SENSITIVITY_LOW and cnt_anomaly_est_s < cnt_anomaly_est)
    ):
        return (
            set_output_anomaly_label(
                df_in, rslt_df_s, detector_anomaly_column=detector_anomaly_column_s
            ),
            use_adesca_s,
        )
    else:
        return (
            set_output_anomaly_label(
                df_in, rslt_df, detector_anomaly_column=detector_anomaly_column
            ),
            use_adesca,
        )

def time_policy_and_anomaly_bnd(
    df_in,
    df_resolution,
    threshold_anomaly_score,
    threshold_no_pattern,
    df_name="",
):
    df = df_in.copy()

    def _time_policy_and_anomaly_bnd(df):
        df_in_values = df[COL_VALUE].copy()
        time_0 = time.time()
        time_policy = recommend_time_policy(df, threshold_no_pattern)
        if is_verbose():
            print(
                f"==HB== {df_name} pattern detected: {str(time_policy)} ({time.time() - time_0:.3f}s)"
            )

        # calcualte anomaly bound based on time policy
        if time_policy.score > threshold_no_pattern:
            history_normal_cluster_dict, subs, label_method = calc_time_series_normal_behavior(df, time_policy)
            sub_length = get_subsequence_length(time_policy, df_resolution)
            df = itsi_thresholding(df, time_policy, history_normal_cluster_dict, sub_length, subs, label_method)
        else:
            df = itsi_thresholding_np(df)

        df[COL_VALUE] = df_in_values
        df = calc_anomaly_label(
            df,
            threshold_anomaly_score,
            post_region=time_policy.score < HIGH_SILHOUETTE_SCORE,
        )

        # if show_anomaly_score():
        #    from eval.notebooks.viz_utils import plot_time_series_with_anomaly_bound_score
        #    plot_time_series_with_anomaly_bound_score(df, df_name, time_policy)

        return df, time_policy

    df, time_policy = _time_policy_and_anomaly_bnd(df)
    return df, time_policy
