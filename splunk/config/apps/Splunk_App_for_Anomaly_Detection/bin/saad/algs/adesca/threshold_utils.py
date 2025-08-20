from collections import defaultdict
from functools import partial
from itertools import groupby
import numpy as np
import pandas as pd

from saad.algs.adesca.data_prepare import (
    get_label_run_length,
    get_resolution,
    sigmoid,
    parse_timestamp,
    COL_VALUE,
    COL_DATE,
    COL_TIMESTAMP,
    COL_BND_LOW,
    COL_BND_UP,
    COL_EDGE_MASK,
    COL_ANOMALY_LABEL,
    COL_ANOMALY_SCORE,
    OUTPUT_ANOMALY_COL,
)
from saad.algs.adesca.pattern_evaluate import (
    evaluate_clustering_quality,
    LABEL_WEEKLY_OFFDAYS,
    LABEL_ALTER_HOUR_BLOCK,
)
from saad.algs.adesca.sub_sequence import (
    df_to_day_sequences,
    df_to_half_day_sequences,
    df_to_hour_sequences,
)

from saad.utils import setup_logging

logger = setup_logging.get_logger()

PARAM_ENSEMBLE_SWITCH = "ensemble_switch_score"
PARAM_EVA_CYCLE = "eva_post_cycle"
PARAM_THRESH_ADESCA = "thresh_anomaly_score_adesca"
PARAM_THRESH_ENSEMBLE = "thresh_anomaly_score_ensemble"

SENSITIVITY_MID = 1  # the default sensitivity
SENSITIVITY_LOW = 0
SENSITIVITY_HIGH = 2

EVA_Z_UP_THRESH = 2.0
EVA_Z_LOW_THRESH = 3.0
EVA_MARGIN_RATIO = 0.05
EVA_GLOBAL_CNT_THRESH = 4
EVA_BUFFER_SIZE = 32

FP_CONTROL_COUNT_THRESH = 4

HIGH_SILHOUETTE_SCORE = 2.7


def get_params_by_sensitivity(sensitivity):
    """The sensitivity switch is implemented by switching to three different sets of values for parameters of the adesca-ensemble detectors.
    The values of the parameters are found using Bayesian optimizer (in the notebook eval/notebooks/ensemble_eval.ipynb).
    The idea is to keep the overall f1 score close, and for different sensitivities change the trade-off of recall vs precision.
    """
    if sensitivity == SENSITIVITY_HIGH:
        return {
            PARAM_ENSEMBLE_SWITCH: 0.25,
            PARAM_EVA_CYCLE: 6,
            PARAM_THRESH_ADESCA: 0.75,
            PARAM_THRESH_ENSEMBLE: 0.5,
        }
    if sensitivity == SENSITIVITY_LOW:
        return {
            PARAM_ENSEMBLE_SWITCH: 0.7,
            PARAM_EVA_CYCLE: 6,
            PARAM_THRESH_ADESCA: 0.75,
            PARAM_THRESH_ENSEMBLE: 0.8,
        }
    # the default SENSITIVITY_MID
    return {
        PARAM_ENSEMBLE_SWITCH: 0.45,
        PARAM_EVA_CYCLE: 2,
        PARAM_THRESH_ADESCA: 0.8,
        PARAM_THRESH_ENSEMBLE: 0.5,
    }


def reduce_anomaly_region(anomaly_labels, anomaly_scores):
    """Reduce regions of continuously anomaly points into single points
    
    The primary objective of the reduce_anomaly_region() function is to synchronize the detector's 
    behavior with the performance metric used for evaluation. For instance, correct identification of 
    an anomaly region would be counted as one true positive, whereas an incorrect report would result 
    in multiple false positive points. The reduce_anomaly_region() function mitigates this issue.

    Args:
        anomaly_labels (numpy.ndarray): 1/0 int array of anomaly labels
        anomaly_scores (numpy.ndarray): float array of anomaly scores

    Returns:
        numpy.ndarray: 1/0 int array of modified anomaly labels with continuous anomaly points reduced
    """
    anomaly_labels_runlength = get_label_run_length(anomaly_labels)

    i = 0
    while i < anomaly_labels.shape[0]:
        label = anomaly_labels[i]
        run_length = anomaly_labels_runlength[i]

        if label == 0:
            i += 1
            continue
        if run_length < 2:
            i += 1
            continue

        # for a segment of continuous anomaly points:
        # first find the max anomaly score in this segment,
        # then find its neighbors in the range between max and q90
        #
        # use the center of these neighbors as the single anomaly point to report, and
        # reset anomaly labels for all other points
        segment = anomaly_scores[i : i + run_length]
        idx_max = np.argmax(segment)
        q90 = np.quantile(segment, 0.9)
        left = idx_max
        while left > 0 and segment[left] >= q90:
            left -= 1
        right = idx_max
        while right < run_length and segment[right] >= q90:
            right += 1
        idx_max = (left + right) // 2

        # reset anomaly labels for all points in this segment, except for one point
        anomaly_labels[i : i + run_length] = 0
        anomaly_labels[i + idx_max] = 1
        i += run_length

    return anomaly_labels


def fp_control(df, col_anomaly_label=COL_ANOMALY_LABEL):
    """False Positive control of anomaly detection by checking timestamps of anomaly points

    Differences between anomaly point timestamps are calculated, and then quantized to hour resolution;
    Unique values and counts of the quantized timestamp diff is used, heuristically, to 
        Decide whether or not the detected anomaly points are actually happening rather regularly.
        If yes, the anomaly labels of these points will be reset. 

    Args:
        df (pandas Dataframe): Dataframe containing results of anomaly detection in column col_anomaly_label
        col_anomaly_label (string, optional): Column name of anomaly label. Defaults to COL_ANOMALY_LABEL.

    Returns:
        pandas Dataframe: Dataframe containing results of anomaly detection, with col_anomaly_label modified to reduce FP
    """

    anomaly_idx = df.loc[df[col_anomaly_label] == 1].index
    if len(anomaly_idx) > FP_CONTROL_COUNT_THRESH:
        idx_sec = (anomaly_idx.astype(np.int64) // 10**9).to_numpy()
        idx_diff = np.diff(idx_sec) // (12 * 3600)  # hour resolution
        unique, counts = np.unique(idx_diff, return_counts=True)
        irregular_diff = set(unique[np.where(counts < FP_CONTROL_COUNT_THRESH)])
        for dif in unique:
            if dif in irregular_diff:
                continue
            df[col_anomaly_label].loc[
                anomaly_idx[np.where(idx_diff == dif)[0] + 1]
            ] = 0  # reset anomaly label for those regularly spaced
    return df


def eva_post(
    df,
    anomaly_score_thresh,
    col_anomaly_label=COL_ANOMALY_LABEL,
    reset_ev_val=False,
    z_up_thresh=EVA_Z_UP_THRESH,
    z_low_thresh=EVA_Z_LOW_THRESH,
    margin_ratio=EVA_MARGIN_RATIO,
    global_cnt_thresh=EVA_GLOBAL_CNT_THRESH,
    buffer_size=EVA_BUFFER_SIZE,
):
    """As a post-processing, detect Extreme Value Anomaly (EVA) in case upstream detectors have missed

    Max an min values of the time series are extract as candidates for anomaly, then
    Candidates are checked both globally and against their left or right neighbors to make anomaly detection decisions.

    Args:
        df (pandas Dataframe): the dataframe containing a time series
        anomaly_score_thresh (float): float number between 0 and 1, the threshold used to convert anomaly score to binary decision
        col_anomaly_label (string, optional): column name for anomaly label. Defaults to COL_ANOMALY_LABEL.
        reset_ev_val (bool, optional): whether to reset extreme values. Defaults to False.
        z_up_thresh (float, optional): z score threshold to check max EVs. Defaults to EVA_Z_UP_THRESH.
        z_low_thresh (float, optional): z score threshold to check min EVs. Defaults to EVA_Z_LOW_THRESH.
        margin_ratio (float, optional): float number between 0 and 1, the ratio to locate the close neighbor of an EV. Defaults to EVA_MARGIN_RATIO.
        global_cnt_thresh (int, optional): the threshold to decide whether or not, globally, the time series containing values close to an EV. Defaults to EVA_GLOBAL_CNT_THRESH.

    Returns:
        _type_: _description_
    """
    df_resoluton = get_resolution(df)
    df_median = df[COL_VALUE].median()

    def _is_outstanding(ev_idx, ev_val, threshold_z):
        def _left_right_mean_std():
            left = max(ev_idx - df_resoluton * buffer_size, df.index[0])
            right = min(ev_idx + df_resoluton * buffer_size, df.index[-1])

            left_win = df[COL_VALUE].loc[left:ev_idx]
            left_mean, left_std = left_win.mean(), left_win.std()

            right_win = df[COL_VALUE].loc[ev_idx:right]
            right_mean, right_std = right_win.mean(), right_win.std()

            return left_mean, left_std, right_mean, right_std

        def _half_outstanding(mean, std):
            if std == 0.0:
                return False, sigmoid(0)
            score = abs(ev_val - mean) / std / threshold_z
            score = sigmoid(score)
            return score > anomaly_score_thresh, score

        left_mean, left_std, right_mean, right_std = _left_right_mean_std()
        left_outstanding, left_score = _half_outstanding(left_mean, left_std)
        right_outstanding, right_score = _half_outstanding(right_mean, right_std)

        margin = (ev_val - df_median) * margin_ratio
        if margin > 0:
            upper = ev_val
            lower = ev_val - margin
        else:
            lower = ev_val
            upper = ev_val - margin

        if not (
            left_outstanding or right_outstanding
        ):  # move ev to the beginning/ending of the ev-region
            ev_iloc = df.index.get_loc(ev_idx)
            i = ev_iloc - 1
            while i >= 0 and (lower <= df[COL_VALUE].iloc[i] <= upper):
                i -= 1
            left_to_edge = False if i >= 0 else True
            left_delta = ev_iloc - i

            i = ev_iloc + 1
            while i < len(df) and (lower <= df[COL_VALUE].iloc[i] <= upper):
                i += 1
            right_to_edge = False if i < len(df) else True
            right_delta = i - ev_iloc

            if left_to_edge:
                ev_iloc += right_delta - 1
            elif right_to_edge:
                ev_iloc -= left_delta
            else:
                if left_delta < right_delta:
                    ev_iloc -= left_delta
                else:
                    ev_iloc += right_delta - 1

            ev_idx = df.index[ev_iloc]
            # re-evaluate left/right outstanding wrt end of ev-region
            left_mean, left_std, right_mean, right_std = _left_right_mean_std()
            left_outstanding, left_score = _half_outstanding(left_mean, left_std)
            right_outstanding, right_score = _half_outstanding(right_mean, right_std)

        def _global_outstanding():
            ev_iloc = df.index.get_loc(ev_idx)
            cnt = len(df[(df[COL_VALUE] >= lower) & (df[COL_VALUE] <= upper)])
            if not left_outstanding:
                i = ev_iloc - 1
                while (
                    i >= 0 and cnt >= global_cnt_thresh
                ):  # not count closest neighbors
                    if lower <= df[COL_VALUE].iloc[i] <= upper:
                        cnt -= 1
                        i -= 1
                    else:
                        break
            if not right_outstanding:
                i = ev_iloc + 1
                while (
                    i < len(df) and cnt >= global_cnt_thresh
                ):  # not count closest neighbors
                    if lower <= df[COL_VALUE].iloc[i] <= upper:
                        cnt -= 1
                        i += 1
                    else:
                        break

            return cnt < global_cnt_thresh

        global_outstanding = (
            False
            if not (left_outstanding or right_outstanding)
            else _global_outstanding()
        )
        outstanding = (left_outstanding or right_outstanding) and global_outstanding

        if reset_ev_val:
            ev_iloc = df.index.get_loc(ev_idx)
            df[COL_VALUE].iloc[
                max(ev_iloc - buffer_size, 0) : min(ev_iloc + buffer_size, len(df))
            ] = df_median

        return outstanding, max(left_score, right_score)

    max_idx, max_val = df[COL_VALUE].idxmax(), df[COL_VALUE].max()
    if df[col_anomaly_label].loc[max_idx] == 0:
        label, score = _is_outstanding(max_idx, max_val, z_up_thresh)
        if label:
            df[col_anomaly_label].loc[max_idx] = 1
            df[COL_ANOMALY_SCORE].loc[max_idx] = score

    min_idx, min_val = df[COL_VALUE].idxmin(), df[COL_VALUE].min()
    if df[col_anomaly_label].loc[min_idx] == 0:
        label, score = _is_outstanding(min_idx, min_val, z_low_thresh)
        if label:
            df[col_anomaly_label].loc[min_idx] = 1
            df[COL_ANOMALY_SCORE].loc[min_idx] = score

    return df


def relative_dist_to_score(val, bnd_low, bnd_up):
    # transfer value to a non-negative number :
    #       0       if val==mid
    #       1       if val==bnd_up or val==bnd_low
    #       > 1     if val beyond bnd
    mid = (bnd_up + bnd_low) / 2.0
    wide = (bnd_up - bnd_low) / 2.0
    if wide == 0.0:
        return sigmoid(-1.0)
    # transfer to [0, 1] using sigmoid function
    return sigmoid(abs(val - mid) / wide)


# generate anomaly label for df with anomaly boundary
def calc_anomaly_label(df, threshold_anomaly_score, post_region=True):
    anomaly_scores = np.vectorize(relative_dist_to_score)(
        df[COL_VALUE].to_numpy(),
        df[COL_BND_LOW].to_numpy(),
        df[COL_BND_UP].to_numpy(),
    )

    df[COL_ANOMALY_SCORE] = anomaly_scores

    # disable post_anomaly_region(..) when (for example) the detector has high confidence, such that
    #       the whole region will be reported as anomalies
    anomaly_labels = (
        anomaly_scores * np.array(df[COL_EDGE_MASK], dtype=int)
        > threshold_anomaly_score
    ).astype(int)
    if post_region:
        anomaly_labels = reduce_anomaly_region(anomaly_labels, anomaly_scores)

    df[COL_ANOMALY_LABEL] = anomaly_labels

    return df


def sub_sequences_groupby(subs, get_seq_label):
    groups = defaultdict(list)
    for label, sub_grouper in groupby(subs, get_seq_label):
        groups[label] += list(sub_grouper)

    return groups


def subsequence_by_time_policy(df, time_policy, is_fit=True):
    if time_policy.label_method is not None:
        label_method = time_policy.label_method
        subs = df_to_day_sequences(df, offset=time_policy.offset)

    elif (
        time_policy.hour_block_length >= 1 and time_policy.hour_block_length < 12
    ):  # hour-block pattern
        label_method = partial(
            LABEL_ALTER_HOUR_BLOCK, time_policy.hour_block_length, time_policy.offset
        )
        subs = df_to_hour_sequences(
            df, time_policy.hour_block_length, time_policy.offset
        )
    elif time_policy.hour_block_length == 12:  # half-day pattern
        label_method = partial(
            LABEL_ALTER_HOUR_BLOCK, time_policy.hour_block_length, time_policy.offset
        )
        subs = df_to_half_day_sequences(df, start_hour=time_policy.offset)
    elif time_policy.has_weekend:  # weekly pattern
        label_method = partial(LABEL_WEEKLY_OFFDAYS, time_policy.offdays_start)
        subs = df_to_day_sequences(df, offset=time_policy.offset)
    else:  # default : 1-hour block
        label_method = partial(LABEL_ALTER_HOUR_BLOCK, 1, 0)
        subs = df_to_hour_sequences(df)

    if is_fit: #TODO: only needed for fit
        _, subs = evaluate_clustering_quality(subs, label_method, kee_all_sequences=True) 

    return subs, label_method

def get_subsequence_length(time_policy, df_resolution):
    if time_policy.has_weekend:
        return int(pd.Timedelta('1d') / df_resolution)
    return int(pd.Timedelta(hours=time_policy.hour_block_length) / df_resolution)

def get_edge_length():
    """The purpose of this function to to calculate the length of the edge for subsequences.
    The cause of the edge effect of subsequences is the hour-grid limitation -- 
        activity changes of time series may not align with the hour-grid.
    To address this issue edge mask is applied to disable reporting anomalies at the edges.

    I tried the logic to calcualte edge length from df_resolution and subsequence length; however,
    the constant 2 seems better, at least for the initial ITSI use case.
    May need to revise this in the more general cases of anomaly detection.
        
    Returns:
        int: edge length
    """
    return 2
# def get_edge_length(df_resolution, sub_duration):
    # sub_length = sub_duration // df_resolution
    # half_hour_edge = max(timedelta(minutes=30) // df_resolution + 1, 2)
    # return half_hour_edge if half_hour_edge * 2 < sub_length else 2


def _df_timestamp_index(df_in):
    df_in[COL_DATE] = parse_timestamp(df_in)
    df_in = df_in.set_index(COL_DATE)
    return df_in

def set_output_anomaly_label(
    df_in,
    df,
    df_resolution=None,
    output_anomly_column=OUTPUT_ANOMALY_COL,
    detector_anomaly_column=COL_ANOMALY_LABEL, #TODO: clean up COL_ANOMALY_LABEL
):
    df_in = _df_timestamp_index(df_in)
    if df_in.shape[0] == df.shape[0]:
        df_in[output_anomly_column] = df[detector_anomaly_column].to_numpy()
        df_in[COL_ANOMALY_SCORE] = df[COL_ANOMALY_SCORE].to_numpy()
    else:
        start = 0
        while np.isnan(df_in[COL_VALUE].iloc[start]):
            start += 1
        end = df_in.shape[0]
        while np.isnan(df_in[COL_VALUE].iloc[end - 1]):
            end -= 1

        if df_resolution is None:
            df_resolution = get_resolution(df)

        df_in[output_anomly_column] = np.concatenate(
            (
                np.zeros(start, dtype=int),
                np.array(
                    df[detector_anomaly_column].loc[
                        df_in.index[start:end].floor(df_resolution)
                    ]
                ),
                np.zeros(df_in.shape[0] - end, dtype=int),
            )
        )
        df_in[COL_ANOMALY_SCORE] = np.concatenate(
            (
                np.full(start, sigmoid(0)),
                np.array(
                    df[COL_ANOMALY_SCORE].loc[
                        df_in.index[start:end].floor(df_resolution)
                    ]
                ),
                np.full(df_in.shape[0] - end, sigmoid(0)),
            )
        )

    # TODO: move the sort-n-logging out of this function
    top_confs = sorted(
        df_in[COL_ANOMALY_SCORE][df_in[output_anomly_column] == 1], reverse=True
    )[:5]
    logger.info(
        f"{setup_logging.ANOMALY_APP_TELEMETRY} Top confidences of detected anoms: {top_confs}"
    )
    return df_in
