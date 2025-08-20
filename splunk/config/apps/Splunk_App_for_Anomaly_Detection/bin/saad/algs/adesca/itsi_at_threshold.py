from datetime import timedelta
import numpy as np

from saad.algs.adesca.dev_util import is_verbose
from saad.algs.adesca.dev_util import format_float_list
from saad.algs.adesca.data_prepare import (
    COL_VALUE,
    COL_BND_LOW,
    COL_BND_UP,
    COL_EDGE_MASK,
    EPSILON
)
from saad.algs.adesca.data_prepare import mean_std_2d
from saad.algs.adesca.data_prepare import calc_iqr, IQR_FACTOR
from saad.algs.adesca.sub_sequence import pack_sub_values_with_filter
from saad.algs.adesca.threshold_utils import (
    subsequence_by_time_policy,
    sub_sequences_groupby,
    get_edge_length,
)

HISTORY_NORMAL_PERCENTILE_THRESHOLD = 50
DEFAULT_Z = 3.3

SPLIT_DELTA_PNT = 12  # minimum split size wrt number of data points

CLIP_LEFT = "L"
CLIP_RIGHT = "R"
CLIP_BOTH = "LR"

DEFAULT_EDGE_CLIP = 1


def _subs_means_std_z(sub_values, clip="", calculate_z=True, df_iqr=0.0):
    assert (
        len(sub_values.shape) == 2
        and sub_values.shape[0] > 1
        and sub_values.shape[1] > 1
    ), "Unexpected sub_values in _subs_means_std_z"

    if clip == CLIP_LEFT:
        sub_values = sub_values[:, DEFAULT_EDGE_CLIP:]
    elif clip == CLIP_RIGHT:
        sub_values = sub_values[:, :-DEFAULT_EDGE_CLIP]
    elif clip == CLIP_BOTH:
        sub_values = sub_values[:, DEFAULT_EDGE_CLIP:-DEFAULT_EDGE_CLIP]

    means, stds = mean_std_2d(sub_values)

    if calculate_z:
        zs = np.empty(sub_values.shape[0])
        for i in range(sub_values.shape[0]):
            if stds[i] < df_iqr * IQR_FACTOR:
                stds[i] = df_iqr * IQR_FACTOR
            stds[i] += EPSILON

            sub = sub_values[i, :]
            sub_zs = np.absolute(sub - means[i]) / stds[i]
            zs[i] = max(sub_zs.max(), 1.0)
        z = zs.max()
    else:
        z = 0.0

    return means, stds.max(), z


class HistoryNormal:
    def __init__(
        self,
        sub_values,
        sub_timestamps,
        split_pnts=None,
        one_hour_len=0,
        df_iqr=0.0,
    ) -> None:
        if not (
            len(sub_values.shape) == 2
            and sub_values.shape[0] > 1
            and sub_values.shape[1] > 1
        ):
            print(f"Check HistoryNormal.init...")
        assert (
            len(sub_values.shape) == 2
            and sub_values.shape[0] > 1
            and sub_values.shape[1] > 1
        ), "Unexpected sub_values in HistoryNormal init"
        self.timestamps = np.array(sub_timestamps)
        self.split_pnts = split_pnts
        self.one_hour_len = one_hour_len

        segment_len = sub_values.shape[1]

        if split_pnts is not None:
            means, stds, zs = [], [], []
            left = 0
            for i in split_pnts:
                right = i * one_hour_len
                piece_data = sub_values[:, left:right]
                clip = CLIP_LEFT if left == 0 else ""
                (mean, std, z) = _subs_means_std_z(
                    piece_data, clip=clip, df_iqr=df_iqr
                )
                means.append(mean)
                stds.append(std)
                zs.append(z)
                left = right
            if left < segment_len:
                piece_data = sub_values[:, left:]
                (mean, std, z) = _subs_means_std_z(
                    piece_data, CLIP_RIGHT, df_iqr=df_iqr
                )
                means.append(mean)
                stds.append(std)
                zs.append(z)
        else:
            (mean, std, z) = _subs_means_std_z(
                sub_values, CLIP_BOTH, df_iqr=df_iqr
            )
            means = [mean]
            stds = [std]
            zs = [z]

        self.means = np.array(means)
        self.stds = np.array(stds)
        self.zs = np.array(zs)

    def __str__(self) -> str:
        if self.split_pnts is not None:
            means = format_float_list(np.mean(self.means, axis=1))
            return f"HistoryNormal({self.split_pnts}, means={means}, stds={format_float_list(self.stds)}, zs={format_float_list(self.zs)})"
        else:
            return f"HistoryNormal(mean={np.mean(self.means):.3f}, std={self.stds[0]:.3f}, z={self.zs[0]:.3f})"


def _calc_history_normal_behavior(
    subs,
    percentile_threshold=HISTORY_NORMAL_PERCENTILE_THRESHOLD,
    df_iqr=0.0,
):
    def is_too_short():  # TODO:
        return subs[0].length <= SPLIT_DELTA_PNT or subs[0].duration <= timedelta(
            hours=4
        )

    assert len(subs) > 3, "Not enough subsequences for _calc_history_normal_behavior"

    if subs[0].length < subs[1].length: # skip the 1st partial subsequence for _calc_history_normal_behavior
        subs = subs[1:]

    sub_filtered_values, sub_timestamps, percentiles = pack_sub_values_with_filter(
        subs, percentile_threshold
    )

    for sub, percentile in zip(subs, percentiles):
        sub.score_percentile = percentile  # set this value here, to show in plot later

    if is_too_short():  # no need to split
        return HistoryNormal(sub_filtered_values, sub_timestamps)

    # split long subsequences to smaller segments for thresholding

    sub_len = subs[0].length
    sub_hour = int(subs[0].duration.total_seconds() // 3600)  # segment duration in hour
    one_hour_len = sub_len // sub_hour  # number of points for one hour duration

    if one_hour_len < SPLIT_DELTA_PNT:
        split_delta = SPLIT_DELTA_PNT // one_hour_len + 1
    else:
        split_delta = 1

    (means, stds) = ([], [])
    for i in range(0, sub_hour, split_delta):
        piece_data = sub_filtered_values[
            :, i * one_hour_len : min(sub_len, (i + split_delta) * one_hour_len)
        ]
        clip = CLIP_LEFT if i == 0 else CLIP_RIGHT if i + split_delta >= sub_len else ""
        mean, std, _ = _subs_means_std_z(
            piece_data, calculate_z=False, clip=clip
        )
        means.append(np.mean(mean))
        stds.append(std)

    split_pnts = []
    for i in range(1, len(means)):
        mean_diff = np.absolute(means[i] - means[i - 1])
        std = min(stds[i], stds[i - 1])
        if mean_diff > std * 0.3:
            split_pnts.append(i)

    if len(split_pnts) > 0:
        return HistoryNormal(
            sub_filtered_values,
            sub_timestamps,
            split_pnts=[i * split_delta for i in split_pnts],
            one_hour_len=one_hour_len,
            df_iqr=df_iqr,
        )
    else:  # no splitting
        return HistoryNormal(
            sub_filtered_values, 
            sub_timestamps, 
            df_iqr=df_iqr
        )

def calc_time_series_normal_behavior(df, tp):
    subs, label_method = subsequence_by_time_policy(
        df, tp
    )  # partitioning based on time-policy

    df_iqr = calc_iqr(df[COL_VALUE].to_numpy())
    history_normal_cluster_dict = {}
    for label, subs_group in sub_sequences_groupby(subs, label_method).items():
        history_normal = _calc_history_normal_behavior(subs_group, df_iqr=df_iqr)
        if is_verbose():
            print(f"==HB== {str(history_normal)}")
        history_normal_cluster_dict[label] = history_normal

    return history_normal_cluster_dict, subs, label_method

def itsi_thresholding(df, tp, history_normal_cluster_dict, sub_length, subs=None, label_method=None):
    if subs is None:
        subs, label_method = subsequence_by_time_policy(
            df, tp, False
        )  # partitioning based on time-policy

    subs_total_len = sum([s.length for s in subs])
    bnd_up = np.empty(subs_total_len)
    bnd_low = np.empty(subs_total_len)

    edge_mask = np.empty(
        subs_total_len, dtype=int
    )  # use edge_mask to reduce FP at segments' edges
    edge_length = get_edge_length()

    idx = 0
    is_first_sub = True
    for sub in subs:
        sub_label = label_method(sub)
        history_normal = history_normal_cluster_dict[sub_label]

        threshold = get_thresholds_from_history(
            history_normal, sub_length, sub.length, is_first_sub
        )
        bnd_up[idx : idx + sub.length] = threshold[0]
        bnd_low[idx : idx + sub.length] = threshold[1]

        if sub.length > 2 * edge_length:
            edge_mask[idx : idx + sub.length] = np.concatenate(
                (
                    np.zeros(edge_length, dtype=int),
                    np.ones(sub.length - 2 * edge_length, dtype=int),
                    np.zeros(edge_length, dtype=int),
                )
            )
        else:
            edge_mask[idx : idx + sub.length] = np.zeros(sub.length, dtype=int)

        idx += sub.length
        is_first_sub = False

    df[COL_BND_UP] = bnd_up
    df[COL_BND_LOW] = bnd_low
    df[COL_EDGE_MASK] = edge_mask

    return df


def get_thresholds_from_history(history_normal, sub_length, length, is_first_sub):
    history_means = history_normal.means.mean(axis=1)
    history_stds = history_normal.means.std(axis=1) + history_normal.stds

    def _get_up_low_bnds(mid, std, z):
        variation = std * z
        return mid + variation, mid - variation

    if history_normal.split_pnts is not None:
        bnd_up = np.empty(length)
        bnd_low = np.empty(length)

        if is_first_sub and length < sub_length:
            sub_length_in_hour = sub_length // history_normal.one_hour_len
            # fill bnd in reverse order
            right = length
            right_in_hour = sub_length_in_hour
            for i, split_pnt in reversed(list(enumerate(history_normal.split_pnts))):
                (up, low) = _get_up_low_bnds(
                    history_means[i+1], history_stds[i+1], history_normal.zs[i+1]
                )
                segment_length = (right_in_hour - split_pnt) * history_normal.one_hour_len
                
                left = max(0, right - segment_length)
                bnd_up[left:right] = up
                bnd_low[left:right] = low
                right = left
                right_in_hour = split_pnt
                if left == 0:
                    break
                
            if right > 0:
                (up, low) = _get_up_low_bnds(
                    history_means[0], history_stds[0], history_normal.zs[0]
                )
                bnd_up[:right] = up
                bnd_low[:right] = low            

        else:
            left = 0
            for i, split_pnt in enumerate(history_normal.split_pnts):
                right = split_pnt * history_normal.one_hour_len
                (up, low) = _get_up_low_bnds(
                    history_means[i], history_stds[i], history_normal.zs[i]
                )
                bnd_up[left:right] = up
                bnd_low[left:right] = low
                left = right
            if left < length:
                (up, low) = _get_up_low_bnds(
                    history_means[-1], history_stds[-1], history_normal.zs[-1]
                )
                bnd_up[left:] = up
                bnd_low[left:] = low
    else:
        (up, low) = _get_up_low_bnds(
            history_means, history_stds[0], history_normal.zs[0]
        )
        bnd_up = np.full(length, up)
        bnd_low = np.full(length, low)

    return bnd_up, bnd_low
