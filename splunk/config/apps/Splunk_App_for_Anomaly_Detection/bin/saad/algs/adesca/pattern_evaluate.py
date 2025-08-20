from datetime import timedelta
from functools import partial
import numpy as np
import pandas as pd
from sklearn import metrics

from saad.algs.adesca.dev_util import is_verbose
from saad.algs.adesca.data_prepare import detrend_daily
from saad.algs.adesca.dev_util import format_float_list
from saad.algs.adesca.sub_sequence import (
    df_to_day_sequences,
    df_to_hour_sequences,
    df_to_half_day_sequences,
)

SILHOUETTE_NEG_INF = -2.0

HOURDELTA_LIST = [1, 2, 3, 4]
THRESHOLD_SILHOUETTE = 0.2
THRESHOLD_SILHOUETTE_HIGH = 0.6
THRESHOLD_IRREGULAR_SEQUENCE_LENGTH = 0.7

MIN_POINTS_PER_SUBSEQUENCE = 4

# the general weekly pattern: seven days in a week can be labeled to all possible clustering schemes
LABEL_WEEKLY_PATTERN = lambda dow_label_dict, seq: dow_label_dict[seq.dayofweek()]

LABEL_WEEKLY_OFFDAYS = (
    lambda offday_start, seq: 1
    if seq.dayofweek() == offday_start or seq.dayofweek() == (offday_start + 1) % 7
    else 0
)
LABEL_WEEKEND = lambda seq: 0 if seq.dayofweek() < 5 else 1  # Monday=0, Sunday=6
LABEL_HOUR = lambda seq: seq.hour()
LABEL_ALTER_HOUR_BLOCK = (
    lambda hourdelta, offset, seq: 0 if ((seq.hour() - offset) // hourdelta) % 2 else 1
)


def evaluate_clustering_quality(in_sequences, get_seq_label, kee_all_sequences=False):
    lengths = [s.length for s in in_sequences]
    counts = np.bincount(lengths)
    len_seq = np.argmax(
        counts
    )  # use the most frequent one as the sequence length of this data set
    sequences = [
        s for s in in_sequences if s.length == len_seq
    ]  # make sure sequences having same length
    cnt_regular_seq = len(sequences)
    if cnt_regular_seq < len(in_sequences) and (
        cnt_regular_seq < THRESHOLD_IRREGULAR_SEQUENCE_LENGTH * len(in_sequences)
    ):
        if is_verbose():
            print(
                f"==HB== Only {cnt_regular_seq} out of {len(in_sequences)} subsequences have regular length."
            )
        return -2, sequences

    # quantify clustering quality with silhouette_score
    X = np.empty((cnt_regular_seq, len_seq))
    labels = np.empty((cnt_regular_seq,), dtype=int)
    for idx, seq in enumerate(sequences):
        X[idx] = seq.values
        labels[idx] = get_seq_label(seq)

    if len(np.unique(labels)) < 2:
        print("==HB== Need at least 2 clusters to calcualte silhouette score.")
        return -2, sequences

    scores = metrics.silhouette_samples(X, labels)
    med_score = np.median(scores)

    # save silhouette_samples of each sub-sequence
    for score, seq in zip(scores, sequences):
        seq.silhouette = score

    if kee_all_sequences:
        idx = 0
        for sub in in_sequences:
            if sub.length == len_seq:
                sub.silhouette = sequences[idx].silhouette
                idx += 1
            else:
                sub.silhouette = -1.0
        return med_score, in_sequences

    else:
        return med_score, sequences


def check_weekend_pattern(df):
    sub_sequences = df_to_day_sequences(df)
    score, _ = evaluate_clustering_quality(sub_sequences, LABEL_WEEKEND)
    if is_verbose():
        print(f"==HB== {score:.3f} for weekend pattern")
    return score > THRESHOLD_SILHOUETTE


def evaluate_weeky_pattern_with_offset(df):
    scores = []
    for offset in range(12):
        sub_sequences = df_to_day_sequences(df, offset=offset)
        score, _ = evaluate_clustering_quality(sub_sequences, LABEL_WEEKEND)
        scores.append(score)
    score, offset = max(scores), np.argmax(scores)
    if is_verbose():
        print(
            f"==HB== offset={offset}, {format_float_list(scores)} for weekend pattern"
        )
    return score, offset


# for weekday-offday pattern NOT aligned with calender weekday-weekend
def check_weekly_pattern(df):
    if df.index[-1] - df.index[0] <= timedelta(days=21):
        if is_verbose():
            print(
                f"==HB== There are only {df.index[-1] - df.index[0]} of data, we skip evaluting weekly patterns for time policy recommendation."
            )
        offday_start = -1
        return False, offday_start, 0, None

    best_score, offday_start, offset = evaluate_async_weeky_pattern_with_offset(df)
    score, label_method = evaluate_5th_workday_pattern(df, offday_start, offset)
    if (
        abs(score - best_score) / best_score < 0.1
    ):  # when the score difference is small, prefer the pattern with 5th workday
        return score, offday_start, offset, label_method
    else:
        return (
            best_score,
            offday_start,
            offset,
            partial(LABEL_WEEKLY_OFFDAYS, offday_start),
        )


# calcualte the day-of-week to label dictionary
def get_dow_label_dict(offday_start, b_5th_workday=True):
    dow_label_dict = {}
    for dow in range(7):
        if dow == offday_start or dow == ((offday_start + 1) % 7):  # the two offdays
            dow_label_dict[dow] = 1
        elif dow == ((offday_start - 1) % 7):  # the 5th workday
            dow_label_dict[dow] = 2
        else:
            dow_label_dict[dow] = 0

    return dow_label_dict


def evaluate_5th_workday_pattern(df, offday_start, offset):
    sub_sequences = df_to_day_sequences(df, offset=offset)
    dow_label_dict = get_dow_label_dict(offday_start)
    label_method = partial(LABEL_WEEKLY_PATTERN, dow_label_dict)
    score, _ = evaluate_clustering_quality(sub_sequences, label_method)
    return score, label_method


def evaluate_async_weeky_pattern_with_offset(df):
    # search route.a: find the async offdays first
    scores_a7 = []
    sub_sequences = df_to_day_sequences(df)
    for start_day in range(7):
        label_weekly = partial(LABEL_WEEKLY_OFFDAYS, start_day)
        # lambda seq: 0 if seq.dayofweek()==start_day or seq.dayofweek()==((start_day+1) % 7) else 1
        score, _ = evaluate_clustering_quality(sub_sequences, label_weekly)
        scores_a7.append(score)

    offday1 = np.argmax(scores_a7)
    label_weekly = partial(
        LABEL_WEEKLY_OFFDAYS, offday1
    )  # lambda seq: 0 if seq.dayofweek()==offday1 or seq.dayofweek()==((offday1+1) % 7) else 1

    scores_a12 = []
    for offset in range(12):
        sub_sequences = df_to_day_sequences(df, offset=offset)
        score, _ = evaluate_clustering_quality(sub_sequences, label_weekly)
        scores_a12.append(score)

    # search route.b: find daily hour offset first
    scores_b12 = []
    for offset in range(12):
        sub_sequences = df_to_day_sequences(df, offset=offset)
        score, _ = evaluate_clustering_quality(sub_sequences, LABEL_WEEKEND)
        scores_b12.append(score)

    offset = int(np.argmax(scores_b12))

    scores_b7 = []
    sub_sequences = df_to_day_sequences(df, offset=offset)
    for start_day in range(7):
        label_weekly = partial(LABEL_WEEKLY_OFFDAYS, start_day)
        # label_weekly = lambda seq: 0 if seq.dayofweek()==start_day or seq.dayofweek()==((start_day+1) % 7) else 1
        score, _ = evaluate_clustering_quality(sub_sequences, label_weekly)
        scores_b7.append(score)

    if is_verbose():
        print(f"==HB== scores_a7  = {format_float_list(scores_a7)}")
    if is_verbose():
        print(f"==HB== scores_a12 = {format_float_list(scores_a12)}")
    if is_verbose():
        print(f"==HB== scores_b7  = {format_float_list(scores_b7)}")
    if is_verbose():
        print(f"==HB== scores_b12 = {format_float_list(scores_b12)}")

    if np.max(scores_a12) > np.max(scores_b7):
        return max(scores_a12), offday1, np.argmax(scores_a12)
    else:
        return max(scores_b7), np.argmax(scores_b7), offset


class HourBlockCandidate:
    def __init__(self, hourdelta, offset=0, silhouette=-2, label=0) -> None:
        self.hourdelta = hourdelta
        self.offset = offset
        self.silhouette = silhouette
        self.label = label

    def __str__(self) -> str:
        return f"HourBlockCandidate(hourdelta={self.hourdelta}, offset={self.offset}, silhouette={self.silhouette:.3f})"


def get_preferred_hour_block(hour_block_candidates):
    """A robust decision rule for hour-block-candidates, by:
    find a high quality clustering of the candidates wrt high score vs low score, then
    pick from the high score cluster the candidate of longest hour block
    """

    assert len(hour_block_candidates) > 2, "len(hour_block_candidates) <= 2"
    hbc_sorted = sorted(hour_block_candidates, key=lambda x: x.silhouette)
    # if is_verbose():
    #     print('    sorted hourdelta: ', [hbc.hourdelta for hbc in hbc_sorted], file=sys.stderr)
    X = np.array([hbc.silhouette for hbc in hbc_sorted])
    X = np.reshape(X, (-1, 1))
    best_score = SILHOUETTE_NEG_INF
    best_cutoff = 0
    for cutoff in range(1, len(hbc_sorted)):
        for i, hbc in enumerate(hbc_sorted):
            hbc.label = 0 if i < cutoff else 1
        score = metrics.silhouette_score(X, [hbc.label for hbc in hbc_sorted])
        if score > best_score:
            best_score = score
            best_cutoff = cutoff
        # if is_verbose():
        #     print(f'      {best_score:.3f} {[hbc.label for hbc in hbc_sorted]}, {score:.3f}', file=sys.stderr)
    return max(hbc_sorted[best_cutoff:], key=lambda hbc: hbc.hourdelta)


def evaluate_hour_block(df, df_resolution):
    if df.index[-1] - df.index[0] <= timedelta(days=1):
        if is_verbose():
            print(
                f"==HB== There are less than 1 day of data. At least 1 day of data is required for time policy recommendation."
            )
        return None

    hour_block_candidates = []
    df_detrended = detrend_daily(df)
    for hourdelta in HOURDELTA_LIST:
        if pd.Timedelta(hours=hourdelta) < MIN_POINTS_PER_SUBSEQUENCE * df_resolution:
            continue
        subs = df_to_hour_sequences(df_detrended, hourDelta=hourdelta, offset=0)
        score, _ = evaluate_clustering_quality(
            subs, partial(LABEL_ALTER_HOUR_BLOCK, hourdelta, 0)
        )
        hour_block_candidates.append(HourBlockCandidate(hourdelta, silhouette=score))
    if is_verbose():
        print(
            f"==HB== {format_float_list([hbc.silhouette for hbc in hour_block_candidates])} for hour block pattern"
        )
    if len(hour_block_candidates) == 0:
        return None
    elif len(hour_block_candidates) == 1:
        return hour_block_candidates[0]
    else:
        return get_preferred_hour_block(hour_block_candidates)


def evaluate_half_day(df):
    if df.index[-1] - df.index[0] <= timedelta(days=3):
        if is_verbose():
            print(
                f"==HB== There are only {df.index[-1] - df.index[0]} of data, we skip evaluting half-day patterns for time policy recommendation."
            )
        return -1.0, -1

    subs = df_to_half_day_sequences(df, start_hour=0)
    score, _ = evaluate_clustering_quality(subs, LABEL_HOUR)
    if is_verbose():
        print(f"==HB== half day: score={score:.3f}")
    return score, 0
