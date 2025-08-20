import numpy as np
import pandas as pd
import dateutil
import datetime

# Constants for time-string units
SECONDS = 'S'
MINUTES = 'M'
HOURS = 'H'
DAYS = 'D'
WEEKS = 'W'
MONTHS = 'MO'
YEARS = 'Y'

# Mapping from each time-string unit to the associated number of seconds
TIME_UNIT_LOOKUP = {
    SECONDS: 1,  # second
    MINUTES: 60,  # minute
    HOURS: 60 * 60,  # hour
    DAYS: 24 * 60 * 60,  # day
    WEEKS: 7 * 24 * 60 * 60,  # week
    MONTHS: int(round((365.25 / 12) * 7 * 24 * 60 * 60)),  # average month
    YEARS: int(round(365.25 * 7 * 24 * 60 * 60))  # average year
}


def convert_to_unix_timestamp(timestamp):
    # Check if it's already unix timestamp
    try:
        return int(round(float(timestamp)))
    except:
        pass
    
    # If it's not a unix timestamp, parse it
    date = dateutil.parser.parse(timestamp)
    unix_seconds = datetime.datetime.timestamp(date)
    unix_timestamp = int(round(unix_seconds))
    return unix_timestamp
    

def parse_time_string(time_str):
    if (time_str is None) or (len(time_str) == 0):
        return None

    # Split the alphanumeric string into alpha and numeric strings
    unit = ''
    value_str = ''
    for ch in time_str.strip().upper():
        if ch.isdigit():
            value_str += ch
        elif ch.isalpha():
            unit += ch
        else:
            raise ValueError('Encountered non-alphanumeric character "' + ch + '" ' +
                             'when parsing time-string "' + time_str + '"')

    # Default to 1 if no digit given
    value = int(value_str) if (len(value_str) > 0) else 1

    # Ensure valid unit
    if unit not in TIME_UNIT_LOOKUP:
        raise ValueError('Encountered unknown time-string-unit "' + unit + '" ' +
                         'when parsing time-string "' + time_str + '". ' +
                         'Allowed units are: ' + str(list(TIME_UNIT_LOOKUP.keys())))

    return value, unit


# Converts time-string to a number of seconds
def time_string_to_seconds(time_str):
    value, unit = parse_time_string(time_str)
    unit_seconds = TIME_UNIT_LOOKUP[unit]
    interval = value * unit_seconds
    return interval


# Divides the first time-string by the second. Exception is thrown if they don't divide evenly.
def divide_time_strings(numerator_time_str, denominator_time_str):
    numerator_seconds = time_string_to_seconds(numerator_time_str)
    denominator_seconds = time_string_to_seconds(denominator_time_str)
    result = numerator_seconds / denominator_seconds
    if (result % 1) != 0:
        raise ValueError('Time string "' + numerator_time_str + '" ' +
                         'does not evenly divide by "' + denominator_time_str + '"')
    return int(result)


def multiply_time_string(time_str, multiplier):
    value, unit_str = parse_time_string(time_str)
    multiplied_time_string = str(int(round(multiplier * value))) + unit_str
    simplified_time_string = simplify_time_string(multiplied_time_string)
    return simplified_time_string


# Given a time string, reduces it to it's simplest unit (e.g. '1440m' -> '1d')
def simplify_time_string(time_str):
    time_str_seconds = time_string_to_seconds(time_str)
    largest_divisor = (float('inf'), None)  # (dividend when divided by divisor, unit)
    for unit_str, unit_seconds in TIME_UNIT_LOOKUP.items():
        if (time_str_seconds % unit_seconds) == 0:
            dividend = int(time_str_seconds / unit_seconds)
            if dividend < largest_divisor[0]:
                largest_divisor = (dividend, unit_str)
    dividend, unit_str = largest_divisor
    return str(dividend) + unit_str


# Given an integer number of seconds, converts it to a time string
def seconds_to_time_string(seconds):
    largest_divisor = (float('inf'), None)  # (dividend when divided by divisor, unit)
    for unit_str, unit_seconds in TIME_UNIT_LOOKUP.items():
        if (seconds % unit_seconds) == 0:
            dividend = int(seconds / unit_seconds)
            if dividend < largest_divisor[0]:
                largest_divisor = (dividend, unit_str)
    dividend, unit_str = largest_divisor
    return str(dividend) + unit_str

import numpy as np

from scipy import signal
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

# VARIABLES CONTROLLING SENSITIVITY TO MAGNITUDE
# How significant the ACF peak needs to be to be accepted
ACF_PROMINENCE_THRESHOLD = 0.3
# How significant the matched SDF peak needs to be to be accepted
SDF_PROMINENCE_THRESHOLD = 1
# How significant a SDF peaks needs to be to be part of the initial peak
ACF_INITIAL_PEAK_PROMINENCE_THRESHOLD = 0.2

# VARIABLES CONTROLLING SENSITIVITY TO PERTURBATION
# How many bandwidths away an expected SDF peak is allowed to be from the real peak
SDF_BANDWIDTH_MATCH_THRESHOLD = 1
# When finding the peak-sequence, how many values should we try to find
ACF_PEAK_SEQ_COUNT = 3
# How much error is allowed in the peak sequence (as a percent of the base-multiples value)
ACF_PEAK_SEQ_ERROR_THRESHOLD = 0.05

# VARIABLES CONTROLLING RECENCY OF ANALYZED DATA
# Samples the most recent DETECTION_SUBSAMPLE_SIZE data points at each resolution when detecting seasonal frequencies
DETECTION_SUBSAMPLE_SIZE = 2000
# When performing iterative down-sampling, what resolutions should we use
DOWNSAMPLING_RESOLUTIONS = ['1M', '5M', '15M', '1H', '4H', '1D', '1W', '1MO']


########################################################################
#                          Top-Level Methods                           #
########################################################################


def detect_seasonal_freqs(series, resolution_str):
    series_res_seconds = time_string_to_seconds(resolution_str)

    # If we weren't given at least 4 data points, don't do anything
    extracted_frequencies = []
    if len(series) < 4:
        return extracted_frequencies

    stl_series = list(series)  # series we'll apply STL to
    prev_resolution = resolution_str
    for sample_resolution in DOWNSAMPLING_RESOLUTIONS:
        sample_res_seconds = time_string_to_seconds(sample_resolution)

        # Downsample both the original and STL series
        downsampled_stl_series = downsample_series(stl_series, prev_resolution, sample_resolution)
        if (downsampled_stl_series is None):  # Case when sample-resolution is finer than data-resolution
            continue
        stl_series = downsampled_stl_series
        series = downsample_series(series, prev_resolution, sample_resolution)
        if len(series) < 4:
            break

        # Find the candidate frequencies in the non-STL downsampled series
        original_series_subsample = series[-DETECTION_SUBSAMPLE_SIZE:]
        initial_candidates = {c['ACF_corrected_peak']: c for c in compute_candidate_freqs(original_series_subsample)}

        # Iteratively detect/remove seasonal frequencies
        while True:

            # Compute the ACF peaks for on the most recent DETECTION_SUBSAMPLE_SIZE points
            stl_series_subsample = stl_series[-DETECTION_SUBSAMPLE_SIZE:]
            candidate_freqs = compute_candidate_freqs(stl_series_subsample)

            # Drop candidates with frequencies less than what we've already extracted.
            # Prevents induced peaks (e.g. remove 5 then 15, and suddenly 10 becomes a peak)
            max_extracted_freq = 0 if (len(extracted_frequencies) == 0) else extracted_frequencies[-1]
            interval = int(sample_res_seconds / series_res_seconds)
            filtered_candidates = [c for c in candidate_freqs if (c['ACF_corrected_peak'] * interval) > max_extracted_freq]

            # Apply our filters to the ACF peaks; if none survived, break
            filtered_candidates = filter_candidate_freqs(filtered_candidates, initial_candidates)
            if len(filtered_candidates) == 0:
                break

            # Find the minimum peak, and remove it from the series using STL
            min_freq = int(min([c['ACF_corrected_peak'] for c in filtered_candidates]))
            if min_freq > 100:  # Lack confidence in freqs > 100; if real, should be detected on next down-sampling pass
                break
            stl_series = STL(stl_series, period=min_freq).fit().resid

            # Record the removed seasonal frequency
            seasonal_freq = min_freq * int(sample_res_seconds / series_res_seconds)
            extracted_frequencies.append(seasonal_freq)

        prev_resolution = sample_resolution

    return extracted_frequencies

# def detect_minimal_seasonal_frequency(series):
#     # Compute the ACF peaks for on the most recent DETECTION_SUBSAMPLE_SIZE points
#     series_subsample = series[-DETECTION_SUBSAMPLE_SIZE:]
#     candidate_freqs = compute_candidate_freqs(series_subsample)
#
#     # Drop candidates with frequencies less than what we've already extracted.
#     # Prevents induced peaks (e.g. remove 5 then 15, and suddenly 10 becomes a peak)
#     max_extracted_freq = 0 if (len(extracted_frequencies) == 0) else extracted_frequencies[-1]
#     interval = int(sample_res_seconds / series_res_seconds)
#     filtered_candidates = [c for c in candidate_freqs if (c['ACF_corrected_peak'] * interval) > max_extracted_freq]
#
#     # Apply our filters to the ACF peaks; if none survived, break
#     filtered_candidates = filter_candidate_freqs(filtered_candidates, initial_candidates)
#     if len(filtered_candidates) == 0:
#         return None
#
#     # Find the minimum peak, and remove it from the series using STL
#     min_freq = int(min([c['ACF_corrected_peak'] for c in filtered_candidates]))
#     return min_freq
#     if min_freq > 100:  # Lack confidence in freqs > 100; if real, should be detected on next down-sampling pass
#         return None
#     re


def compute_candidate_freqs(series):
    peak_dicts = []

    # Compute the ACF peaks
    diff_acf = compute_diff_acf(series)
    acf_peaks = find_peaks(diff_acf, ACF_INITIAL_PEAK_PROMINENCE_THRESHOLD)
    acf_peak_prominences = compute_prominences(diff_acf, acf_peaks)

    # Add ACF stats to the peak-dict
    for peak_index, peak_prominence in zip(acf_peaks, acf_peak_prominences):
        peak_index = int(peak_index)

        peak_seq = find_longest_approximate_sequence(acf_peaks, peak_index, limit=ACF_PEAK_SEQ_COUNT)

        peak_dicts.append({
            'ACF_peak_index': peak_index,
            'ACF_corrected_peak': int(compute_base_multiple(peak_seq)),
            'ACF_peak_sequence': peak_seq,
            'ACF_peak_sequence_avg_error': compute_avg_seq_error(peak_seq),
            'ACF_prominence': compute_prominences(diff_acf, [peak_index])[0]
        })

    # Compute SDF peaks
    freqs, dens = compute_sdf(series)
    log_dens = np.log10(dens)  # used for computing prominences
    raw_spec_peaks = find_peaks(log_dens).astype(np.float64)  # index of peak in the density values
    bandwidth = freqs[1]
    spec_peaks = raw_spec_peaks * bandwidth  # x-values of the peaks

    # If there's no SDF peaks, then we say there's no peaks at all
    if len(spec_peaks) == 0:
        return []

    # Add SDF stats to the peak-dict
    for peak_dict in peak_dicts:
        peak = peak_dict['ACF_corrected_peak']
        expected_peak = 1.0 / peak
        nearest_peak_index = np.argmin(np.abs(spec_peaks - expected_peak))
        diff = abs(spec_peaks[nearest_peak_index] - expected_peak)
        nearest_peak_x_index = int(raw_spec_peaks[nearest_peak_index])  # nearest peak's index in the density values

        peak_dict['SDF_nearest_peak_x_index'] = nearest_peak_x_index  # Identifies which peak this is
        peak_dict['SDF_bandwidths_from_nearest_peak'] = diff / bandwidth  # Diff between expected and real SDF peak
        peak_dict['SDF_prominence'] = compute_prominences(log_dens, [nearest_peak_x_index])[0]

    return peak_dicts


def filter_candidate_freqs(candidates, initial_diff_acf_peaks):
    filtered_candidates = initial_freq_filter(candidates, initial_diff_acf_peaks)
    filtered_candidates = multiplicity_filter(filtered_candidates)
    filtered_candidates = sdf_peak_match_filter(filtered_candidates)
    filtered_candidates = sdf_peak_match_conflict_filter(filtered_candidates)
    filtered_candidates = acf_peak_magnitude_filter(filtered_candidates)
    filtered_candidates = sdf_peak_magnitude_filter(filtered_candidates)
    return filtered_candidates


########################################################################
#                            Filter Methods                            #
########################################################################

# Ensures the ACF peak's prominence is of sufficient magnitude
def acf_peak_magnitude_filter(candidates):
    return [c for c in candidates if c['ACF_prominence'] > ACF_PROMINENCE_THRESHOLD]


# Ensures the matched SDF peak's prominence is of sufficient magnitude
def sdf_peak_magnitude_filter(candidates):
    return [c for c in candidates if c['SDF_prominence'] > SDF_PROMINENCE_THRESHOLD]


# Ensures the candidate's expected SDF peak is sufficiently close to the actual SDF peak
def sdf_peak_match_filter(candidates):
    return [c for c in candidates if c['SDF_bandwidths_from_nearest_peak'] < SDF_BANDWIDTH_MATCH_THRESHOLD]


# Ensures candidate frequencies were present in the original data (before we extracted any frequencies)
def initial_freq_filter(candidates, initial_acf_peak_set):
    return [c for c in candidates if c['ACF_corrected_peak'] in initial_acf_peak_set]


# Ensures that if a candidate has frequency X, then there's also a non-trivial frequency at ~2x
def multiplicity_filter(candidates):
    peak_lookup = {c['ACF_peak_index']: c for c in candidates}
    keep_peaks = []
    for candidate in candidates:
        # Ensure the peak-sequence isn't too noisy
        if candidate['ACF_peak_sequence_avg_error'] >= ACF_PEAK_SEQ_ERROR_THRESHOLD:
            continue

        # Ensure a ~2x peak exists
        if len(candidate['ACF_peak_sequence']) < 2:
            continue

        # Ensure the 2x peak wasn't filtered out already
        nearest_2x_peak = peak_lookup.get(candidate['ACF_peak_sequence'][1], None)
        if nearest_2x_peak is None:
            continue

        # Ensure the 2x peak has at least half the prominence of the initial peak (max required prominence of 0.2)
        if (nearest_2x_peak is None) or (nearest_2x_peak['ACF_prominence'] < min(candidate['ACF_prominence'] / 2, 0.2)):
            continue

        keep_peaks.append(candidate)

    return keep_peaks


# If multiple ACF peaks matched the same SDF peak, select the one with the highest prominence
def sdf_peak_match_conflict_filter(candidates):
    # Group peaks by nearest SDF peak
    peak_groups = {}
    for c in candidates:
        peak_id = c['SDF_nearest_peak_x_index']
        if peak_id not in peak_groups:
            peak_groups[peak_id] = []
        peak_groups[peak_id].append(c)

    # Select the max-prominence ACF peak for each group
    keep_peaks = []
    for peak_group in peak_groups.values():
        max_peak_index = np.argmax([p['ACF_prominence'] for p in peak_group])
        keep_peaks.append(peak_group[max_peak_index])
    return keep_peaks

########################################################################
#                          Helper Methods                              #
########################################################################


def find_longest_approximate_sequence(nums, start_num, limit=None):
    nums_set = set(nums)
    if start_num not in nums_set:
        return None

    sequence = [int(start_num)]
    while True:
        base_multiple = compute_base_multiple(sequence)
        margin = min(4, max(1, int(round(0.1 * base_multiple))))
        search_val = base_multiple * (len(sequence) + 1)
        search_min = (base_multiple - margin) * (len(sequence) + 1)
        search_max = (base_multiple + margin) * (len(sequence) + 1)
        search_range = [v for v in range(search_min, search_max + 1)]
        search_range.sort(key=lambda v: abs(v - search_val))  # Ordered by distance [0, 1, -1, 2, -2, ...]

        # Search for any of the search values, selecting the closest one
        match_found = False
        for search_val in search_range:
            if (search_val != sequence[-1]) and (search_val in nums_set):
                sequence.append(search_val)
                match_found = True
                break

        # If no search value was found, or we've hit the limit, break
        if (not match_found) or ((limit is not None) and (len(sequence) >= limit)):
            break

    return sequence


def compute_next_search_val(seq):
    result = 0
    for i, v in enumerate(seq):
        result += v * ((len(seq) + 1) / (i + 1))  # Properly weight each term
    return int(round(result / len(seq)))


def compute_base_multiple(seq):
    scaled_terms = [v / (i + 1) for i, v in enumerate(seq)]
    expected_start = int(round(sum(scaled_terms) / len(scaled_terms)))
    return expected_start


def compute_avg_seq_error(seq):
    expected_start = compute_base_multiple(seq)
    errors = [abs(v - (expected_start * (i + 1))) for i, v in enumerate(seq)]
    avg_error = sum(errors) / len(errors)
    scaled_error = avg_error / expected_start  # error as a % of vals' magnitude
    return scaled_error


def compute_prominences(values, peaks):
    values = balance_end_peak(values)
    return signal.peak_prominences(values, peaks)[0]


def compute_diff_acf(series):
    return acf(np.diff(series), nlags=int(len(series) / 2), fft=False)


def compute_sdf(series):
    freqs, dens = signal.welch(
        series,
        nperseg=len(series)
    )
    freqs[-1] = 0.5
    return freqs, dens


def find_peaks(values, min_prominence=0.0):
    values = balance_end_peak(values)
    return signal.find_peaks(values, prominence=min_prominence)[0]


# If the series ends in a 1-sided peak, duplicate the peak's side so it can be detected
def balance_end_peak(values):
    lti, _ = find_troughs(values, len(values) - 1)
    diff = len(values) - 1 - lti
    return list(values) + list(reversed(values[-(diff + 1): -1]))


# Given a list of numbers and a starting index, searches in each direction until
# it encounters an increasing point. The goal is to find the troughs on either side
def find_troughs(values, peak_i):
    left_trough_i = peak_i
    while (left_trough_i > 0) and (values[left_trough_i - 1] < values[left_trough_i]):
        left_trough_i -= 1

    right_trough_i = peak_i
    while (right_trough_i < len(values) - 1) and (values[right_trough_i + 1] < values[right_trough_i]):
        right_trough_i += 1

    return left_trough_i, right_trough_i


def downsample_series(series, series_resolution, sample_resolution):
    series_res_seconds = time_string_to_seconds(series_resolution)
    sample_res_seconds = time_string_to_seconds(sample_resolution)
    if (sample_res_seconds % series_res_seconds) != 0:
        return None
    sample_size = int(sample_res_seconds / series_res_seconds)

    sample_series = []
    stat_dict = {'count': 0, 'sum': 0}
    for val in series:
        stat_dict['count'] += 1
        stat_dict['sum'] += val

        # If end-of-sample, close our sample, record the value, and reset the counters
        if stat_dict['count'] == sample_size:
            sample_series.append(stat_dict['sum'] / stat_dict['count'])
            stat_dict['count'] = stat_dict['sum'] = 0

    return np.array(sample_series)


def detect_dominant_period(series, timestamps):
    """
    Detects the presence (and length) of repeated seasonal patterns in time series data.
    Assumes the series (and the corresponding timestamps) are sorted in chronological order.

    Arguments:
        - `series`, list of floats, the series' values
        - `timestamps`, list of integer Unix seconds (e.g. 1660340326), the series' timestamps
        
    Returns:
        - `dominant_period`, int, the length of the dominant seasonal pattern in the data, or 1 if no seasonality was detected.
        Note: We use 1 instead of 0 to indicate no seasonality because this is what MLTK's `StateSpaceForecast` model expects.
    """
    # Estimate the resolution of the series by finding the median elapsed time between timestamps.
    timestamps = [convert_to_unix_timestamp(timestamp) for timestamp in timestamps] 
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    sorted_diffs = sorted(diffs)
    median_idx = int(len(sorted_diffs) / 2)
    resolution_seconds = sorted_diffs[median_idx]
    resolution_string = seconds_to_time_string(resolution_seconds)

    # Detect seasonal periods and return the dominant one, if any
    freqs = detect_seasonal_freqs(series, resolution_string)
    dominant_period = 1 if len(freqs) == 0 else freqs[0]
    return dominant_period
