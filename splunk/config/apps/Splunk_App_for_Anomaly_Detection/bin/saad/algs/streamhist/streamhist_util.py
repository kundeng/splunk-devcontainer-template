from bisect import bisect_left
import math
import numpy as np
import os
import pandas as pd
import sys

# The first of these path inserts handles the case when this file is invoked from {PROJECT_DIR}/saad, as happens in testing/eval,
# and the second handles the case where it is invoked from {PROJECT_DIR}/packages/splunk-react-app/stage, as happens in-app.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../packages/splunk-react-app/src/main/resources/splunk", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../..", "lib"))
from distogram import Distogram, frequency_density_distribution, update

from saad.algs.adesca.data_prepare import (
    COL_VALUE,
    COL_ANOMALY_SCORE,
    COL_ANOMALY_LABEL,
    sigmoid,
    calc_iqr
)

ANOMALY_ENSEMBLE = "anomaly_ensemble"

def detect_anomaly(df_in, anomaly_score_thresh):
    """
    Anomaly Detection via Streaming Histogram

    Keep this around until we refactor the codebase to use the StreamHist() class in AutoAnomalyDetection; this function is currently in production.
    """
    df = df_in.copy()
    values = df[COL_VALUE].to_numpy()
    maxAnomalies = 2**max(0, math.floor(math.log(len(values), 10)) - 1)
    
    # Populate streaming histogram
    hist = Distogram()
    for value in values:
        hist = update(hist, value)
    
    # Get pointwise log-probabilities
    densities = distogram_pdf(hist, values)
    log_probs = np.array([math.log10(density + 1e-8) for density in densities])

    # IQR method to establish threshold
    q1 = np.quantile(log_probs, 0.25)
    iqr = calc_iqr(log_probs)
    thresh = q1 - 1.5 * iqr

    # Threshold the log-probabilities
    anoms = (log_probs <= thresh).astype(int)
    if anoms.sum() > maxAnomalies: # Only return the smallest `maxAnomalies` number of log-probabilities
        thresh = log_probs[np.argsort(log_probs)[maxAnomalies]]
        anoms = (log_probs <= thresh).astype(int)
    
    df[ANOMALY_ENSEMBLE] = anoms

    # Confidence score: how many IQRs below the threshold did the point lie? 
    # Confidence will be <= 0.5 for non-anoms and > 0.5 for anoms.
    # TODO: Calibrate this more thoroughly
    df[COL_ANOMALY_SCORE] = [sigmoid(num_iqrs_below, trans=0) for num_iqrs_below in (thresh - log_probs) / iqr]
    df[ANOMALY_ENSEMBLE] = df[ANOMALY_ENSEMBLE] * (
        df[COL_ANOMALY_SCORE] > anomaly_score_thresh
    ).astype(int)

    return df


def distogram_pdf(distogram, values):
    """ Returns estimates, under a given distogram, of the probability densities for a collection of given values

    Args:
        distogram: A Distogram object to be used for estimating probability densities.
        values: A list of floats, the values whose densities are to be estimated.

    Returns:
        A list of floats (the same length as `values`), the probability density estimates.
    """

    def get_bin_idx(distogram, value):
        """ Gets the index of the bin in which `value` belongs.

        If `value` lies out-of-range of (below or above) the distogram bounds, simply assign it to either the first or last
        bin respectively. While this approach seems to discard information (points ~below~ the lowest bin should intuitively 
        have smaller densities than points ~in~ the lowest bin), it consistently seems to yield better anomaly detection results.
        """
        if value <= distogram.bins[0][0]: # Lower than lowest bin
            index = 0
        elif value >= distogram.bins[-1][0]: # Higher than highest bin
            index = -1
        else: # Search for bin
            index = bisect_left(distogram.bins, (value, 1)) - 1 # Subtract 1 because `bin_densities` has one fewer element than `h.bins`
        return index

    # Case: No values provided - return empty list
    if len(values) == 0:
        return []

    # Case: Empty distogram - return uniform densities
    elif len(distogram.bins) == 0:
        return [1.0 / len(values)] * len(values)

    # Case: The max number of distogram bins have not all been populated yet, so `distogram.bins` is still representing 
    # exact value frequencies rather than bin boundaries and approximate frequencies. 
    elif len(distogram.bins) < distogram.bin_count:
        count_dict = {val: count for val, count in distogram.bins}
        total_count = float(np.sum([count for _, count in distogram.bins]))
        # Values get zero density if they are not in the distogram, and freq/total_count density if they are.
        return np.array([count_dict[val] / total_count if val in count_dict else 0.0 for val in values])

    # Case:  distogram bins - get density for each value, and normalize
    else:
        bin_densities, _ = frequency_density_distribution(distogram)
        total_density = np.sum(bin_densities)
        return [bin_densities[get_bin_idx(distogram, value)] / total_density for value in values]


def compute_density(distogram, value):
    """
    * NOT CURRENTLY IN USE *

    Alternate way of computing densities that seems more accurate empirically but gives worse AD results
    """
    total_count = float(np.sum([count for _, count in distogram.bins]))

    if value <= distogram.bins[0][0]: # Lower than lowest bin
        return distogram.bins[0][1] / total_count
    elif value >= distogram.bins[-1][0]: # Higher than highest bin
        return distogram.bins[-1][1] / total_count
    else: # Search for bin
        i = bisect_left(distogram.bins, (value, 1)) # Value lies in-between bins i-1 and i
        left_val, left_count = distogram.bins[i-1]
        right_val, right_count = distogram.bins[i]
        frac_from_left_to_right = (value - left_val) / (right_val - left_val)
        return (frac_from_left_to_right * right_count + (1 - frac_from_left_to_right) * left_count) / total_count