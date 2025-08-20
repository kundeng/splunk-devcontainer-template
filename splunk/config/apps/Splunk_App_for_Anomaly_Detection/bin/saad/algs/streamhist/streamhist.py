import numpy as np
import pandas as pd
import math 

from distogram import Distogram, update
from saad.algs.streamhist.streamhist_util import detect_anomaly, distogram_pdf
from saad.algs.adesca.data_prepare import (
    sigmoid,
    calc_iqr,
    COL_VALUE,
)
from saad.algs.base.anomaly_detector import AnomalyDetector, AnomalyDetectorException
from saad.utils import setup_logging

logger = setup_logging.get_logger()


class StreamHist(AnomalyDetector):
    def __init__(self):
        self.distogram = None
        self.log_prob_iqr = None
        self.log_prob_thresh = None


    def fit(self, df):
        self.distogram = Distogram()

        # Populate distogram
        values = df[COL_VALUE].to_numpy()
        for value in values:
            self.distogram = update(self.distogram, value)
        
        # IQR method to establish threshold (a reasonable boundary for separating anomalies from non-anomalies)
        # for calibrating confidence scoring based on initial fit data's log probabilities
        log_probs = get_log_probs(self.distogram, values)
        self.log_prob_thresh, self.log_prob_iqr = establish_threshold_and_iqr(log_probs)
        

    def predict(self, df):
        if self.distogram is None:
            raise AnomalyDetectorException("StreamHist model must be fit before it can predict.")

        # Get pointwise log-probabilities and return scores
        values = df[COL_VALUE].to_numpy()
        log_probs = get_log_probs(self.distogram, values)
        return score_anomalies(log_probs, self.log_prob_thresh, self.log_prob_iqr)


    def fit_predict(self, df):
        """
        This implementation is slightly more efficient than calling `fit` and `predict` in succession because it doesn't 
        unnecessarily call `get_log_probs()` twice. 
        """
        self.distogram = Distogram()

        # Populate distogram
        values = df[COL_VALUE].to_numpy()
        for value in values:
            self.distogram = update(self.distogram, value)

        # IQR method to establish threshold (a reasonable boundary for separating anomalies from non-anomalies)
        # for calibrating confidence scoring based on initial fit data's log probabilities
        log_probs = get_log_probs(self.distogram, values)
        self.log_prob_thresh, self.log_prob_iqr = establish_threshold_and_iqr(log_probs)

        # Return scores
        return score_anomalies(log_probs, self.log_prob_thresh, self.log_prob_iqr)


    def partial_fit(self, df):
        if self.distogram is None:
            raise AnomalyDetectorException("StreamHist model must be fit before it can be partial_fit.")

        # Populate distogram
        values = df[COL_VALUE].to_numpy()
        for value in values:
            self.distogram = update(self.distogram, value)


def get_log_probs(distogram, values):
    """
    Get pointwise log-probabilities for the values in `values` based on their densities in `distogram`.
    """
    densities = distogram_pdf(distogram, values)
    log_probs = np.array([math.log10(density + 1e-8) for density in densities])
    return log_probs


def establish_threshold_and_iqr(log_probs):
    """
    Establish the IQR and calibration threshold on the log-probabilities. Points with log-probabilities below the threshold will 
    get anomaly scores > 0.5, proportional to how many IQRs they are below the threshold. Threshold calculation is based
    on the methodology in https://splunk.atlassian.net/wiki/spaces/PROD/pages/313542900753/Anomaly+Detection+Command.
    """
    log_prob_iqr = calc_iqr(log_probs)
    log_prob_thresh = np.quantile(log_probs, 0.25) - 1.5 * log_prob_iqr
    maxAnomalies = 2**max(0, math.floor(math.log(len(log_probs), 10)) - 1)
    if (log_probs <= log_prob_thresh).astype(int).sum() > maxAnomalies:
        log_prob_thresh = log_probs[np.argsort(log_probs)[maxAnomalies]]
    return log_prob_thresh, log_prob_iqr


def score_anomalies(log_probs, log_prob_thresh, log_prob_iqr):
    """
    Anomaly scores: how many IQRs below the threshold did the point's log_prob fall? 
    Confidence will be <= 0.5 for points with log_probs above the threshold, and >0.5 for points with log_probs below the threshold.
    """
    return np.array([sigmoid(num_iqrs_below, trans=0) for num_iqrs_below in (log_prob_thresh - log_probs) / log_prob_iqr])