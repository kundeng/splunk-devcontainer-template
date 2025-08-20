# Base modules
import os
import sys
import time
from datetime import datetime

# PSC modules
import numpy as np
import pandas as pd

# MLTK/MLSPL modules
mltk_root_dir = os.getenv("MLTK_ROOT_DIR")
if mltk_root_dir is not None:
    sys.path.append(f'{mltk_root_dir}/bin')  # Needed for local runs
from base import BaseAlgo
from cexc import get_logger, get_messages_logger

# ADESCA modules
from saad.algs.adesca.detectors import (
    detect_anomaly,
    ENSEMBLE_STREAMHIST,
    OUTPUT_ANOMALY_COL,
    COL_ANOMALY_SCORE,
)
from saad.algs.adesca.threshold_utils import (
    SENSITIVITY_LOW,
    SENSITIVITY_HIGH,
    SENSITIVITY_MID,
)
from saad.utils.parsing_utils import (
    standardize_df,
    DEFAULT_TIME_FIELD_NAME,
    DEFAULT_VALUE_FIELD_NAME
)
from saad.utils.setup_logging import ANOMALY_APP_TELEMETRY

logger = get_logger(__name__)
messages = get_messages_logger()


class AutoAnomalyDetection(BaseAlgo):
    """
    Custom ML-SPL algorithm for AnomalyApp (the Splunk App for Anomaly Detection).
    Returns:
        - output_df: (pandas.Dataframe) result of anomalies detected on the timeseries data
    """

    def __init__(self, options):
        logger.info("algo_name=AutoAnomalyDetection, params={null}")

        feature_variables = options.get("feature_variables", [])
        if len(feature_variables) != 1:
            logger.error(
                f"{ANOMALY_APP_TELEMETRY} fit invoked with wrong number (!= 1) of arguments"
            )
            raise RuntimeError(
                "AutoAnomalyDetection can only (and must) be called with a single input field name."
            )
        self.feature_variable = feature_variables[0]
        self.target_variable = "isOutlier"

        params = options.get("params", {})
        if "job_name" not in params:
            raise RuntimeError(
                "AutoAnomalyDetection must be provided with a `job_name` argument."
            )
        self.job_name = params["job_name"]
        self._validate_sensitivity(params["sensitivity"])

    def _validate_sensitivity(self, sensitivity_in):
        try:
            self.sensitivity = int(sensitivity_in)
        except ValueError as e:
            # "invalid literal for int()" meaning it wasn't digits
            raise RuntimeError(e)

        if self.sensitivity > SENSITIVITY_HIGH or self.sensitivity < SENSITIVITY_LOW:
            raise RuntimeError(
                f"Sensitivity param must equal either {SENSITIVITY_LOW}, {SENSITIVITY_MID}, or {SENSITIVITY_HIGH}."
            )

    def _get_relevant_fields(self, df):
        """Drop irrelevant fields from the dataframe.
        Returns: Dataframe with valid feilds.
        """
        to_drop = []
        required_fields = [self.feature_variable]
        for f in df.columns:
            if f.startswith("__mv_") and f not in required_fields:
                to_drop.append(f)
        df.drop(to_drop, axis=1, inplace=True)
        return df



    def _process_columns(self, df, dlength):
        """
        Temp function to add `isOutlier` feild to the Dataset
        Will be replaced with a call to function that fetches anomalies.
        """
        df.rename(
            columns={
                OUTPUT_ANOMALY_COL: self.target_variable,
                COL_ANOMALY_SCORE: "anomConf",
                "value": self.feature_variable,
                "timestamp": "_time",
            },
            inplace=True,
        )
        df["anomConf"] = df["anomConf"].apply(lambda val: round(val, 2))

        return df


    def fit(self, df, options):
        start = time.time()
        time_series = np.array(df[self.feature_variable])

        self._validate_sensitivity(options.get("sensitivity", self.sensitivity))
        logger.info(f"{ANOMALY_APP_TELEMETRY} Sensitivity Parameter: {self.sensitivity}")

        data_length = len(time_series)
        if data_length == 840 and self.feature_variable == "value":
            logger.info(f"{ANOMALY_APP_TELEMETRY} Using our included inputlookup data")
        else:
            logger.info(f"{ANOMALY_APP_TELEMETRY} Using custom data")

        # Figure out the time column, and invoke the alg
        time_col = (
            "time"
            if "time" in df.columns
            else "timestamp"
            if "timestamp" in df.columns
            else "_time"
        )

        # Store the original data
        original_df = df.copy()

        df = self._get_relevant_fields(df)
        input_df = pd.DataFrame({
            DEFAULT_TIME_FIELD_NAME: df[time_col],
            DEFAULT_VALUE_FIELD_NAME: df[self.feature_variable]
        })
        data_df = standardize_df(input_df)

        try:
            # Run detector
            result_df, _ = detect_anomaly(
                data_df, ensemble=ENSEMBLE_STREAMHIST, sensitivity=self.sensitivity
            )
        except Exception as exe:
            raise RuntimeError(f"Error: {str(exe)}")

        # Process columns for anomaly detection results
        result_df = self._process_columns(result_df, data_length)

        # Convert the '_time' columns of both dataframes to strings to ensure consistent datatypes.
        # This step addresses potential type mismatches during the merge operation, where one dataframe might have 
        # '_time' as a datetime object and the other as a string.
        original_df['_time'] = original_df['_time'].astype(str)
        result_df['_time'] = result_df['_time'].astype(str)

        # Merge the anomaly detection results with the original data
        output_df = pd.merge(original_df, result_df[["_time", "isOutlier", "anomConf"]], on="_time", how="left")

        end = time.time()
        logger.info(
            f"{ANOMALY_APP_TELEMETRY} Total execution time in seconds for `fit AutoAnomalyDetection` call: {end-start}"
        )
        return output_df