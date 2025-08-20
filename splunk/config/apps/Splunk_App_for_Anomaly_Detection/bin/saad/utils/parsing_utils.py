from datetime import datetime, timezone

import math
import pandas as pd
from dateutil import parser


DEFAULT_TIME_FIELD_NAME = "timestamp"
DEFAULT_VALUE_FIELD_NAME = "value"


def infer_timestamp_format(ts):
    """
    Infers the format of the given timestamp string.
    Returns the string `s` if the timestamp is in UNIX seconds, `ms` if the timestamp is in UNIX milliseconds,
    and `datetime_str` if the timestamp is a human-readable datetime string. Raises a ValueError if the timestamp
    cannot be parsed.
    """
    ts = str(ts)

    if is_number(ts): # Numeric; either unix seconds or unix milliseconds
        val = float(ts.replace(",", "")) # Strip commas
        if val < 0:
            raise ValueError(f"Failed to parse timestamp: {ts}")
        # In seconds, this is year 2128 (by which time the UNIX convention will have been abandoned due to overflow). In milliseconds, it is February 1970.
        elif val < 5000000000: 
            return 's'
        else:
            return 'ms'

    else: # Non-numeric; see if it can be parsed as a datetime string
        try:
            parser.parse(ts)
            return 'datetime_str'
        except:
            raise ValueError(f"Failed to parse timestamp: {ts}")


def datetime_is_tz_aware(d):
    """
    Returns True if the datetime object `d` is timezone-aware, and False otherwise.
    Based directly on https://docs.python.org/3/library/datetime.html#determining-if-an-object-is-aware-or-naive.
    """
    return d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None


def timestamp_to_unix_ms(ts):
    """
    Converts a timestamp string to its UNIX representation in milliseconds
    """
    ts = str(ts)
    ts_format = infer_timestamp_format(ts)
    
    if ts_format == 'datetime_str':
        unix_s = parser.parse(ts).timestamp()
        return int(round(unix_s * 1000))

    else: # Numeric (unix seconds (`s`) or milliseconds (`ms`))
        val = float(ts.replace(",", "")) # Strip commas
        if ts_format == 's':
            val *= 1000
        return int(round(val))


def is_number(s):
    """
    Checks whether or not the input string `s` can be interpreted as a numeric value.
    """
    try:
        # Check that the string (with any commas removed) can be converted to a non-NaN float
        if not math.isnan(float(str(s).replace(",", ""))):
            return True  # Non-NaN float
        return False  # NaN
    except ValueError:
        return False  # Could not convert to float


def standardize_df(df_in: pd.DataFrame,
                   time_col: str = DEFAULT_TIME_FIELD_NAME,
                   val_col: str = DEFAULT_VALUE_FIELD_NAME):
    """
    Standardizes the input data before passing it on to the actual Anomaly alg.
    df: DataFrame with columns ["timestamp", "value"]
    """

    df = df_in.copy()

    # Filter out all non-numeric values
    df = df[df[val_col].apply(is_number)]

    # Randomly select a record for duplicate timestamps
    df = df.drop_duplicates(subset=time_col)

    # Sort data by the unix MS timestamp
    df['timestamp_ms'] = df[time_col].apply(timestamp_to_unix_ms)
    df = df.sort_values(by='timestamp_ms')

    # Reset index
    df = df.reset_index()[[time_col, val_col]]

    return df
