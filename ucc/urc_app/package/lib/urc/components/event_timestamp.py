# Event timestamp components — resolve _time for Splunk events.
# Splunk-specific extension; no Airbyte CDK equivalent.

import datetime
import logging
import time

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


def _parse_time(value, fmt):
    """Parse a timestamp value into an epoch float.

    Args:
        value: Raw value from the record (str, int, float, or None).
        fmt: strftime format string. Special values: "%s" (epoch seconds),
             "%ms" (epoch milliseconds).

    Returns:
        Epoch float suitable for Splunk _time.
    """
    if value is None:
        return time.time()

    if isinstance(value, (int, float)):
        return float(value)

    try:
        if fmt == "%s":
            return int(value)
        if fmt == "%ms":
            return int(value) / 1000.0

        dt = datetime.datetime.strptime(str(value), fmt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError, OverflowError) as exc:
        logger.warning("Failed to parse timestamp value=%r fmt=%r: %s", value, fmt, exc)
        return time.time()


@component("CursorBasedTimestamp")
class CursorBasedTimestamp:
    """Derive _time from the stream's incremental_sync cursor field."""

    def __init__(self, definition, config, **kwargs):
        self._config = config

    def resolve(self, record, config, stream_def):
        inc = stream_def.get("incremental_sync", {})
        field = eval_string(inc.get("cursor_field", ""), config)
        fmt = inc.get("datetime_format", "%Y-%m-%dT%H:%M:%SZ")
        return _parse_time(record.get(field), fmt)


@component("FieldBasedTimestamp")
class FieldBasedTimestamp:
    """Derive _time from an explicit record field with a given format."""

    def __init__(self, definition, config, **kwargs):
        self._field_template = definition.get("field", "")
        self._format_template = definition.get("format", "%Y-%m-%dT%H:%M:%SZ")

    def resolve(self, record, config, stream_def):
        field = eval_string(self._field_template, config)
        fmt = eval_string(self._format_template, config)
        return _parse_time(record.get(field), fmt)


@component("FetchTimestamp")
class FetchTimestamp:
    """Always return the current wall-clock time as _time."""

    def __init__(self, definition=None, config=None, **kwargs):
        pass

    def resolve(self, record, config, stream_def):
        return time.time()
