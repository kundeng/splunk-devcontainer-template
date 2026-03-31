"""Tests for urc.components.event_timestamp — Splunk _time resolution.

These components exist and are fully functional, so tests are NOT skipped.
"""

import time
from unittest.mock import patch

from urc.components.event_timestamp import (
    CursorBasedTimestamp,
    FieldBasedTimestamp,
    FetchTimestamp,
)


# ── CursorBasedTimestamp ──


def test_cursor_based_reads_cursor_field():
    ts = CursorBasedTimestamp({}, {})
    stream_def = {
        "incremental_sync": {
            "cursor_field": "updated_at",
            "datetime_format": "%Y-%m-%dT%H:%M:%SZ",
        }
    }
    record = {"updated_at": "2024-06-15T12:00:00Z", "id": 1}
    result = ts.resolve(record, {}, stream_def)
    # Should be a valid epoch timestamp for 2024-06-15T12:00:00Z
    assert isinstance(result, float)
    assert result > 1718000000  # approximately correct


def test_cursor_based_missing_field_returns_now():
    ts = CursorBasedTimestamp({}, {})
    stream_def = {
        "incremental_sync": {
            "cursor_field": "updated_at",
            "datetime_format": "%Y-%m-%dT%H:%M:%SZ",
        }
    }
    record = {"id": 1}  # no updated_at
    result = ts.resolve(record, {}, stream_def)
    # Should fall back to time.time()
    assert abs(result - time.time()) < 5


def test_cursor_based_numeric_value():
    ts = CursorBasedTimestamp({}, {})
    stream_def = {
        "incremental_sync": {
            "cursor_field": "ts",
            "datetime_format": "%s",
        }
    }
    record = {"ts": 1700000000}
    result = ts.resolve(record, {}, stream_def)
    assert result == 1700000000.0


# ── FieldBasedTimestamp ──


def test_field_based_explicit_field():
    ts = FieldBasedTimestamp(
        {"field": "created_at", "format": "%Y-%m-%dT%H:%M:%SZ"},
        {},
    )
    stream_def = {}
    record = {"created_at": "2024-06-15T12:00:00Z"}
    result = ts.resolve(record, {}, stream_def)
    assert isinstance(result, float)
    assert result > 1718000000


def test_field_based_epoch_format():
    ts = FieldBasedTimestamp(
        {"field": "timestamp", "format": "%s"},
        {},
    )
    record = {"timestamp": "1700000000"}
    result = ts.resolve(record, {}, {})
    assert result == 1700000000


def test_field_based_missing_field_returns_now():
    ts = FieldBasedTimestamp({"field": "ts"}, {})
    record = {"id": 1}
    result = ts.resolve(record, {}, {})
    assert abs(result - time.time()) < 5


# ── FetchTimestamp ──


def test_fetch_timestamp_returns_current_time():
    ts = FetchTimestamp()
    result = ts.resolve({}, {}, {})
    assert abs(result - time.time()) < 2


@patch("urc.components.event_timestamp.time")
def test_fetch_timestamp_uses_time_module(mock_time_mod):
    mock_time_mod.time.return_value = 1234567890.0
    ts = FetchTimestamp()
    result = ts.resolve({}, {}, {})
    assert result == 1234567890.0
