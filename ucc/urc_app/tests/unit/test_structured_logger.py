"""Tests for urc.structured_logger — Splunk key=value emitter."""

import logging

from urc.structured_logger import emit, _quote, after_hook, error_hook


# ── _quote ──


def test_quote_simple_value():
    assert _quote("hello") == "hello"


def test_quote_value_with_spaces():
    assert _quote("hello world") == '"hello world"'


def test_quote_value_with_equals():
    assert _quote("key=val") == '"key=val"'


def test_quote_value_with_quotes():
    result = _quote('say "hi"')
    assert '"' in result


def test_quote_numeric():
    assert _quote(42) == "42"
    assert _quote(3.14) == "3.14"


# ── emit ──


def test_emit_produces_key_value_log(caplog):
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        emit(action="test", component="Foo", elapsed="0.123")

    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "action=test" in msg
    assert "component=Foo" in msg
    assert "elapsed=0.123" in msg


def test_emit_filters_none_values(caplog):
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        emit(action="test", missing=None, present="yes")

    msg = caplog.records[0].message
    assert "missing" not in msg
    assert "present=yes" in msg


def test_emit_empty_fields_no_log(caplog):
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        emit()  # no fields

    assert len(caplog.records) == 0


# ── after_hook ──


def test_after_hook_emits_call(caplog):
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        after_hook("HttpRequester", "send", {"status": 200}, 0.285, {})

    msg = caplog.records[0].message
    assert "action=call" in msg
    assert "component=HttpRequester" in msg
    assert "method=send" in msg
    assert "elapsed=0.285" in msg


# ── error_hook ──


def test_error_hook_emits_error(caplog):
    exc = ValueError("something went wrong")
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        error_hook("CsvDecoder", "decode", exc, 0.050, {})

    msg = caplog.records[0].message
    assert "action=error" in msg
    assert "component=CsvDecoder" in msg
    assert "error_type=ValueError" in msg
    assert "something went wrong" in msg


def test_error_hook_truncates_long_messages(caplog):
    exc = ValueError("x" * 500)
    with caplog.at_level(logging.INFO, logger="urc.engine"):
        error_hook("Comp", "method", exc, 0.0, {})

    msg = caplog.records[0].message
    # Error message should be truncated to 200 chars
    assert len(msg) < 500
