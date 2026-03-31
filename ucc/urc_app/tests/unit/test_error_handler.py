"""Tests for urc.components.error_handler — error handling and backoff."""

import json

import requests

from urc.components.error_handler import (
    ConstantBackoffStrategy,
    ExponentialBackoffStrategy,
    WaitTimeFromHeader,
    HttpResponseFilter,
    DefaultErrorHandler,
    CompositeErrorHandler,
)


def _make_response(status=200, body=None, headers=None):
    resp = requests.models.Response()
    resp.status_code = status
    resp._content = json.dumps(body or {}).encode()
    resp.headers["Content-Type"] = "application/json"
    if headers:
        resp.headers.update(headers)
    return resp


# ── ConstantBackoffStrategy ──


def test_constant_backoff_returns_configured_time():
    s = ConstantBackoffStrategy({"backoff_time_in_seconds": 10}, {})
    resp = _make_response()
    assert s.get_backoff_time(resp, 0) == 10.0
    assert s.get_backoff_time(resp, 5) == 10.0


def test_constant_backoff_default():
    s = ConstantBackoffStrategy({}, {})
    resp = _make_response()
    assert s.get_backoff_time(resp, 0) == 5.0


# ── ExponentialBackoffStrategy ──


def test_exponential_backoff_formula():
    s = ExponentialBackoffStrategy({"factor": 2}, {})
    resp = _make_response()
    assert s.get_backoff_time(resp, 0) == 2.0  # 2 * 2^0
    assert s.get_backoff_time(resp, 1) == 4.0  # 2 * 2^1
    assert s.get_backoff_time(resp, 2) == 8.0  # 2 * 2^2
    assert s.get_backoff_time(resp, 3) == 16.0  # 2 * 2^3


def test_exponential_backoff_default_factor():
    s = ExponentialBackoffStrategy({}, {})
    resp = _make_response()
    assert s.get_backoff_time(resp, 0) == 5.0  # factor=5, 5*2^0


# ── WaitTimeFromHeader ──


def test_wait_time_from_header_basic():
    s = WaitTimeFromHeader({"header": "Retry-After"}, {})
    resp = _make_response(headers={"Retry-After": "30"})
    assert s.get_backoff_time(resp, 0) == 30.0


def test_wait_time_from_header_missing():
    s = WaitTimeFromHeader({"header": "Retry-After"}, {})
    resp = _make_response()
    assert s.get_backoff_time(resp, 0) is None


def test_wait_time_from_header_regex():
    s = WaitTimeFromHeader(
        {"header": "X-RateLimit-Reset", "regex": r"in (\d+) seconds"},
        {},
    )
    resp = _make_response(headers={"X-RateLimit-Reset": "retry in 45 seconds"})
    assert s.get_wait_time(resp) == 45.0


def test_wait_time_from_header_max_wait():
    s = WaitTimeFromHeader(
        {"header": "Retry-After", "max_waiting_time_in_seconds": 10},
        {},
    )
    resp = _make_response(headers={"Retry-After": "60"})
    assert s.get_wait_time(resp) == 10.0


# ── HttpResponseFilter ──


def test_filter_matches_http_codes():
    f = HttpResponseFilter({"http_codes": [429, 503], "action": "RETRY"}, {})
    assert f.matches(_make_response(429)) is True
    assert f.matches(_make_response(503)) is True
    assert f.matches(_make_response(200)) is False
    assert f.matches(_make_response(500)) is False


def test_filter_matches_error_message():
    f = HttpResponseFilter(
        {"error_message_contains": "rate limit", "action": "RETRY"},
        {},
    )
    resp_match = _make_response(429, body={"error": "rate limit exceeded"})
    resp_no_match = _make_response(429, body={"error": "unauthorized"})
    assert f.matches(resp_match) is True
    assert f.matches(resp_no_match) is False


def test_filter_action_property():
    f = HttpResponseFilter({"action": "FAIL"}, {})
    assert f.action == "FAIL"


# ── DefaultErrorHandler ──


def test_default_handler_retries_429():
    h = DefaultErrorHandler({"type": "DefaultErrorHandler"}, {})
    assert h.should_retry(_make_response(429)) is True


def test_default_handler_retries_5xx():
    h = DefaultErrorHandler({"type": "DefaultErrorHandler"}, {})
    assert h.should_retry(_make_response(500)) is True
    assert h.should_retry(_make_response(502)) is True
    assert h.should_retry(_make_response(503)) is True


def test_default_handler_does_not_retry_4xx():
    h = DefaultErrorHandler({"type": "DefaultErrorHandler"}, {})
    assert h.should_retry(_make_response(400)) is False
    assert h.should_retry(_make_response(401)) is False
    assert h.should_retry(_make_response(404)) is False


def test_default_handler_does_not_retry_200():
    h = DefaultErrorHandler({"type": "DefaultErrorHandler"}, {})
    assert h.should_retry(_make_response(200)) is False


def test_default_handler_custom_filter():
    h = DefaultErrorHandler(
        {
            "type": "DefaultErrorHandler",
            "response_filters": [
                {
                    "type": "HttpResponseFilter",
                    "http_codes": [403],
                    "action": "RETRY",
                }
            ],
        },
        {},
    )
    assert h.should_retry(_make_response(403)) is True


def test_default_handler_custom_filter_fail_action():
    h = DefaultErrorHandler(
        {
            "type": "DefaultErrorHandler",
            "response_filters": [
                {
                    "type": "HttpResponseFilter",
                    "http_codes": [429],
                    "action": "FAIL",
                }
            ],
        },
        {},
    )
    # Custom filter overrides default 429 retry
    assert h.should_retry(_make_response(429)) is False


def test_default_handler_backoff_time():
    h = DefaultErrorHandler(
        {
            "type": "DefaultErrorHandler",
            "backoff_strategies": [
                {"type": "ConstantBackoffStrategy", "backoff_time_in_seconds": 7},
            ],
        },
        {},
    )
    resp = _make_response(429)
    assert h.get_backoff_time(resp, 0) == 7.0


def test_default_handler_max_retries():
    h = DefaultErrorHandler({"type": "DefaultErrorHandler", "max_retries": 3}, {})
    assert h.max_retries == 3


# ── CompositeErrorHandler ──


def test_composite_handler_delegates():
    h = CompositeErrorHandler(
        {
            "type": "CompositeErrorHandler",
            "error_handlers": [
                {
                    "type": "DefaultErrorHandler",
                    "response_filters": [
                        {
                            "type": "HttpResponseFilter",
                            "http_codes": [418],
                            "action": "RETRY",
                        }
                    ],
                },
            ],
        },
        {},
    )
    assert h.should_retry(_make_response(418)) is True


def test_composite_handler_fallback_on_5xx():
    h = CompositeErrorHandler(
        {
            "type": "CompositeErrorHandler",
            "error_handlers": [],
        },
        {},
    )
    assert h.should_retry(_make_response(500)) is True
    assert h.should_retry(_make_response(200)) is False
