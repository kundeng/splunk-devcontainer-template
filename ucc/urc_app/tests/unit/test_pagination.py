"""Tests for urc.components.pagination — pagination strategies."""

import json
from typing import Optional

import requests

from urc.components.pagination import (
    NoPagination,
    OffsetIncrement,
    PageIncrement,
    CursorPagination,
    DefaultPaginator,
)


def make_response(status=200, json_data=None, headers=None):
    """Create a mock requests.Response."""
    resp = requests.models.Response()
    resp.status_code = status
    resp._content = json.dumps(json_data or {}).encode()
    resp.headers["Content-Type"] = "application/json"
    if headers:
        resp.headers.update(headers)
    return resp


# ── NoPagination ──


def test_no_pagination_always_none():
    pag = NoPagination({}, {})
    resp = make_response()
    assert pag.next_page(resp, {}, [{"id": 1}], 0) is None
    assert pag.next_page(resp, {}, [], 5) is None


# ── OffsetIncrement ──


def test_offset_increment_advances():
    pag = OffsetIncrement({"page_size": 10}, {})
    resp = make_response()
    records = [{"id": i} for i in range(10)]  # full page
    token = pag.next_token(resp, records, 0)
    assert token == 10


def test_offset_increment_stops_on_short_page():
    pag = OffsetIncrement({"page_size": 10}, {})
    resp = make_response()
    records = [{"id": i} for i in range(5)]  # short page
    token = pag.next_token(resp, records, 0)
    assert token is None


def test_offset_increment_accumulates():
    pag = OffsetIncrement({"page_size": 5}, {})
    resp = make_response()
    full_page = [{"id": i} for i in range(5)]

    t1 = pag.next_token(resp, full_page, 0)
    assert t1 == 5
    t2 = pag.next_token(resp, full_page, 1)
    assert t2 == 10
    t3 = pag.next_token(resp, full_page, 2)
    assert t3 == 15


def test_offset_increment_page_size_from_config():
    config = {"page_size": "25"}
    pag = OffsetIncrement({"page_size": "{{ config['page_size'] }}"}, config)
    assert pag.page_size == 25


# ── PageIncrement ──


def test_page_increment_advances():
    pag = PageIncrement({"page_size": 10, "start_from_page": 1}, {})
    resp = make_response()
    records = [{"id": i} for i in range(10)]
    token = pag.next_token(resp, records, 0)
    assert token == 2  # start_from_page(1) + page_count(0) + 1


def test_page_increment_stops_on_short_page():
    pag = PageIncrement({"page_size": 10}, {})
    resp = make_response()
    records = [{"id": i} for i in range(3)]
    token = pag.next_token(resp, records, 0)
    assert token is None


def test_page_increment_default_start_from_zero():
    pag = PageIncrement({"page_size": 5}, {})
    resp = make_response()
    full_page = [{"id": i} for i in range(5)]
    token = pag.next_token(resp, full_page, 0)
    assert token == 1  # 0 + 0 + 1


# ── CursorPagination ──


def test_cursor_pagination_extracts_from_body():
    pag = CursorPagination(
        {"cursor_value": "{{ response['next_cursor'] }}"},
        {},
    )
    resp = make_response(json_data={"next_cursor": "abc123", "data": []})
    token = pag.next_token(resp, [], 0)
    assert token == "abc123"


def test_cursor_pagination_stops_when_cursor_is_none():
    pag = CursorPagination(
        {"cursor_value": "{{ response.get('next_cursor') }}"},
        {},
    )
    resp = make_response(json_data={"data": []})
    # response.get('next_cursor') returns None -> rendered as "None" -> token is None
    token = pag.next_token(resp, [], 0)
    assert token is None


def test_cursor_pagination_stop_condition():
    pag = CursorPagination(
        {
            "cursor_value": "{{ response['next'] }}",
            "stop_condition": "{{ response['has_more'] == false }}",
        },
        {},
    )
    resp = make_response(json_data={"next": "page2", "has_more": False})
    token = pag.next_token(resp, [], 0)
    assert token is None  # stop condition is true


def test_cursor_pagination_continues_when_stop_false():
    pag = CursorPagination(
        {
            "cursor_value": "{{ response['next'] }}",
            "stop_condition": "{{ response['has_more'] == false }}",
        },
        {},
    )
    resp = make_response(json_data={"next": "page2", "has_more": True})
    token = pag.next_token(resp, [], 0)
    assert token == "page2"


def test_cursor_pagination_extracts_from_headers():
    pag = CursorPagination(
        {"cursor_value": "{{ headers['X-Next-Page'] }}"},
        {},
    )
    resp = make_response(headers={"X-Next-Page": "next-page-token"})
    token = pag.next_token(resp, [], 0)
    assert token == "next-page-token"


# ── DefaultPaginator ──


def test_default_paginator_injects_token():
    definition = {
        "pagination_strategy": {
            "type": "OffsetIncrement",
            "page_size": 5,
        },
        "page_token_option": {
            "field_name": "offset",
            "inject_into": "request_parameter",
        },
        "page_size_option": {
            "field_name": "limit",
        },
    }
    pag = DefaultPaginator(definition, {})
    resp = make_response()
    records = [{"id": i} for i in range(5)]

    next_params = pag.next_page(resp, {}, records, 0)
    assert next_params is not None
    assert next_params["offset"] == 5
    assert next_params["limit"] == 5


def test_default_paginator_returns_none_on_last_page():
    definition = {
        "pagination_strategy": {
            "type": "OffsetIncrement",
            "page_size": 10,
        },
        "page_token_option": {
            "field_name": "offset",
            "inject_into": "request_parameter",
        },
    }
    pag = DefaultPaginator(definition, {})
    resp = make_response()
    records = [{"id": i} for i in range(3)]  # short page

    result = pag.next_page(resp, {}, records, 0)
    assert result is None


def test_default_paginator_request_path_injection():
    definition = {
        "pagination_strategy": {
            "type": "CursorPagination",
            "cursor_value": "{{ response['next_url'] }}",
        },
        "page_token_option": {
            "type": "RequestPath",
        },
    }
    pag = DefaultPaginator(definition, {})
    resp = make_response(json_data={"next_url": "https://api.test/page2"})

    next_params = pag.next_page(resp, {}, [{"id": 1}], 0)
    assert next_params is not None
    assert next_params["_path_override"] == "https://api.test/page2"
