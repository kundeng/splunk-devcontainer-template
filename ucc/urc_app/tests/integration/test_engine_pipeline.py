"""Integration tests: full engine pipeline with mocked HTTP."""

import json
import time

import responses

from urc.engine import collect

from fixtures.sample_manifests import (
    SIMPLE_GET_MANIFEST,
    PAGINATED_OFFSET_MANIFEST,
    BEARER_AUTH_MANIFEST,
    TRANSFORM_MANIFEST,
    INCREMENTAL_MANIFEST,
)


# ── Simple single-page GET ──


def test_simple_get_pipeline(activate_responses, mock_config):
    rsps = activate_responses
    records_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    rsps.add(rsps.GET, "https://mock-api.test/users", json=records_data, status=200)

    results = list(collect(SIMPLE_GET_MANIFEST, mock_config))

    # Filter out empty final-state records
    data_records = [(name, rec, st) for name, rec, st in results if rec]
    assert len(data_records) == 2
    assert data_records[0][0] == "users"
    assert data_records[0][1]["name"] == "Alice"
    # Every record should have _time
    for _, rec, _ in data_records:
        assert "_time" in rec
        assert isinstance(rec["_time"], float)


# ── Paginated (offset-based) ──


def test_paginated_pipeline(activate_responses, mock_config):
    rsps = activate_responses
    all_items = [{"id": i, "name": f"item_{i}"} for i in range(25)]

    def paginated_callback(request):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        offset = int(qs.get("offset", ["0"])[0])
        limit = int(qs.get("limit", ["10"])[0])
        page = all_items[offset:offset + limit]
        return (200, {}, json.dumps({"data": page}))

    rsps.add_callback(rsps.GET, "https://mock-api.test/items",
                       callback=paginated_callback, content_type="application/json")

    results = list(collect(PAGINATED_OFFSET_MANIFEST, mock_config))
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 25


# ── With Bearer auth ──


def test_bearer_auth_pipeline(activate_responses, mock_config):
    rsps = activate_responses

    def auth_callback(request):
        auth = request.headers.get("Authorization", "")
        if auth != "Bearer test-key-123":
            return (401, {}, json.dumps({"error": "unauthorized"}))
        return (200, {}, json.dumps([{"id": 1, "secret": "data"}]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/protected",
                       callback=auth_callback, content_type="application/json")

    results = list(collect(BEARER_AUTH_MANIFEST, mock_config))
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 1
    assert data_records[0]["secret"] == "data"


# ── With transformations ──


def test_transform_pipeline(activate_responses, mock_config):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/items",
             json=[{"internal_id": "x", "userName": "Alice", "emailAddress": "a@b.com"}],
             status=200)

    results = list(collect(TRANSFORM_MANIFEST, mock_config))
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 1
    rec = data_records[0]
    # AddFields added "source"
    assert rec.get("source") == "api"
    # RemoveFields removed "internal_id"
    assert "internal_id" not in rec
    # KeysToSnakeCase converted camelCase
    assert "user_name" in rec
    assert "email_address" in rec


# ── Incremental with _time from cursor ──


def test_incremental_with_time(activate_responses, mock_config):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/events",
             json=[
                 {"id": 1, "updated_at": "2024-06-15T10:30:00Z", "data": "event1"},
                 {"id": 2, "updated_at": "2024-06-16T14:00:00Z", "data": "event2"},
             ], status=200)

    results = list(collect(INCREMENTAL_MANIFEST, mock_config))
    data_records = [(name, rec, st) for name, rec, st in results if rec]
    assert len(data_records) == 2

    # _time should be derived from cursor field (updated_at), not fetch time
    import datetime
    expected_time1 = datetime.datetime(2024, 6, 15, 10, 30, 0,
                                        tzinfo=datetime.timezone.utc).timestamp()
    assert abs(data_records[0][1]["_time"] - expected_time1) < 1

    # Final state should have advanced cursor
    all_states = [st for _, _, st in results if st]
    final_state = all_states[-1]
    assert final_state["cursor_value"] == "2024-06-16T14:00:00Z"


# ── Empty stream ──


def test_empty_stream(activate_responses, mock_config):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/users", json=[], status=200)

    results = list(collect(SIMPLE_GET_MANIFEST, mock_config))
    # Only the final state record (empty)
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 0
