"""Integration tests: incremental sync with per-partition state."""

import json

import responses

from urc.engine import collect


# ── First run (no checkpoint) ──

INCREMENTAL_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: events
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/events"
        http_method: GET
        request_parameters:
          since: "{{ stream_slice.get('start_time', '') }}"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: updated_at
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
"""


def test_first_run_no_checkpoint(activate_responses):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/events",
             json=[
                 {"id": 1, "updated_at": "2024-03-01T00:00:00Z"},
                 {"id": 2, "updated_at": "2024-03-15T00:00:00Z"},
             ], status=200)

    config = {"base_url": "https://mock-api.test"}
    results = list(collect(INCREMENTAL_MANIFEST, config, checkpoint=None))

    all_states = [st for _, _, st in results if st]
    assert len(all_states) > 0
    final_state = all_states[-1]
    assert final_state["cursor_value"] == "2024-03-15T00:00:00Z"


# ── Resume from checkpoint ──


def test_resume_from_checkpoint(activate_responses):
    rsps = activate_responses

    def callback(request):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        since = qs.get("since", [""])[0]
        # If resuming, since should be the checkpoint value
        if since and since >= "2024-03-15":
            return (200, {}, json.dumps([
                {"id": 3, "updated_at": "2024-04-01T00:00:00Z"},
            ]))
        return (200, {}, json.dumps([
            {"id": 1, "updated_at": "2024-03-01T00:00:00Z"},
            {"id": 2, "updated_at": "2024-03-15T00:00:00Z"},
        ]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/events",
                       callback=callback, content_type="application/json")

    config = {"base_url": "https://mock-api.test"}
    checkpoint = {
        "events": {
            "cursor_field": "updated_at",
            "cursor_value": "2024-03-15T00:00:00Z",
        }
    }

    results = list(collect(INCREMENTAL_MANIFEST, config, checkpoint=checkpoint))
    data_records = [rec for _, rec, _ in results if rec]
    # Should only get the new record
    assert len(data_records) == 1
    assert data_records[0]["id"] == 3

    all_states = [st for _, _, st in results if st]
    final_state = all_states[-1]
    assert final_state["cursor_value"] == "2024-04-01T00:00:00Z"


# ── Per-partition state with ListPartitionRouter ──

PARTITION_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: events_by_region
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/events"
        http_method: GET
        request_parameters:
          region: "{{ stream_partition['region'] }}"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
      partition_router:
        type: ListPartitionRouter
        cursor_field: region
        values:
          - us-east
          - us-west
          - eu-west
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: updated_at
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
"""


def test_per_partition_state(activate_responses):
    rsps = activate_responses

    region_data = {
        "us-east": [{"id": 1, "updated_at": "2024-06-01T00:00:00Z"}],
        "us-west": [{"id": 2, "updated_at": "2024-07-01T00:00:00Z"}],
        "eu-west": [{"id": 3, "updated_at": "2024-08-01T00:00:00Z"}],
    }

    def callback(request):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        region = qs.get("region", [""])[0]
        data = region_data.get(region, [])
        return (200, {}, json.dumps(data))

    rsps.add_callback(rsps.GET, "https://mock-api.test/events",
                       callback=callback, content_type="application/json")

    config = {"base_url": "https://mock-api.test"}
    results = list(collect(PARTITION_MANIFEST, config))

    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 3

    # Get final state
    all_states = [st for _, _, st in results if st]
    final_state = all_states[-1]

    # Should have per-partition state (not flat)
    assert "partitions" in final_state
    assert len(final_state["partitions"]) == 3

    # Each partition should have its own cursor value
    cursor_values = set()
    for pk, pdata in final_state["partitions"].items():
        cursor_values.add(pdata["cursor"]["cursor_value"])
    assert "2024-06-01T00:00:00Z" in cursor_values
    assert "2024-07-01T00:00:00Z" in cursor_values
    assert "2024-08-01T00:00:00Z" in cursor_values


# ── State round-trip ──


def test_state_round_trip(activate_responses):
    """Collect → get state → use as checkpoint → collect again."""
    rsps = activate_responses

    call_count = {"n": 0}

    def callback(request):
        call_count["n"] += 1
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        since = qs.get("since", [""])[0]
        if since and since >= "2024-05-01":
            return (200, {}, json.dumps([
                {"id": 10, "updated_at": "2024-06-01T00:00:00Z"},
            ]))
        return (200, {}, json.dumps([
            {"id": 1, "updated_at": "2024-05-01T00:00:00Z"},
        ]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/events",
                       callback=callback, content_type="application/json")

    config = {"base_url": "https://mock-api.test"}

    # First run
    results1 = list(collect(INCREMENTAL_MANIFEST, config))
    states1 = [st for _, _, st in results1 if st]
    final_state1 = states1[-1]
    assert final_state1["cursor_value"] == "2024-05-01T00:00:00Z"

    # Second run with checkpoint
    checkpoint = {"events": final_state1}
    results2 = list(collect(INCREMENTAL_MANIFEST, config, checkpoint=checkpoint))
    data2 = [rec for _, rec, _ in results2 if rec]
    assert len(data2) == 1
    assert data2[0]["id"] == 10

    states2 = [st for _, _, st in results2 if st]
    final_state2 = states2[-1]
    assert final_state2["cursor_value"] == "2024-06-01T00:00:00Z"


# ── 10+ partitions fidelity ──

MANY_PARTITIONS_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: multi_partition
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/data"
        http_method: GET
        request_parameters:
          partition: "{{ stream_partition['p'] }}"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
      partition_router:
        type: ListPartitionRouter
        cursor_field: p
        values:
          - p01
          - p02
          - p03
          - p04
          - p05
          - p06
          - p07
          - p08
          - p09
          - p10
          - p11
          - p12
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: ts
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
"""


def test_many_partitions_fidelity(activate_responses):
    rsps = activate_responses

    def callback(request):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        p = qs.get("partition", [""])[0]
        # Each partition returns a record with a unique timestamp
        num = int(p.replace("p", ""))
        return (200, {}, json.dumps([
            {"id": num, "ts": f"2024-{num:02d}-15T00:00:00Z"},
        ]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/data",
                       callback=callback, content_type="application/json")

    config = {"base_url": "https://mock-api.test"}
    results = list(collect(MANY_PARTITIONS_MANIFEST, config))

    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 12

    all_states = [st for _, _, st in results if st]
    final_state = all_states[-1]

    assert "partitions" in final_state
    assert len(final_state["partitions"]) == 12

    # Verify each partition has distinct cursor
    cursor_values = [pdata["cursor"]["cursor_value"]
                     for pdata in final_state["partitions"].values()]
    assert len(set(cursor_values)) == 12  # All unique
