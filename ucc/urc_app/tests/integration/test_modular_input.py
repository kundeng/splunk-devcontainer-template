"""Integration tests: modular input with mocked Splunk SDK.

Simulates the full modular input lifecycle:
- Read credentials from Splunk passwords
- Read checkpoint from KV Store
- Run engine.collect() with mocked HTTP
- Write events with _time
- Save checkpoint to KV Store
"""

import json
import time as real_time
from unittest.mock import MagicMock, patch

import responses

from urc.engine import collect


# ── Thin modular input handler (the code under test) ──


def run_modular_input(service, input_config, manifest_yaml):
    """Simulate what a real Splunk modular input handler does.

    Args:
        service: Mocked splunklib.client.Service
        input_config: Dict with input settings
        manifest_yaml: YAML manifest string

    Returns:
        List of (event_data, event_time) tuples written
    """
    # 1. Read credentials from Splunk passwords
    creds = {}
    for credential in service.storage_passwords:
        if credential.realm == input_config.get("account", ""):
            creds["username"] = credential.username
            creds["password"] = credential.clear_password
            break

    # 2. Build config for engine
    config = {**input_config, **creds}

    # 3. Read checkpoint from KV Store
    checkpoint = None
    kv_collection = service.kvstore["checkpoints"].data
    existing = kv_collection.query(
        query=json.dumps({"_key": input_config["name"]})
    )
    if existing:
        checkpoint = json.loads(existing[0].get("state", "{}"))

    # 4. Run collection
    events_written = []
    last_state = None

    for stream_name, record, state in collect(manifest_yaml, config, checkpoint=checkpoint):
        if record:
            event_time = record.pop("_time", real_time.time())
            events_written.append((record, event_time))
        if state:
            last_state = state

    # 5. Save checkpoint
    if last_state:
        kv_collection.update(
            input_config["name"],
            {"_key": input_config["name"], "state": json.dumps(last_state)},
        )

    return events_written, last_state


MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: items
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/items"
        http_method: GET
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


def _build_mock_service(checkpoint_state=None):
    """Build a mocked Splunk Service with passwords and KV Store."""
    service = MagicMock()

    # Mock storage_passwords
    cred = MagicMock()
    cred.realm = "my_account"
    cred.username = "api_user"
    cred.clear_password = "api_secret"
    service.storage_passwords = [cred]

    # Mock KV Store
    kv_data = MagicMock()
    if checkpoint_state:
        kv_data.query.return_value = [
            {"_key": "test_input", "state": json.dumps(checkpoint_state)}
        ]
    else:
        kv_data.query.return_value = []
    service.kvstore.__getitem__ = MagicMock(
        return_value=MagicMock(data=kv_data)
    )

    return service, kv_data


# ── First run: no checkpoint ──


def test_first_run_saves_checkpoint(activate_responses):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/items",
             json=[
                 {"id": 1, "updated_at": "2024-05-01T00:00:00Z", "value": "a"},
                 {"id": 2, "updated_at": "2024-06-01T00:00:00Z", "value": "b"},
             ], status=200)

    service, kv_data = _build_mock_service(checkpoint_state=None)
    input_config = {"name": "test_input", "account": "my_account",
                     "base_url": "https://mock-api.test"}

    events, last_state = run_modular_input(service, input_config, MANIFEST)

    assert len(events) == 2
    assert last_state is not None
    assert last_state["cursor_value"] == "2024-06-01T00:00:00Z"

    # Verify KV Store update was called
    kv_data.update.assert_called_once()
    call_args = kv_data.update.call_args
    saved_state = json.loads(call_args[0][1]["state"])
    assert saved_state["cursor_value"] == "2024-06-01T00:00:00Z"


# ── Resume: checkpoint read ──


def test_resume_from_kv_store(activate_responses):
    rsps = activate_responses

    # The engine doesn't filter by cursor in the HTTP request itself —
    # that depends on the manifest's request_parameters using stream_slice.
    # The checkpoint just affects the cursor's starting value.
    # So we just verify the checkpoint is loaded and cursor advances.
    rsps.add(rsps.GET, "https://mock-api.test/items",
             json=[
                 {"id": 3, "updated_at": "2024-07-01T00:00:00Z", "value": "c"},
             ], status=200)

    checkpoint = {
        "items": {
            "cursor_field": "updated_at",
            "cursor_value": "2024-06-01T00:00:00Z",
        }
    }
    service, kv_data = _build_mock_service(checkpoint_state=checkpoint)
    input_config = {"name": "test_input", "account": "my_account",
                     "base_url": "https://mock-api.test"}

    events, last_state = run_modular_input(service, input_config, MANIFEST)

    assert len(events) == 1
    assert events[0][0]["id"] == 3
    # Cursor should have advanced past the checkpoint
    assert last_state["cursor_value"] == "2024-07-01T00:00:00Z"


# ── Rate limit recovery ──


def test_rate_limit_recovery(activate_responses):
    rsps = activate_responses

    call_count = {"n": 0}

    def callback(request):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return (429, {"Retry-After": "0"}, json.dumps({"error": "rate limited"}))
        return (200, {}, json.dumps([
            {"id": 1, "updated_at": "2024-05-01T00:00:00Z", "value": "recovered"},
        ]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/items",
                       callback=callback, content_type="application/json")

    service, kv_data = _build_mock_service()
    input_config = {"name": "test_input", "account": "my_account",
                     "base_url": "https://mock-api.test"}

    with patch("time.sleep", MagicMock()):
        events, _ = run_modular_input(service, input_config, MANIFEST)

    assert len(events) == 1
    assert events[0][0]["value"] == "recovered"


# ── event.time from _time ──


def test_event_time_from_record(activate_responses):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/items",
             json=[{"id": 1, "updated_at": "2024-06-15T12:00:00Z"}],
             status=200)

    service, _ = _build_mock_service()
    input_config = {"name": "test_input", "account": "my_account",
                     "base_url": "https://mock-api.test"}

    events, _ = run_modular_input(service, input_config, MANIFEST)

    assert len(events) == 1
    event_data, event_time = events[0]
    # _time should have been popped from record and used as event_time
    assert "_time" not in event_data
    # event_time should be epoch of 2024-06-15T12:00:00Z
    import datetime
    expected = datetime.datetime(2024, 6, 15, 12, 0, 0,
                                  tzinfo=datetime.timezone.utc).timestamp()
    assert abs(event_time - expected) < 1
