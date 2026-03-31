# URC Engine Testing Strategy

**Last updated:** 2026-03-31

## Overview

The URC engine has a three-tier test suite: unit tests for individual components, integration tests for end-to-end pipelines, and live validation against real APIs. All automated tests run without network access and complete in under 1 second.

## Test Stack

| Tool | Purpose |
|------|---------|
| **pytest** | Test runner and fixtures |
| **responses** | Mock `requests` library HTTP calls at the transport layer |
| **unittest.mock** | Mock `time.monotonic()` (rate limiters), `time.time()` (timestamps), Splunk SDK |

## How to Run

```bash
# All automated tests (unit + integration)
python -m pytest ucc/urc_app/tests/ -v

# Unit tests only
python -m pytest ucc/urc_app/tests/unit/ -v

# Integration tests only
python -m pytest ucc/urc_app/tests/integration/ -v

# Live API validation (needs network; JSONPlaceholder needs no creds)
python ucc/urc_app/tests/test_manifests.py --validate        # parse-only, offline
python ucc/urc_app/tests/test_manifests.py --live jsonplaceholder
python ucc/urc_app/tests/test_manifests.py --live github      # needs GITHUB_TOKEN

# Build + AppInspect validation
task ucc:build
task ucc:appinspect
```

## Directory Structure

```
ucc/urc_app/tests/
├── conftest.py                          # sys.path, auto-responses fixture, mock_config
├── pytest.ini                           # testpaths, collect_ignore
├── test_manifests.py                    # Live API tests (manual CLI, not pytest)
├── fixtures/
│   ├── mock_apis.py                     # Reusable responses callbacks
│   ├── sample_manifests.py              # Inline YAML manifests
│   └── sample_data.py                   # Sample records, checkpoints
├── unit/                                # 157 tests
│   ├── test_interpolation.py            # Jinja2 eval_string, filters, macros
│   ├── test_auth.py                     # All 9 auth types
│   ├── test_pagination.py               # 5 pagination strategies
│   ├── test_decoders.py                 # JSON, CSV, JSONL decoders
│   ├── test_transformations.py          # All 8 transformation types
│   ├── test_error_handler.py            # Backoff, filters, composite handlers
│   ├── test_rate_limiter.py             # TokenBucket, MovingWindow, APIBudget
│   ├── test_event_timestamp.py          # CursorBased, FieldBased, FetchTimestamp
│   ├── test_manifest.py                 # YAML parse, $ref resolution, type propagation
│   └── test_registry.py                 # Component registration, factory
└── integration/                         # 19 tests
    ├── test_engine_pipeline.py          # Full collect() pipeline
    ├── test_incremental_sync.py         # Per-partition state, round-trip, 12 partitions
    ├── test_async_retriever.py          # Bulk job create/poll/download lifecycle
    └── test_modular_input.py            # Mocked Splunk SDK (passwords, KV Store)
```

## Key Design Decisions

### No Real Network in Automated Tests

Every HTTP call is intercepted by the `responses` library via an `autouse` fixture in `conftest.py`. This means:

- Tests run offline (CI/CD, air-gapped environments)
- Tests are deterministic (no flaky API responses)
- Tests are fast (0.19s for 176 tests)

The `responses` library patches `requests.Session` at the transport layer, so all components that use `requests` (HttpRequester, OAuth2Auth, SessionTokenAuth) are automatically mocked.

### No Real Sleeps

Rate limiter and polling tests mock `time.monotonic()` and `time.sleep()`:

```python
@patch("urc.components.rate_limiter.time")
def test_token_bucket(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    bucket = TokenBucket(max_tokens=5, refill_rate=1.0)
    # ... test logic with controlled time
```

This ensures deterministic behavior and zero-delay execution.

### Mocked Splunk SDK

Integration tests for the modular input mock `splunklib.client.Service`:

- `service.storage_passwords` — returns credential dicts (api_key, username/password)
- `service.kvstore["checkpoints"].data` — `.query()` returns checkpoint state, `.update()` saves it

This validates the full lifecycle without requiring a running Splunk instance:
1. Read credentials from Splunk passwords
2. Read checkpoint from KV Store
3. Run `engine.collect()` with mocked HTTP
4. Write events with `_time` from record
5. Save checkpoint to KV Store

### Fixtures Are Composable

`fixtures/mock_apis.py` provides reusable scenario builders:

```python
register_paginated_api(rsps, base_url, total_records=100, page_size=10)
register_rate_limited_api(rsps, base_url, fail_count=3)
register_bulk_job_api(rsps, base_url, poll_count=2)
register_csv_api(rsps, base_url, csv_text="id,name\n1,Alice")
register_error_sequence(rsps, base_url, status_codes=[500, 500, 200])
```

`fixtures/sample_manifests.py` provides inline YAML for common patterns (simple GET, paginated, auth, incremental, substream, transforms).

### test_manifests.py Is Separate

The original CLI-based test harness (`test_manifests.py`) is excluded from pytest via `conftest.py`'s `collect_ignore`. It's a manual validation tool for testing against live APIs with real credentials:

```bash
# Set env vars for API credentials
export GITHUB_TOKEN=ghp_...
export GITHUB_ORG=myorg

# Run live test
python ucc/urc_app/tests/test_manifests.py --live github
```

## What Each Test Tier Validates

### Unit Tests (157 tests)

Test individual components in isolation. Each component is instantiated directly, inputs are controlled, outputs are asserted.

| Component Area | Tests | Tough Scenarios |
|----------------|-------|-----------------|
| Interpolation | 16 | Undefined vars return original string; filter chains; literal coercion |
| Auth | 13 | OAuth2 token refresh + expiry; JWT HS256 signing; session token path extraction |
| Pagination | 13 | Empty pages; short last page; cursor from headers vs body; stop conditions |
| Decoders | 10 | CSV with BOM; malformed rows skipped; JSONL empty lines |
| Transformations | 29 | DpathFlatten with delete_origin; KeysReplace recursive; SchemaNorm special chars |
| Error Handling | 17 | WaitTimeFromHeader regex; CompositeErrorHandler delegation; predicate filters |
| Rate Limiting | 13 | TokenBucket burst then throttle; MovingWindow expiry; APIBudget URL matching |
| Event Timestamp | 8 | CursorBased auto-derive; epoch passthrough; missing field fallback |
| Manifest | 11 | 3-level nested $ref; type propagation; missing ref error |
| Registry | 8 | Unknown type lists available types; duplicate registration |

### Integration Tests (19 tests)

Test the full engine pipeline with mocked HTTP. `collect(manifest_yaml, config)` is called with real manifests and the output records/state are verified.

| Scenario | Tests | What's Validated |
|----------|-------|------------------|
| Engine Pipeline | 6 | Simple GET, pagination, auth, transforms, `_time`, empty stream |
| Incremental Sync | 5 | First run, resume from checkpoint, per-partition state (3 + 12 partitions), state round-trip |
| Async Retriever | 4 | Happy path (create/poll/download), FAILED status, timeout, URL extraction |
| Modular Input | 4 | Credential access, checkpoint save/load, rate limit recovery, `event.time` from `_time` |

### Live Validation (manual)

Tests against real APIs using the test manifests at `ucc/urc_app/tests/manifests/`:

| Manifest | API | What's Tested |
|----------|-----|---------------|
| `jsonplaceholder_paginated.yaml` | JSONPlaceholder | Pagination, IncrementingCountCursor, no auth |
| `github_repos_issues.yaml` | GitHub | BearerAuth, CursorPagination, SubstreamPartitionRouter, DatetimeBasedCursor |
| `servicenow_incidents.yaml` | ServiceNow | BasicAuth, OffsetPagination, DatetimeBasedCursor |
| `infoblox_wapi.yaml` | Infoblox | BasicAuth, CursorPagination |
| `azure_eventhub.yaml` | Azure | OAuth2, CursorPagination |

## Adding New Tests

1. **New component?** Add a unit test file in `tests/unit/test_<component>.py`
2. **New API pattern?** Add a mock scenario to `fixtures/mock_apis.py`, add a sample manifest to `fixtures/sample_manifests.py`, write an integration test
3. **New live API?** Add a manifest to `tests/manifests/`, add config to `test_manifests.py`'s `get_live_configs()`
