# Design: URC Engine Parity + Comprehensive Test Suite

## Overview

Two parallel workstreams: (1) close MEDIUM-priority engine gaps (~1800 lines of new components), and (2) build a pytest-based test suite with `responses` mock HTTP library that exercises every engine component through tough real-world scenarios.

## Code Reuse Analysis

### Existing Components to Leverage
- **`urc.registry`**: @component decorator + `create()` factory — all new components register the same way
- **`urc.interpolation`**: `eval_string()` / `eval_dict()` — extend context vars, don't replace
- **`urc.components.transformations`**: `_set_nested()`, `_delete_nested()`, `_check_condition()` — reuse for new transform types
- **`urc.components.decoders.JsonDecoder`**: base pattern for CsvDecoder, JsonlDecoder, ZipfileDecoder
- **`urc.engine.PerPartitionStateManager`**: already handles per-partition cursors — tests validate it
- **`urc.components.rate_limiter`**: TokenBucket, MovingWindowRateLimiter — tests validate edge cases

### Integration Points
- **`requests.Session`**: all HTTP goes through HttpRequester — `responses` library patches this
- **Splunk SDK**: `splunklib.client.Service`, `storage/passwords`, KV Store — mocked via `unittest.mock`
- **`ucc/urc_app/tests/test_manifests.py`**: existing live test harness — keep as-is, new tests are pytest

## Architecture

### Test Architecture

```
ucc/urc_app/tests/
├── conftest.py                    # Shared fixtures: mock APIs, configs, manifests
├── test_manifests.py              # (existing) live API tests — unchanged
├── unit/
│   ├── test_interpolation.py      # Jinja2 context, macros, filters, edge cases
│   ├── test_auth.py               # All 8 auth types
│   ├── test_pagination.py         # 4 strategies + edge cases
│   ├── test_decoders.py           # JSON, CSV, JSONL, Zipfile
│   ├── test_transformations.py    # All transform types including new ones
│   ├── test_extraction.py         # DpathExtractor, record selection
│   ├── test_error_handler.py      # Retry logic, backoff, response filters
│   ├── test_rate_limiter.py       # Token bucket, moving window, API budget
│   ├── test_partition_router.py   # List + Substream routers
│   ├── test_manifest.py           # YAML parse, $ref resolution, type propagation
│   └── test_registry.py           # Component registration, unknown type errors
├── integration/
│   ├── test_engine_pipeline.py    # Full manifest → records pipeline with mocked HTTP
│   ├── test_incremental_sync.py   # Cursor types, per-partition state, lookback windows
│   ├── test_async_retriever.py    # Bulk job lifecycle: create → poll → download
│   └── test_modular_input.py      # Splunk modular input with mocked SDK
└── fixtures/
    ├── mock_apis.py               # Reusable responses callbacks for common API patterns
    ├── sample_manifests.py        # Inline YAML manifests for test scenarios
    └── sample_data.py             # Sample records, configs, checkpoint states
```

### Engine Gap Components

```
ucc/urc_app/package/lib/urc/
├── components/
│   ├── decoders.py            # + CsvDecoder, JsonlDecoder, ZipfileDecoder
│   ├── transformations.py     # + DpathFlattenFields, KeysReplace, SchemaNormalization
│   ├── partition_router.py    # + CartesianProductStreamSlicer
│   └── event_timestamp.py     # NEW: CursorBasedTimestamp, FieldBasedTimestamp, FetchTimestamp
├── interpolation.py           # + response, headers, next_page_token context vars
├── engine.py                  # + event_timestamp component integration
ucc/urc_app/schema/
├── declarative_component_schema.yaml  # Airbyte upstream — untouched
└── urc_extensions_schema.yaml         # NEW: Splunk-specific component types
```

## Components and Interfaces

### Event Timestamp — URC Extension Schema

Every Splunk event needs `_time` (epoch float). The Airbyte schema has no equivalent —
it handles time only for incremental sync slicing (`DatetimeBasedCursor.cursor_field`,
`stream_interval`), not for designating event time. This is a Splunk-specific extension,
defined in a **separate schema file** to keep the upstream Airbyte schema untouched.

#### Schema separation

```
ucc/urc_app/schema/
├── declarative_component_schema.yaml   # Airbyte upstream — pinned, never modified
└── urc_extensions_schema.yaml          # Splunk-specific extensions
```

This works because:
- Airbyte's `DeclarativeStream` has `additionalProperties: true`
- The registry doesn't care about schema origin — `@component()` registers identically
- Airbyte model generation stays clean; extension models are hand-defined (trivial)
- Schema upgrades: bump Airbyte version, regenerate, zero conflicts

#### Extension components (registered in registry like all others)

**`CursorBasedTimestamp`** — zero-config default, auto-derives from `incremental_sync`:
```yaml
streams:
  - type: DeclarativeStream
    name: issues
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: updated_at
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
    event_timestamp:
      type: CursorBasedTimestamp
    # _time = updated_at, parsed with %Y-%m-%dT%H:%M:%SZ — all derived
```

**`FieldBasedTimestamp`** — explicit field when it differs from cursor:
```yaml
    event_timestamp:
      type: FieldBasedTimestamp
      field: "{{ config.get('timestamp_field', 'created_at') }}"
      format: "{{ config.get('timestamp_format', '%Y-%m-%dT%H:%M:%SZ') }}"
```

**`FetchTimestamp`** — always uses collection time (for full-refresh, no time field):
```yaml
    event_timestamp:
      type: FetchTimestamp
```

**Default behavior** (no `event_timestamp` block):
- If `incremental_sync` exists → behaves like `CursorBasedTimestamp` (auto-derive)
- If no `incremental_sync` → behaves like `FetchTimestamp` (collection time)

#### Implementation

Each timestamp type is a registered component in `urc/components/event_timestamp.py`:

```python
@component("CursorBasedTimestamp")
class CursorBasedTimestamp:
    def resolve(self, record, config, stream_def) -> float:
        # Read cursor_field and datetime_format from sibling incremental_sync
        ...

@component("FieldBasedTimestamp")
class FieldBasedTimestamp:
    def resolve(self, record, config, stream_def) -> float:
        # Use configured field/format (Jinja2 interpolated)
        ...

@component("FetchTimestamp")
class FetchTimestamp:
    def resolve(self, record, config, stream_def) -> float:
        return time.time()
```

The engine calls `timestamp_component.resolve(record, config, stream_def)` after
transformations, injects `record["_time"]` before yielding. The modular input handler
reads `record.pop("_time", time.time())` and sets `event.time`.

#### Value handling
- String → parse with resolved format → epoch float
- Numeric (int/float) → use directly as epoch (skip format parsing)
- Missing from record → `time.time()` (fetch time fallback)
- All config fields support Jinja2 interpolation via `eval_string()`

#### Extension schema file (`urc_extensions_schema.yaml`)

Defines the three timestamp component types as JSON Schema, validates manifest
authoring, and documents the extension. Kept separate so `task urc:update-schema`
only touches the Airbyte file.

### New Engine Components

#### CsvDecoder
- **Purpose:** Decode CSV response bodies into list of dicts using stdlib `csv`
- **Interface:** `decode(response) -> list[dict]`
- **Config:** `delimiter`, `quotechar`, `encoding`, `skip_rows`
- **Edge cases:** BOM handling, malformed rows (skip + log warning), empty responses

#### JsonlDecoder
- **Purpose:** Decode newline-delimited JSON (one JSON object per line)
- **Interface:** `decode(response) -> list[dict]`
- **Edge cases:** Empty lines, malformed lines (skip + warn), trailing newline

#### ZipfileDecoder
- **Purpose:** Decompress ZIP response, decode contained files
- **Interface:** `decode(response) -> list[dict]`
- **Config:** `inner_decoder` (default: JsonDecoder) — what format are the files inside the ZIP?
- **Dependencies:** stdlib `zipfile`, `io.BytesIO`

#### DpathFlattenFields
- **Purpose:** Flatten specific nested fields using dpath library
- **Interface:** `transform(record, config, **ctx) -> dict`
- **Config:** `field_path` (which field to flatten), `delete_origin` (remove original nested field)

#### KeysReplace
- **Purpose:** Find/replace patterns in record keys
- **Interface:** `transform(record, config, **ctx) -> dict`
- **Config:** `replacements` list of `{old, new}` pairs, applied recursively

#### SchemaNormalization
- **Purpose:** Normalize keys to a standard format (lowercase, underscores, strip special chars)
- **Interface:** `transform(record, config, **ctx) -> dict`

#### CartesianProductStreamSlicer
- **Purpose:** Generate cross-product of multiple partition routers
- **Interface:** `get_partitions(config) -> Iterator[dict]`
- **Config:** `stream_slicers` — list of partition router definitions
- **Example:** `regions x resource_types -> [{region: us-east, type: vm}, {region: us-east, type: disk}, ...]`

#### Interpolation Context Expansion
- **Purpose:** Add `response`, `headers`, `next_page_token`, `stream_interval` to Jinja2 context
- **Where:** `eval_string()` already accepts `**additional_context` — callers need to pass more vars
- **Impact:** pagination CursorPagination already passes headers/response; need to ensure consistency

### Test Infrastructure Components

#### conftest.py — Shared Fixtures
- `mock_config()` — standard config dict with base_url, api_key, etc.
- `mock_checkpoint()` — sample checkpoint states (empty, single-stream, per-partition)
- `responses_active()` — auto-activate `responses` mock for all tests
- `splunk_service_mock()` — mocked `splunklib.client.Service` with KV Store + passwords

#### mock_apis.py — Reusable API Scenarios
Each scenario is a function that registers `responses` callbacks:

- `register_paginated_api(base_url, pages, page_size)` — returns N pages with Link/offset pagination
- `register_rate_limited_api(base_url, fail_count)` — returns 429 N times, then 200
- `register_auth_api(base_url, auth_type)` — validates auth headers/params
- `register_bulk_job_api(base_url, poll_count)` — create→poll(N times)→download lifecycle
- `register_csv_api(base_url, rows)` — returns CSV response body
- `register_error_api(base_url, status_codes)` — returns errors in sequence
- `register_incremental_api(base_url, records_by_cursor)` — returns different records based on cursor params

#### sample_manifests.py — Inline Test Manifests
YAML strings for each test scenario, kept close to the tests (not in files) so tests are self-contained:

- `SIMPLE_GET_MANIFEST` — single stream, no auth, no pagination
- `PAGINATED_MANIFEST` — offset, page, cursor, link-header variants
- `AUTH_MANIFEST_*` — one per auth type (Bearer, Basic, OAuth2, ApiKey, JWT)
- `INCREMENTAL_MANIFEST` — DatetimeBasedCursor with lookback
- `SUBSTREAM_MANIFEST` — parent→child with per-partition state
- `BULK_JOB_MANIFEST` — AsyncRetriever lifecycle
- `RATE_LIMITED_MANIFEST` — with APIBudget policy
- `TRANSFORM_MANIFEST` — chain of AddFields, RemoveFields, KeysToSnakeCase, FlattenFields

## Error Handling

### Test Error Scenarios
1. **Malformed CSV row:** CsvDecoder skips row, logs warning, continues
2. **ZIP with no decodable files:** ZipfileDecoder returns empty list, logs warning
3. **Undefined Jinja2 variable:** eval_string returns original template string
4. **Unknown component type:** registry.create raises ValueError with helpful message
5. **Async job timeout:** AsyncRetriever logs timeout, yields no records (no partial results)
6. **Connection reset mid-pagination:** HttpRequester retries per error_handler config
7. **Checkpoint deserialization with missing partition:** PerPartitionStateManager treats as new partition

## Testing Strategy

### Unit Tests (ucc/urc_app/tests/unit/)
- One file per component module
- Mock HTTP via `responses` library — no network
- Mock `time.monotonic()` for rate limiter tests (deterministic, no sleeps)
- Each test is independent, < 100ms

### Integration Tests (ucc/urc_app/tests/integration/)
- Full engine pipeline: manifest YAML → `collect()` → records + state
- Multi-page collection with cursor advancement
- Per-partition state round-trip: collect → checkpoint → resume → collect
- Async retriever: multi-step job lifecycle
- Modular input: mocked Splunk SDK (Service, passwords, KV Store)

### Tough Scenarios (spread across unit + integration)
1. **10+ partitions with different cursor values** — state serialization fidelity
2. **Empty pages in middle of pagination** — engine continues to next page
3. **429 with Retry-After: 0** — engine retries immediately
4. **OAuth2 token expires mid-collection** — re-auth and continue
5. **Nested $ref → $ref → $ref** — manifest resolver handles 3+ levels
6. **CSV with BOM + quoted newlines** — CsvDecoder handles correctly
7. **Bulk job returns FAILED status** — AsyncRetriever logs error, yields nothing
8. **Rate limiter under burst** — TokenBucket allows burst then throttles
9. **Jinja2 template with filter chain** — `{{ value | hash('sha256') | base64encode }}`
10. **Manifest with CartesianProduct of 3 lists** — correct cross-product count
11. **Timestamp from event field** — `_time` matches parsed `updated_at`, not fetch time
12. **Timestamp field missing from record** — `_time` falls back to fetch time (epoch now)
13. **Timestamp field is already epoch int** — `_time` uses value directly, no format parsing
14. **Timestamp field/format from Jinja2 config** — config-driven field selection works
