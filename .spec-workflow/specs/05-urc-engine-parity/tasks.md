# Tasks: URC Engine Parity + Comprehensive Test Suite

## Phase A: Extension Schema + Event Timestamp

- [x] 1. Create URC extensions schema and event timestamp components
  - Files: `ucc/urc_app/schema/urc_extensions_schema.yaml`, `ucc/urc_app/package/lib/urc/components/event_timestamp.py`
  - Define JSON Schema for `CursorBasedTimestamp`, `FieldBasedTimestamp`, `FetchTimestamp`
  - Implement all three as registered components with `resolve(record, config, stream_def) -> float`
  - `CursorBasedTimestamp`: reads `cursor_field` + `datetime_format` from sibling `incremental_sync`
  - `FieldBasedTimestamp`: explicit field/format with Jinja2 interpolation
  - `FetchTimestamp`: returns `time.time()`
  - Handle: string parsing, epoch passthrough (int/float), missing field fallback

- [x] 2. Integrate event timestamp into engine pipeline
  - File: `ucc/urc_app/package/lib/urc/engine.py`
  - In `_collect_stream()`: after transformations, before yield, resolve `_time`
  - Build timestamp component from `stream_def.get("event_timestamp")` or auto-derive default
  - Pass `stream_def` to resolver so `CursorBasedTimestamp` can read sibling `incremental_sync`
  - Register `event_timestamp` module in `components/__init__.py`

## Phase B: Engine Gap Components

- [x] 3. Implement CsvDecoder, JsonlDecoder, ZipfileDecoder
  - File: `ucc/urc_app/package/lib/urc/components/decoders.py`
  - `CsvDecoder`: stdlib `csv.DictReader`, config for `delimiter`, `quotechar`, `encoding`, `skip_rows`. Handle BOM, malformed rows (skip + log warning), empty responses.
  - `JsonlDecoder`: split on newlines, `json.loads()` each line. Skip empty/malformed lines with warning.
  - `ZipfileDecoder`: stdlib `zipfile` + `io.BytesIO`, configurable `inner_decoder` (default JSON). Decode each file in the archive.

- [x] 4. Implement DpathFlattenFields, KeysReplace, SchemaNormalization transformations
  - File: `ucc/urc_app/package/lib/urc/components/transformations.py`
  - `DpathFlattenFields`: flatten a specific nested field using dpath, config for `field_path` and `delete_origin`
  - `KeysReplace`: find/replace patterns in record keys, config for `replacements` list of `{old, new}` pairs, recursive
  - `SchemaNormalization`: normalize all keys to lowercase + underscores, strip special chars

- [x] 5. Implement CartesianProductStreamSlicer
  - File: `ucc/urc_app/package/lib/urc/components/partition_router.py`
  - Generate cross-product of multiple partition routers via `itertools.product`
  - Config: `stream_slicers` — list of partition router definitions, each built via `registry.create()`
  - Merge partition dicts from each router into a single dict per combination
  - Safety limit of 100,000 combinations

## Phase C: Test Infrastructure

- [x] 6. Set up pytest infrastructure with `responses` library
  - Files: `conftest.py`, `fixtures/mock_apis.py`, `fixtures/sample_manifests.py`, `fixtures/sample_data.py`
  - Install `responses` and `pytest` as test dependencies
  - `conftest.py`: fixtures for `mock_config`, `mock_checkpoint`, auto-activate `responses`
  - `mock_apis.py`: reusable scenario builders (paginated, rate-limited, auth, bulk job, CSV, errors, incremental)
  - `sample_manifests.py`: inline YAML manifests for each test scenario
  - `sample_data.py`: sample records, configs, checkpoint states
  - All mock URLs use `https://mock-api.test/` prefix

## Phase D: Unit Tests

- [x] 7. Unit tests for interpolation
  - File: `tests/unit/test_interpolation.py`
  - `eval_string()`: config vars, nested access, filters (hash, base64, regex), macros (now_utc, timestamp, day_delta, max)
  - `eval_dict()`: recursive evaluation, mixed types
  - Edge cases: undefined var fallback, non-template passthrough, literal coercion
  - Context vars: response, headers, stream_partition, record

- [x] 8. Unit tests for auth components
  - File: `tests/unit/test_auth.py`
  - All 8 types: NoAuth, ApiKeyAuth, BearerAuth, BasicAuth, OAuth2Auth, DigestAuth, JwtAuth, SessionTokenAuth, SelectiveAuth
  - OAuth2: mock token endpoint, test refresh + expiry
  - JWT: HS256 signing, token caching
  - SessionToken: mock login, token path extraction

- [x] 9. Unit tests for pagination strategies
  - File: `tests/unit/test_pagination.py`
  - NoPagination, OffsetIncrement, PageIncrement, CursorPagination, DefaultPaginator
  - Edge cases: empty pages, short last page, cursor from body vs headers, stop conditions

- [x] 10. Unit tests for decoders (including new CSV/JSONL/ZIP)
  - File: `tests/unit/test_decoders.py`
  - JsonDecoder: valid JSON, array, empty body
  - CsvDecoder: normal, BOM, custom delimiter, malformed rows, quoted fields with newlines
  - JsonlDecoder: normal, empty lines, malformed lines, trailing newline
  - ZipfileDecoder: ZIP with JSON files, ZIP with CSV (inner decoder), empty ZIP

- [x] 11. Unit tests for transformations (including new types)
  - File: `tests/unit/test_transformations.py`
  - Existing: AddFields, RemoveFields, KeysToLower, KeysToSnakeCase, FlattenFields
  - New: DpathFlattenFields, KeysReplace, SchemaNormalization
  - Test with conditions, nested records, empty records

- [x] 12. Unit tests for error handling and rate limiting
  - Files: `tests/unit/test_error_handler.py`, `tests/unit/test_rate_limiter.py`
  - Error handler: retry logic, backoff strategies, response filters, CompositeErrorHandler
  - Rate limiter: TokenBucket, MovingWindowRateLimiter, APIBudget
  - Mock `time.monotonic()` for deterministic tests — NO real sleeps

- [x] 13. Unit tests for event timestamp components
  - File: `tests/unit/test_event_timestamp.py`
  - CursorBasedTimestamp: auto-derive from incremental_sync, various formats, epoch passthrough
  - FieldBasedTimestamp: explicit field, Jinja2 interpolated, missing field fallback
  - FetchTimestamp: always returns current time (mocked)
  - Default behavior: auto-derive vs fetch time

- [x] 14. Unit tests for manifest parsing and registry
  - Files: `tests/unit/test_manifest.py`, `tests/unit/test_registry.py`
  - Manifest: $ref resolution (1/2/3 levels), type propagation, missing refs
  - Registry: @component registration, create() happy path, unknown type error

## Phase E: Integration Tests

- [x] 15. Full engine pipeline with mocked HTTP
  - File: `tests/integration/test_engine_pipeline.py`
  - Scenarios: simple GET, paginated (offset + cursor), with auth, with transformations, with rate limiter
  - Verify: record count, field values, `_time` presence, state emission

- [x] 16. Incremental sync with per-partition state
  - File: `tests/integration/test_incremental_sync.py`
  - First run (no checkpoint) → second run (resume from cursor)
  - Per-partition: 3+ partitions with different cursors, state round-trip
  - Lookback window: cursor minus lookback = actual query start
  - 10+ partitions: serialization/deserialization fidelity

- [x] 17. Async retriever bulk job lifecycle
  - File: `tests/integration/test_async_retriever.py`
  - Happy path: create → poll PENDING → RUNNING → COMPLETED → download
  - Failure: FAILED status → no records, error logged
  - Timeout: mock time, verify timeout handling
  - URL extraction: downloadUrl from poll response

- [x] 18. Modular input with mocked Splunk SDK
  - File: `tests/integration/test_modular_input.py`
  - Mock `splunklib.client.Service`: passwords + KV Store
  - First run → checkpoint save; resume → checkpoint read
  - 429 rate limit → retries succeed
  - `event.time` set from `record["_time"]`

## Phase F: Validation

- [x] 19. Run full test suite, fix any failures
  - `pytest ucc/urc_app/tests/ -v` — all green
  - `python ucc/urc_app/tests/test_manifests.py --validate` — all valid
  - `task ucc:build` — succeeds
  - `task ucc:appinspect` — 0 errors, 0 failures

- [x] 20. Push to origin and update gap analysis memory
  - Commit + push all changes
  - Update `memory/project_engine_gap.md` with completed items
