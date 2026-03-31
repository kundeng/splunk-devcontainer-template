# Requirements: URC Engine Parity + Comprehensive Test Suite

## Introduction

Close remaining gaps between the URC pure-Python declarative engine and the Airbyte CDK declarative runtime, reaching 99%+ manifest compatibility. Simultaneously build a comprehensive test suite that validates the engine end-to-end using a mock REST server, covering authentication, pagination, error handling, rate limiting, incremental sync, per-partition checkpointing, and the async bulk-job lifecycle.

## Alignment with Product Vision

URC is a Splunk add-on that executes Airbyte declarative manifests to collect data from REST APIs. The engine must handle any real-world API pattern a customer would encounter. Without parity, customers hit "unsupported component" errors. Without tests, regressions are invisible until production.

## Requirements

### R1: Engine Gap Closure (MEDIUM priority components)

**User Story:** As a manifest author, I want all MEDIUM-priority CDK components available so that I can write manifests for real-world APIs without hitting "unsupported" errors.

#### Acceptance Criteria

1. WHEN a manifest uses `DpathFlattenFields`, `KeysReplace`, or `SchemaNormalization` transformations THEN the engine SHALL apply them correctly to records.
2. WHEN a manifest uses `CsvDecoder` or `JsonlDecoder` THEN the engine SHALL extract records from CSV/JSONL response bodies using stdlib csv.
3. WHEN a manifest uses `ZipfileDecoder` THEN the engine SHALL decompress and decode records from ZIP response bodies.
4. WHEN a manifest uses response/headers/next_page_token context variables in Jinja2 templates THEN the engine SHALL provide them in the interpolation context.
5. WHEN a manifest uses `CartesianProductStreamSlicer` THEN the engine SHALL generate the cross-product of partition values.

### R2: Comprehensive Test Infrastructure

**User Story:** As a developer, I want a pytest-based test suite with a mock REST server so that I can validate every engine component without network access or API credentials.

#### Acceptance Criteria

1. WHEN `pytest` is run THEN all tests SHALL pass without network access (offline-safe).
2. WHEN a test needs HTTP responses THEN it SHALL use the `responses` library to mock `requests` calls.
3. WHEN the test suite runs THEN it SHALL cover: auth (5 types), pagination (4 strategies), error handling (retry + backoff), rate limiting, incremental sync, per-partition state, async retriever, transformations, decoders, interpolation, and the full engine pipeline.
4. WHEN a mock API scenario is defined THEN it SHALL be reusable as a pytest fixture.

### R3: Modular Input Integration Tests

**User Story:** As a Splunk admin, I want confidence that the engine works with Splunk's modular input framework — including credential access, checkpoint persistence, and event writing.

#### Acceptance Criteria

1. WHEN the modular input runs THEN it SHALL read credentials from Splunk's storage/passwords endpoint (mocked).
2. WHEN the modular input completes a collection cycle THEN it SHALL persist checkpoint state (mocked KV Store).
3. WHEN the modular input resumes after restart THEN it SHALL read the saved checkpoint and continue from where it left off.
4. WHEN the modular input encounters a rate-limited API (429) THEN it SHALL back off and retry without crashing.

### R4: Tough Scenario Coverage

**User Story:** As a developer, I want tests that exercise edge cases and failure modes so that regressions are caught before they reach production.

#### Acceptance Criteria

1. WHEN an API returns paginated results with varying page sizes and empty pages THEN the engine SHALL handle them correctly.
2. WHEN an API returns a 429 with Retry-After header THEN the engine SHALL honor the header value.
3. WHEN an async bulk job times out THEN the engine SHALL log the timeout and not yield partial results.
4. WHEN per-partition cursors have different values across 10+ partitions THEN state serialization/deserialization SHALL be lossless.
5. WHEN a Jinja2 template references an undefined variable THEN the engine SHALL return the original template string (no crash).
6. WHEN a manifest has nested $ref references THEN the engine SHALL resolve them recursively.
7. WHEN a CSV response has malformed rows THEN the CsvDecoder SHALL skip bad rows and log warnings.

## Non-Functional Requirements

### Performance
- Full test suite completes in under 30 seconds (no real network, no sleeps in unit tests)
- Mock server responses are instantaneous

### Reliability
- Tests are deterministic — no flaky timing-dependent assertions
- Rate limiter tests mock `time.monotonic()` instead of sleeping

### Modularity
- Each component has its own test file
- Fixtures are shared via conftest.py
- Mock API scenarios are declarative and composable
