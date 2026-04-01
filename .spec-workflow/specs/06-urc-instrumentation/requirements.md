# Requirements: Aspect-Oriented Instrumentation for URC Engine

## Introduction

Replace 80+ scattered `logger.xxx()` calls across 12 component files with a centralized, aspect-oriented instrumentation layer. Uses Python 3.9+ standard library (`inspect`, `functools`, `contextlib`, `typing`) to wrap component methods at the registry level — zero changes to component business logic. Extensible to lifecycle management (disable, run-once, reconfigure).

## Alignment with Product Vision

URC runs inside Splunk where logs *are* the observability data. Free-text `logger.info()` messages are unsearchable. Structured `key=value` events are instantly queryable (`| where action="error" | stats count by component`). Centralizing error recovery policy means one place to audit, not 12 files.

## Requirements

### R1: Method Instrumentation via `inspect` + `functools`

**User Story:** As a developer, I want component methods automatically instrumented at creation time so that I never write logging or timing code in business logic.

#### Acceptance Criteria

1. WHEN `registry.create()` builds a component THEN all public methods (not starting with `_`) SHALL be wrapped with before/after/error hooks.
2. WHEN a wrapped method is called THEN `inspect.signature()` SHALL be used to bind args to parameter names for structured context.
3. WHEN a wrapped method completes THEN `functools.wraps()` SHALL preserve the original method's name, docstring, and signature.
4. WHEN `isinstance()` is called on an instrumented component THEN it SHALL return True for the original class (no proxy objects).

### R2: Structured Splunk-Native Logging

**User Story:** As a Splunk admin, I want engine logs emitted as `key=value` pairs so that I can search, alert, and dashboard without regex extraction.

#### Acceptance Criteria

1. WHEN a component method succeeds THEN the emitter SHALL log: `action=call component=<type> method=<name> elapsed=<seconds>`.
2. WHEN a component method fails THEN the emitter SHALL log: `action=error component=<type> method=<name> elapsed=<seconds> error_type=<class> error=<message>`.
3. WHEN an HTTP request is made THEN the emitter SHALL log: `action=http_request method=<GET/POST> url=<url> status=<code> elapsed=<seconds>`.
4. WHEN a stream collection completes THEN the emitter SHALL log: `action=stream_complete stream=<name> records=<count> elapsed=<seconds> partitions=<count>`.
5. WHEN values contain spaces or special characters THEN they SHALL be quoted (`key="value with spaces"`).

### R3: Centralized Error Recovery Policy

**User Story:** As a developer, I want one file that defines error recovery behavior for every component so that I can audit and change policies without hunting through 12 files.

#### Acceptance Criteria

1. WHEN a `CsvDecoder.decode()` encounters a malformed row THEN the policy SHALL skip the row and emit a warning (not crash).
2. WHEN an `HttpRequester.send()` encounters a connection error THEN the policy SHALL re-raise (let the retry loop handle it).
3. WHEN an `AddFields.transform()` encounters a bad Jinja2 template THEN the policy SHALL skip the record and emit a warning.
4. WHEN an unknown `(component, method)` pair errors THEN the default policy SHALL be to re-raise.
5. WHEN the recovery policy file is read THEN all policies SHALL be visible as a single dict/table.

### R4: Lifecycle Hooks (Extensible)

**User Story:** As a Splunk admin, I want modular input lifecycle events (disable, run-once, reconfigure) handled cleanly without scattered if-checks in every component.

#### Acceptance Criteria

1. WHEN an input is disabled mid-collection THEN a "before" hook SHALL detect the flag and stop collection cleanly (no partial state corruption).
2. WHEN an input is configured for "run once" THEN a "before" hook SHALL skip collection if the previous run completed successfully.
3. WHEN an input's config changes between collection cycles THEN a "before" hook SHALL trigger component re-creation via `registry.create()`.
4. WHEN a new lifecycle hook is added THEN it SHALL be a function registered in a hooks list — no changes to component code or the wrapping mechanism.

### R5: Remove Scattered Logging

**User Story:** As a developer, I want component files to contain only business logic so that I can read and modify them without wading through logging boilerplate.

#### Acceptance Criteria

1. WHEN all instrumentation is applied THEN all `logger.info/warning/error/debug` calls in component files SHALL be removed.
2. WHEN a component file is read THEN it SHALL contain zero import of `logging` module.
3. WHEN the instrumentation layer is disabled (e.g., for unit testing) THEN components SHALL still function identically — just without log output.

## Non-Functional Requirements

### Performance
- Method wrapping overhead under 10 microseconds per call (setattr, not proxy)
- No allocation per call beyond the structured log string

### Modularity
- `instrumentation.py`: method wrapping via `inspect` + `functools` (~80 lines)
- `structured_logger.py`: Splunk `key=value` emitter (~50 lines)
- `recovery.py`: error recovery policy table + `SkipRecord` exception (~40 lines)
- `lifecycle.py`: before/after hook registration for lifecycle events (~50 lines)

### Compatibility
- Python 3.9+ (no 3.10+ features)
- Zero external dependencies (stdlib only: `inspect`, `functools`, `contextlib`, `time`, `typing`)
- Components remain testable without instrumentation (unit tests bypass `registry.create()`)
