# Tasks: Aspect-Oriented Instrumentation for URC Engine

## Phase A: Core Instrumentation Layer

- [x] 1. Create `instrumentation.py` — method wrapping via inspect + functools
  - File: `ucc/urc_app/package/lib/urc/instrumentation.py`
  - `instrument(instance, component_type)` using `inspect.getmembers()` + `setattr()`
  - `_make_wrapper()` with `functools.wraps()` + `inspect.signature().bind()`
  - `register_hook(phase, hook)` for before/after/error hook lists
  - `_bind_context()` extracts safe params (url, stream_slice, stream_name)
  - SkipRecord and SkipComponent pass through before generic except

- [x] 2. Create `structured_logger.py` — Splunk key=value emitter
  - File: `ucc/urc_app/package/lib/urc/structured_logger.py`
  - `emit(**fields)` formats as `key=value` pairs, quotes values with spaces
  - `after_hook()` — emits action=call with component, method, elapsed
  - `error_hook()` — emits action=error with error_type, truncated message

- [x] 3. Create `recovery.py` — centralized error recovery policy
  - File: `ucc/urc_app/package/lib/urc/recovery.py`
  - `SkipRecord` and `SkipComponent` exception classes
  - `RECOVERY_POLICY` dict mapping (component_type, method) to "skip" or "raise"
  - `recovery_error_hook()` — looks up policy, raises SkipRecord or re-raises

- [x] 4. Create `lifecycle.py` — before-hooks for input lifecycle
  - File: `ucc/urc_app/package/lib/urc/lifecycle.py`
  - `check_disabled()` — raises SkipComponent if `_disabled` in config
  - `check_run_once()` — raises SkipComponent if already completed
  - `check_config_change()` — flags `_needs_rebuild` if config hash changed

## Phase B: Integration

- [-] 5. Wire instrumentation into registry + engine
  - Modify `registry.py`: add `instrument()` call in `create()`
  - Modify `engine.py`: replace inline logger calls with emitter, add SkipRecord handler in record loop
  - Add hook registration in `__init__.py` or engine startup
  - Verify: `python ucc/urc_app/tests/test_manifests.py --validate` still passes

- [ ] 6. Strip logging from all component files
  - Remove `import logging` and `logger = logging.getLogger(...)` from 9 files
  - Remove all 37 `logger.xxx()` calls from component code
  - Files: auth.py, decoders.py, event_timestamp.py, partition_router.py, rate_limiter.py, requester.py, retriever.py, async_retriever.py, engine.py
  - Keep only 2 `emit()` calls in engine.py (stream_start, stream_complete)

## Phase C: Tests

- [ ] 7. Unit tests for instrumentation layer
  - File: `ucc/urc_app/tests/unit/test_instrumentation.py`
  - Test `instrument()` wraps public methods, skips private
  - Test hooks fire in order (before → method → after, or before → method → error)
  - Test `_bind_context()` extracts correct params
  - Test SkipRecord passes through wrapper
  - Test SkipComponent passes through wrapper
  - Test `isinstance()` still works on instrumented instance

- [ ] 8. Unit tests for structured logger and recovery
  - Files: `ucc/urc_app/tests/unit/test_structured_logger.py`, `ucc/urc_app/tests/unit/test_recovery.py`
  - Structured logger: test emit() format, quoting, None filtering
  - Recovery: test policy lookup, skip vs raise, unknown defaults to raise
  - Lifecycle: test check_disabled, check_run_once, check_config_change

- [ ] 9. Integration test: instrumented pipeline
  - File: `ucc/urc_app/tests/integration/test_instrumented_pipeline.py`
  - Test full collect() with instrumentation enabled, capture log output
  - Test SkipRecord: bad transform skips record, collection continues
  - Verify structured log events contain correct action/component/method/elapsed
  - Verify existing integration tests still pass (no behavioral change)

## Phase D: Validation

- [ ] 10. Run full test suite + build + appinspect
  - `pytest ucc/urc_app/tests/ -v` — all green
  - `python ucc/urc_app/tests/test_manifests.py --validate` — all valid
  - `python ucc/urc_app/tests/test_manifests.py --live jsonplaceholder` — e2e passes
  - `task ucc:build` — succeeds
  - `task ucc:appinspect` — 0 errors, 0 failures
  - Commit + push

- [ ] 11. Verify no component file imports logging
  - `grep -rn "import logging" ucc/urc_app/package/lib/urc/components/` — must return empty
  - `grep -rn "logger\." ucc/urc_app/package/lib/urc/components/` — must return empty
  - Only `structured_logger.py` should import `logging`
