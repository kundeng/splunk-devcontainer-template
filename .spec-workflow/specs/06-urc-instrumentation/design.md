# Design: Aspect-Oriented Instrumentation for URC Engine

## Overview

A centralized instrumentation layer that wraps component methods at the registry level using `inspect.getmembers()` + `functools.wraps()`. Four modules, ~220 lines total. No `__getattr__` proxies, no metaclasses, no external dependencies.

## Code Reuse Analysis

### Existing Components to Leverage
- **`urc/registry.py`** — single creation point (`create()`) where wrapping happens
- **`urc/engine.py`** — `_collect_stream()` is the main orchestration loop, currently has inline logging
- **`urc/components/error_handler.py`** — existing retry logic stays, instrumentation wraps around it

### Integration Points
- **`registry.create()`** — one line change: `return instrument(instance, type_name)`
- **`engine.py`** — replace inline `logger.xxx()` with emitter calls at stream boundaries only
- **All 9 component files** — remove `import logging` and `logger = ...` lines, remove all `logger.xxx()` calls

## Architecture

```
registry.create()
    │
    ▼
instrument(instance, type_name)          ← instrumentation.py
    │
    ├── inspect.getmembers(instance)     ← find public methods
    ├── for each method:
    │   ├── inspect.signature(method)    ← capture param names
    │   ├── functools.wraps(method)      ← preserve identity
    │   └── setattr(instance, name, wrapped)
    │
    └── return instance                  ← same object, not a proxy
           │
           ▼
    wrapped method called
           │
           ├── BEFORE hooks             ← lifecycle.py
           │   ├── check_disabled()
           │   ├── check_run_once()
           │   └── check_config_change()
           │
           ├── actual method()
           │
           ├── AFTER hooks
           │   └── emit structured log  ← structured_logger.py
           │
           └── ERROR hooks              ← recovery.py
               ├── lookup policy
               ├── emit error log
               └── skip or re-raise
```

## Components and Interfaces

### Module 1: `instrumentation.py` (~80 lines)

The core AOP mechanism. Uses `inspect` and `functools` from stdlib.

```python
import time
from contextlib import contextmanager
from functools import wraps
from inspect import getmembers, ismethod, signature
from typing import Any, Callable, Dict, List, Optional

# Global hook registries
_before_hooks: List[Callable] = []
_after_hooks: List[Callable] = []
_error_hooks: List[Callable] = []

def register_hook(phase: str, hook: Callable):
    """Register a before/after/error hook."""
    {"before": _before_hooks, "after": _after_hooks, "error": _error_hooks}[phase].append(hook)

def instrument(instance: Any, component_type: str) -> Any:
    """Wrap all public methods of a component instance with hooks.

    Uses inspect.getmembers() to find methods, functools.wraps() to
    preserve identity, and setattr() to replace — no proxy object.
    isinstance() checks still work on the returned object.
    """
    for name, method in getmembers(instance, predicate=ismethod):
        if name.startswith('_'):
            continue
        wrapped = _make_wrapper(method, name, component_type)
        setattr(instance, name, wrapped)
    return instance

def _make_wrapper(method: Callable, method_name: str, component_type: str) -> Callable:
    """Create a wrapped version of a method with before/after/error hooks."""
    sig = signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        # Bind args to parameter names for structured context
        context = _bind_context(sig, args, kwargs)

        # BEFORE hooks
        for hook in _before_hooks:
            hook(component_type, method_name, context)

        start = time.monotonic()
        try:
            result = method(*args, **kwargs)
        except Exception as exc:
            elapsed = time.monotonic() - start
            # ERROR hooks
            for hook in _error_hooks:
                hook(component_type, method_name, exc, elapsed, context)
            raise
        else:
            elapsed = time.monotonic() - start
            # AFTER hooks
            for hook in _after_hooks:
                hook(component_type, method_name, result, elapsed, context)
            return result

    wrapper.__signature__ = sig
    return wrapper

def _bind_context(sig, args, kwargs) -> dict:
    """Extract loggable context from method arguments."""
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        # Only include known useful params, not large dicts
        safe_keys = {'url', 'stream_slice', 'stream_name', 'config', 'record'}
        return {k: v for k, v in bound.arguments.items() if k in safe_keys}
    except TypeError:
        return {}
```

**Key design decisions:**
- `setattr` on the real instance, not a proxy — `isinstance()` works
- `inspect.signature()` + `bind()` gives parameter-aware context without parsing
- Hooks are plain functions in a list — no registration framework needed
- `_bind_context` filters to safe_keys to avoid logging entire config dicts

### Module 2: `structured_logger.py` (~50 lines)

Splunk-native `key=value` emitter. One `logging.Logger` call per event, formatted for instant Splunk searchability.

```python
import logging
from typing import Any

logger = logging.getLogger("urc.instrumentation")

def _quote(value: Any) -> str:
    """Quote values containing spaces or special chars."""
    s = str(value)
    if ' ' in s or '=' in s or '"' in s:
        return f'"{s}"'
    return s

def emit(**fields):
    """Emit a structured key=value log event.

    Example output:
        action=http_request component=HttpRequester method=send
        status=200 elapsed=0.285 url=https://api.github.com/repos
    """
    parts = [f"{k}={_quote(v)}" for k, v in fields.items() if v is not None]
    logger.info(" ".join(parts))
```

**Hook implementations using the emitter:**

```python
def after_hook(component_type, method_name, result, elapsed, context):
    """Standard after-hook: emit structured call log."""
    emit(
        action="call",
        component=component_type,
        method=method_name,
        elapsed=f"{elapsed:.3f}",
    )

def error_hook(component_type, method_name, exc, elapsed, context):
    """Standard error-hook: emit structured error log."""
    emit(
        action="error",
        component=component_type,
        method=method_name,
        elapsed=f"{elapsed:.3f}",
        error_type=type(exc).__name__,
        error=str(exc)[:200],  # truncate long error messages
    )
```

### Module 3: `recovery.py` (~40 lines)

Centralized error recovery policy. A single dict defines skip-vs-raise for every `(component, method)` pair.

```python
class SkipRecord(Exception):
    """Raised by recovery policy to skip the current record and continue."""
    pass

class SkipComponent(Exception):
    """Raised by lifecycle hooks to skip the current component entirely."""
    pass

# Policy table: (component_type, method_name) → action
# Actions: "raise" (default), "skip" (swallow + warn)
RECOVERY_POLICY = {
    ("CsvDecoder", "decode"):           "skip",
    ("JsonlDecoder", "decode"):         "skip",
    ("DpathExtractor", "extract"):      "skip",
    ("AddFields", "transform"):         "skip",
    ("RemoveFields", "transform"):      "skip",
    ("KeysReplace", "transform"):       "skip",
    ("SchemaNormalization", "transform"): "skip",
    # Retryable components — always re-raise, retry loop handles it
    ("HttpRequester", "send"):          "raise",
    ("AsyncRetriever", "read_records"): "raise",
    ("OAuth2Auth", "apply"):            "raise",
}

def recovery_error_hook(component_type, method_name, exc, elapsed, context):
    """Error hook that applies the recovery policy."""
    key = (component_type, method_name)
    action = RECOVERY_POLICY.get(key, "raise")

    if action == "skip":
        emit(
            action="skip",
            component=component_type,
            method=method_name,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise SkipRecord(str(exc)) from exc
    # else: re-raise (default) — let caller handle
```

The engine's record loop catches `SkipRecord`:

```python
# engine.py — in the record iteration loop
for record in retriever.read_records(config, stream_slice=merged_slice):
    try:
        for t in transformations:
            record = t.transform(record, config, ...)
        record["_time"] = ts_resolver.resolve(record, config, stream_def)
    except SkipRecord:
        continue  # skip this record, continue with next
```

### Module 4: `lifecycle.py` (~50 lines)

Before-hooks for modular input lifecycle events.

```python
def check_disabled(component_type, method_name, context):
    """Skip collection if input was disabled mid-run."""
    config = context.get('config', {})
    if config.get('_disabled'):
        raise SkipComponent("input disabled")

def check_run_once(component_type, method_name, context):
    """Skip if input is in run-once mode and already completed."""
    config = context.get('config', {})
    if config.get('_run_once') and config.get('_last_run_completed'):
        raise SkipComponent("run-once already completed")

def check_config_change(component_type, method_name, context):
    """Flag for component rebuild if config hash changed."""
    config = context.get('config', {})
    if config.get('_config_changed'):
        config['_needs_rebuild'] = True
```

### Integration: `registry.py` change (1 line)

```python
def create(component_def: dict, config: dict, **kwargs) -> Any:
    # ... existing validation ...
    instance = cls(component_def, config, **kwargs)
    return instrument(instance, type_name)  # ← one line added
```

### Integration: `engine.py` changes

Replace 8 inline `logger.xxx()` calls with 2 emitter calls at stream boundaries:

```python
def _collect_stream(stream_def, config, stream_name, checkpoint=None):
    emit(action="stream_start", stream=stream_name)
    # ... existing logic, no more logger calls ...
    emit(action="stream_complete", stream=stream_name,
         records=record_count, elapsed=f"{elapsed:.1f}")
```

### Initialization: `__init__.py` or engine startup

```python
from urc.instrumentation import register_hook
from urc.structured_logger import after_hook, error_hook
from urc.recovery import recovery_error_hook
from urc.lifecycle import check_disabled, check_run_once

# Register hooks — order matters for error hooks
register_hook("after", after_hook)
register_hook("error", error_hook)
register_hook("error", recovery_error_hook)
register_hook("before", check_disabled)
register_hook("before", check_run_once)
```

## What Changes in Component Files

### Before (current — auth.py example)

```python
import logging
logger = logging.getLogger(__name__)

class OAuth2Auth(BaseAuth):
    def apply(self, session, config):
        if self._access_token is None or time.time() >= self._token_expiry:
            self._refresh_token(config)
        if self._access_token:
            session.headers["Authorization"] = f"Bearer {self._access_token}"

    def _refresh_token(self, config):
        # ... token fetch logic ...
        try:
            resp = requests.post(token_url, data=body, ...)
            resp.raise_for_status()
            # ...
        except Exception as e:
            logger.error(f"OAuth2 token refresh failed: {e}")
            raise
```

### After (instrumented — auth.py cleaned)

```python
class OAuth2Auth(BaseAuth):
    def apply(self, session, config):
        if self._access_token is None or time.time() >= self._token_expiry:
            self._refresh_token(config)
        if self._access_token:
            session.headers["Authorization"] = f"Bearer {self._access_token}"

    def _refresh_token(self, config):
        resp = requests.post(token_url, data=body, ...)
        resp.raise_for_status()
        # ...
```

No `import logging`. No `logger = ...`. No `logger.error(...)`. The `apply()` method is automatically instrumented — if `_refresh_token` raises, the error hook emits a structured log event with component type, method name, elapsed time, and error details.

## Testing Strategy

### Unit Tests
- Test `instrument()` wraps methods correctly (before/after/error hooks fire)
- Test `_bind_context()` extracts safe params
- Test `emit()` produces correct `key=value` format
- Test recovery policy: skip vs raise for each component
- Test lifecycle hooks: disabled, run-once, config-change

### Integration Tests
- Test full pipeline with instrumentation enabled: verify structured log output
- Test `SkipRecord` in engine loop: bad record skipped, collection continues
- Test with instrumentation disabled: components still work (unit test mode)

### Key: Unit tests bypass instrumentation

Existing unit tests create components directly (`BearerAuth({...}, config)`), not via `registry.create()`. So they're not instrumented — which is correct. Unit tests test business logic, not the wrapping layer.

## Error Handling

### SkipRecord flow
1. Transform raises `ValueError` on bad Jinja2
2. Wrapper catches it, runs error hooks
3. Recovery hook sees `("AddFields", "transform")` → policy is "skip"
4. Recovery hook raises `SkipRecord`
5. Wrapper re-raises `SkipRecord` (it's not caught by the generic except)
6. Engine's record loop catches `SkipRecord`, continues to next record

Wait — the current design has the recovery hook inside the error hook chain, but `SkipRecord` would then be caught by the wrapper's `except Exception`. Let me fix this:

```python
def wrapper(*args, **kwargs):
    for hook in _before_hooks:
        hook(component_type, method_name, context)

    start = time.monotonic()
    try:
        result = method(*args, **kwargs)
    except SkipRecord:
        raise  # pass through — engine handles this
    except SkipComponent:
        raise  # pass through — engine handles this
    except Exception as exc:
        elapsed = time.monotonic() - start
        for hook in _error_hooks:
            hook(component_type, method_name, exc, elapsed, context)
        raise
    else:
        elapsed = time.monotonic() - start
        for hook in _after_hooks:
            hook(component_type, method_name, result, elapsed, context)
        return result
```

`SkipRecord` and `SkipComponent` are explicitly passed through before the generic `except Exception`.

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `urc/instrumentation.py` | ~80 | `instrument()`, `register_hook()`, `_make_wrapper()` |
| `urc/structured_logger.py` | ~50 | `emit()`, `after_hook()`, `error_hook()` |
| `urc/recovery.py` | ~40 | `RECOVERY_POLICY`, `SkipRecord`, `recovery_error_hook()` |
| `urc/lifecycle.py` | ~50 | `check_disabled()`, `check_run_once()`, `check_config_change()` |
| `urc/registry.py` | +1 | Add `instrument()` call in `create()` |
| `urc/engine.py` | -8, +5 | Replace logger calls with 2 emit calls + SkipRecord handler |
| 9 component files | -37 | Remove all `logger.xxx()` and `import logging` |
| **Total** | ~220 new, ~45 removed | Net ~175 lines added |
