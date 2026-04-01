# Aspect-oriented instrumentation — wraps component methods at creation time.
#
# Uses inspect.getmembers() + functools.wraps() + setattr() to wrap public
# methods with before/after/error hooks. No __getattr__ proxies, no metaclasses.
# isinstance() checks still work on instrumented instances.
#
# Integration point: registry.create() calls instrument() after instantiation.

import time
from contextlib import contextmanager
from functools import wraps
from inspect import getmembers, ismethod, signature
from typing import Any, Callable, Dict, List, Optional


# ── Hook registries ──

_before_hooks: List[Callable] = []
_after_hooks: List[Callable] = []
_error_hooks: List[Callable] = []

# Whether instrumentation is active (disabled for unit tests)
_enabled = True


def register_hook(phase: str, hook: Callable) -> None:
    """Register a before/after/error hook.

    Args:
        phase: One of "before", "after", "error".
        hook: Callable with signature depending on phase:
            before(component_type, method_name, context)
            after(component_type, method_name, result, elapsed, context)
            error(component_type, method_name, exc, elapsed, context)
    """
    registry = {"before": _before_hooks, "after": _after_hooks, "error": _error_hooks}
    if phase not in registry:
        raise ValueError(f"Unknown hook phase '{phase}', must be: before, after, error")
    registry[phase].append(hook)


def clear_hooks() -> None:
    """Remove all registered hooks. Useful for testing."""
    _before_hooks.clear()
    _after_hooks.clear()
    _error_hooks.clear()


def set_enabled(enabled: bool) -> None:
    """Enable or disable instrumentation globally."""
    global _enabled
    _enabled = enabled


# ── Safe parameter keys for context extraction ──

_CONTEXT_KEYS = frozenset({
    'url', 'stream_slice', 'stream_name', 'stream_partition',
    'config', 'record', 'response', 'attempt',
})


# ── Core instrumentation ──


def instrument(instance: Any, component_type: str) -> Any:
    """Wrap all public methods of a component instance with hooks.

    Uses inspect.getmembers() to discover methods, functools.wraps() to
    preserve identity, and setattr() to replace on the real instance.

    Args:
        instance: The component instance to instrument.
        component_type: The manifest type name (e.g. "HttpRequester").

    Returns:
        The same instance with public methods wrapped. Not a proxy —
        isinstance() checks still work.
    """
    if not _enabled:
        return instance

    for name, method in getmembers(instance, predicate=ismethod):
        if name.startswith('_'):
            continue
        wrapped = _make_wrapper(method, name, component_type)
        setattr(instance, name, wrapped)
    return instance


def _make_wrapper(method: Callable, method_name: str, component_type: str) -> Callable:
    """Create a wrapped version of a method with before/after/error hooks."""
    # Avoid circular import — SkipRecord/SkipComponent may not be defined yet
    # at import time. Import lazily in the wrapper.
    sig = signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        context = _bind_context(sig, args, kwargs)

        # BEFORE hooks
        for hook in _before_hooks:
            hook(component_type, method_name, context)

        start = time.monotonic()
        try:
            result = method(*args, **kwargs)
        except Exception as exc:
            # Let SkipRecord/SkipComponent pass through without running error hooks
            from urc.recovery import SkipRecord, SkipComponent
            if isinstance(exc, (SkipRecord, SkipComponent)):
                raise

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
    """Extract loggable context from method arguments.

    Uses inspect.signature().bind() to map positional/keyword args to
    parameter names, then filters to safe keys to avoid logging large dicts.
    """
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k in _CONTEXT_KEYS}
    except TypeError:
        return {}
