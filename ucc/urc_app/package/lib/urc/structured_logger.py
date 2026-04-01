# Structured Splunk-native logging — key=value emitter.
#
# Emits log events formatted as key=value pairs, instantly searchable
# in Splunk without regex extraction:
#   action=http_request component=HttpRequester method=send status=200 elapsed=0.285

import logging
from typing import Any

logger = logging.getLogger("urc.engine")


def _quote(value: Any) -> str:
    """Quote values containing spaces, equals, or quotes."""
    s = str(value)
    if ' ' in s or '=' in s or '"' in s:
        return '"{}"'.format(s.replace('"', '\\"'))
    return s


def emit(**fields) -> None:
    """Emit a structured key=value log event.

    Args:
        **fields: Arbitrary key=value pairs. None values are filtered out.

    Example output:
        action=call component=HttpRequester method=send elapsed=0.285
    """
    parts = [f"{k}={_quote(v)}" for k, v in fields.items() if v is not None]
    if parts:
        logger.info(" ".join(parts))


# ── Standard hooks (register via instrumentation.register_hook) ──


def after_hook(component_type: str, method_name: str, result: Any,
               elapsed: float, context: dict) -> None:
    """Standard after-hook: emit structured call log."""
    emit(
        action="call",
        component=component_type,
        method=method_name,
        elapsed=f"{elapsed:.3f}",
    )


def error_hook(component_type: str, method_name: str, exc: Exception,
               elapsed: float, context: dict) -> None:
    """Standard error-hook: emit structured error log."""
    emit(
        action="error",
        component=component_type,
        method=method_name,
        elapsed=f"{elapsed:.3f}",
        error_type=type(exc).__name__,
        error=str(exc)[:200],
    )
