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


def emit(level="info", **fields) -> None:
    """Emit a structured key=value log event.

    Args:
        level: Log level — "info" (default) or "debug".
        **fields: Arbitrary key=value pairs. None values are filtered out.

    Example output:
        action=call component=HttpRequester method=send elapsed=0.285
    """
    parts = [f"{k}={_quote(v)}" for k, v in fields.items() if v is not None]
    if parts:
        msg = " ".join(parts)
        if level == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)


# ── Standard hooks (register via instrumentation.register_hook) ──


def before_hook(component_type: str, method_name: str, context: dict) -> None:
    """Standard before-hook: emit method entry at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        emit(level="debug", action="enter",
             component=component_type, method=method_name)


def after_hook(component_type: str, method_name: str, result: Any,
               elapsed: float, context: dict) -> None:
    """Standard after-hook: emit structured call log."""
    emit(
        action="call",
        component=component_type,
        method=method_name,
        elapsed=f"{elapsed:.3f}",
    )
    # Emit verbose context at DEBUG level
    if context and logger.isEnabledFor(logging.DEBUG):
        debug_fields = {k: str(v)[:200] for k, v in context.items()
                        if k in ('url', 'stream_slice', 'stream_name')}
        if debug_fields:
            emit(level="debug", action="call_detail",
                 component=component_type, method=method_name, **debug_fields)


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
