# Jinja2-based interpolation for manifest template expressions.
# Evaluates {{ config['api_key'] }}, {{ now_utc() }}, etc.

import datetime
from functools import lru_cache
from typing import Any, Mapping, Optional

from jinja2 import Environment, StrictUndefined, meta


# Sandboxed Jinja2 environment with custom macros/filters
_ENV = Environment(undefined=StrictUndefined)

# ── Built-in macros (available as {{ macro() }} in manifests) ──

def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)

def _today_utc():
    return datetime.date.today()

def _timestamp(dt_value):
    """Convert ISO8601 string or number to unix timestamp."""
    if isinstance(dt_value, (int, float)):
        return int(dt_value)
    if isinstance(dt_value, str):
        from dateutil import parser as dateutil_parser
        parsed = dateutil_parser.isoparse(dt_value)
        return int(parsed.timestamp())
    return int(dt_value)

def _max_fn(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return max(args[0])
    return max(args)

def _day_delta(num_days, format=None):
    dt = _now_utc() + datetime.timedelta(days=num_days)
    if format:
        return dt.strftime(format)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")


_ENV.globals.update({
    "now_utc": _now_utc,
    "today_utc": _today_utc,
    "timestamp": _timestamp,
    "max": _max_fn,
    "day_delta": _day_delta,
    "true": True,
    "false": False,
    "none": None,
})

# ── Built-in filters (available as {{ value | filter }} in manifests) ──

import base64
import hashlib
import re as _re

def _hash_filter(value, hash_type="md5", salt=""):
    h = hashlib.new(hash_type)
    h.update(f"{salt}{value}".encode())
    return h.hexdigest()

def _base64encode(value):
    return base64.b64encode(str(value).encode()).decode()

def _base64decode(value):
    return base64.b64decode(str(value)).decode()

def _regex_search(value, regex):
    match = _re.search(regex, str(value))
    if match and match.groups():
        return match.group(1)
    return ""

def _regex_replace(value, regex, replacement):
    return _re.sub(regex, replacement, str(value))


_ENV.filters.update({
    "hash": _hash_filter,
    "base64encode": _base64encode,
    "base64decode": _base64decode,
    "string": str,
    "regex_search": _regex_search,
    "regex_replace": _regex_replace,
})


@lru_cache(maxsize=256)
def _compile(template_str: str):
    return _ENV.from_string(template_str)


def eval_string(
    input_str: str,
    config: Mapping[str, Any],
    **additional_context: Any,
) -> Any:
    """Evaluate a Jinja2 template string against config and context.

    Args:
        input_str: Template string, e.g. "Bearer {{ config['api_key'] }}"
        config: User config dict.
        **additional_context: Extra variables (parameters, record, response, etc.)

    Returns:
        Rendered string, or the original value if not a template.
    """
    if not isinstance(input_str, str) or "{{" not in input_str:
        return input_str

    context = {"config": config, **additional_context}
    try:
        template = _compile(input_str)
        result = template.render(context)
        # Try to coerce back to Python literal (int, float, bool, None)
        return _literal_eval(result)
    except Exception:
        return input_str


def _literal_eval(value: str) -> Any:
    """Try to convert a rendered string to a Python literal."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "True" or stripped == "true":
        return True
    if stripped == "False" or stripped == "false":
        return False
    if stripped == "None" or stripped == "none":
        return None
    try:
        return int(stripped)
    except (ValueError, TypeError):
        pass
    try:
        return float(stripped)
    except (ValueError, TypeError):
        pass
    return value


def eval_dict(
    obj: Any,
    config: Mapping[str, Any],
    **context: Any,
) -> Any:
    """Recursively evaluate all string values in a dict/list structure."""
    if isinstance(obj, str):
        return eval_string(obj, config, **context)
    elif isinstance(obj, dict):
        return {k: eval_dict(v, config, **context) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [eval_dict(v, config, **context) for v in obj]
    return obj
