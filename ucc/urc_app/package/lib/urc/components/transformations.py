# Record transformation components — modify records after extraction.

import logging
import re
from typing import Any, Dict, List

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


def _set_nested(record: dict, path: List[str], value: Any) -> dict:
    """Set a value at a nested path, creating intermediate dicts as needed."""
    current = record
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value
    return record


def _delete_nested(record: dict, path: List[str]) -> dict:
    """Delete a value at a nested path. No-op if path doesn't exist."""
    current = record
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return record
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)
    return record


def _check_condition(condition: str, record: dict, config: dict, **context) -> bool:
    """Evaluate an optional Jinja2 condition. Returns True if no condition set."""
    if not condition:
        return True
    result = eval_string(condition, config, record=record, **context)
    return result is True


@component("AddFields")
class AddFields:
    """Add new fields to records using Jinja2-evaluated values."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._fields = definition.get("fields", [])
        self._condition = definition.get("condition", "")
        self._config = config

    def transform(self, record: dict, config: dict, **context) -> dict:
        if not _check_condition(self._condition, record, config, **context):
            return record

        for field_def in self._fields:
            path = field_def.get("path", [])
            value_template = field_def.get("value", "")
            if not path:
                continue
            value = eval_string(
                value_template, config,
                record=record, **context,
            )
            _set_nested(record, path, value)

        return record


@component("RemoveFields")
class RemoveFields:
    """Remove fields from records by path."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._field_pointers = definition.get("field_pointers", [])
        self._condition = definition.get("condition", "")
        self._config = config

    def transform(self, record: dict, config: dict, **context) -> dict:
        if not _check_condition(self._condition, record, config, **context):
            return record

        for path in self._field_pointers:
            if path:
                _delete_nested(record, path)

        return record


def _keys_to_lower(obj: Any) -> Any:
    """Recursively transform all dict keys to lowercase."""
    if isinstance(obj, dict):
        return {k.lower(): _keys_to_lower(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_keys_to_lower(item) for item in obj]
    return obj


@component("KeysToLower")
class KeysToLower:
    """Transform all keys in a record to lowercase (recursive)."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def transform(self, record: dict, config: dict, **context) -> dict:
        return _keys_to_lower(record)


_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z0-9])([A-Z])")


def _to_snake_case(name: str) -> str:
    """Convert a camelCase or PascalCase string to snake_case."""
    s = _CAMEL_RE1.sub(r"\1_\2", name)
    s = _CAMEL_RE2.sub(r"\1_\2", s)
    return s.lower()


def _keys_to_snake(obj: Any) -> Any:
    """Recursively transform all dict keys to snake_case."""
    if isinstance(obj, dict):
        return {_to_snake_case(k): _keys_to_snake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_keys_to_snake(item) for item in obj]
    return obj


@component("KeysToSnakeCase")
class KeysToSnakeCase:
    """Transform all keys from camelCase to snake_case (recursive)."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def transform(self, record: dict, config: dict, **context) -> dict:
        return _keys_to_snake(record)


def _flatten(obj: Any, parent_key: str, sep: str, flatten_lists: bool) -> dict:
    """Recursively flatten a nested dict."""
    items: List[tuple] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(_flatten(v, new_key, sep, flatten_lists).items())
        return dict(items)
    if isinstance(obj, list) and flatten_lists:
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(_flatten(v, new_key, sep, flatten_lists).items())
        return dict(items)
    return {parent_key: obj}


@component("FlattenFields")
class FlattenFields:
    """Flatten nested objects to single-level with dot-separated keys."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._flatten_lists = definition.get("flatten_lists", True)
        self._separator = definition.get("separator", ".")

    def transform(self, record: dict, config: dict, **context) -> dict:
        return _flatten(record, "", self._separator, self._flatten_lists)
