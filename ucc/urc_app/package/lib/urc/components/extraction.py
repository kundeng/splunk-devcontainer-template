# Record extraction components — extract records from HTTP responses.

from typing import Any, Dict, List

import dpath

from urc.interpolation import eval_string
from urc.registry import component


@component("DpathExtractor")
class DpathExtractor:
    """Extract records from a response using a dpath field_path."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._field_path = definition.get("field_path", [])
        self._config = config

    def extract(self, response_body: Any) -> List[dict]:
        if not self._field_path:
            # No path specified — if response is a list, return it directly
            if isinstance(response_body, list):
                return response_body
            return [response_body] if isinstance(response_body, dict) else []

        # Evaluate interpolated path segments
        path_parts = [
            eval_string(p, self._config) if isinstance(p, str) else p
            for p in self._field_path
        ]

        # Build dpath-style path string
        path_str = "/".join(str(p) for p in path_parts)

        try:
            result = dpath.get(response_body, path_str, separator="/")
        except (KeyError, TypeError):
            return []

        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return []


@component("RecordSelector")
class RecordSelector:
    """Selects and optionally filters records from a response."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        extractor_def = definition.get("extractor", {})
        self._extractor = create_component(extractor_def, config)

        filter_def = definition.get("record_filter")
        self._filter = None
        if filter_def:
            self._filter = RecordFilter(filter_def, config)

    def select(self, response_body: Any, config: dict, **context) -> List[dict]:
        records = self._extractor.extract(response_body)
        if self._filter:
            records = self._filter.filter(records, config, **context)
        return records


@component("RecordFilter")
class RecordFilter:
    """Filter records based on a Jinja2 condition."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._condition = definition.get("condition", "")
        self._config = config

    def filter(self, records: List[dict], config: dict, **context) -> List[dict]:
        if not self._condition:
            return records
        return [
            record for record in records
            if eval_string(self._condition, config, record=record, **context)
            is True
        ]
