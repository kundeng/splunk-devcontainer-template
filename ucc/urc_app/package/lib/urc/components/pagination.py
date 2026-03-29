# Pagination components — determine the next page of results.

import logging
from typing import Any, Dict, Optional

import requests

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


@component("NoPagination")
class NoPagination:
    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def next_page(self, response: requests.Response, current_params: dict,
                  last_records: list, page_count: int) -> Optional[dict]:
        return None


@component("DefaultPaginator")
class DefaultPaginator:
    """Delegates to a pagination_strategy and injects tokens into requests."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        strategy_def = definition.get("pagination_strategy", {})
        self._strategy = create_component(strategy_def, config)
        self._page_size_option = definition.get("page_size_option")
        self._page_token_option = definition.get("page_token_option")
        self._config = config

    def next_page(self, response: requests.Response, current_params: dict,
                  last_records: list, page_count: int) -> Optional[dict]:
        token = self._strategy.next_token(response, last_records, page_count)
        if token is None:
            return None

        params = dict(current_params)

        # Inject page token
        if self._page_token_option:
            field = eval_string(
                self._page_token_option.get("field_name", ""),
                self._config,
            )
            inject_into = self._page_token_option.get("inject_into", "request_parameter")
            if inject_into == "request_parameter" and field:
                params[field] = token
            elif inject_into == "header" and field:
                params.setdefault("_headers", {})[field] = str(token)
            elif self._page_token_option.get("type") == "RequestPath":
                params["_path_override"] = str(token)

        # Inject page size
        if self._page_size_option and hasattr(self._strategy, 'page_size'):
            field = eval_string(
                self._page_size_option.get("field_name", ""),
                self._config,
            )
            if field:
                params[field] = self._strategy.page_size

        return params


@component("OffsetIncrement")
class OffsetIncrement:
    def __init__(self, definition: dict, config: dict, **kwargs):
        self.page_size = eval_string(definition.get("page_size", 100), config)
        if isinstance(self.page_size, str):
            self.page_size = int(self.page_size)
        self._inject_on_first = definition.get("inject_on_first_request", False)
        self._offset = 0

    def next_token(self, response: requests.Response, last_records: list,
                   page_count: int) -> Optional[int]:
        if len(last_records) < self.page_size:
            self._offset = 0
            return None
        self._offset += len(last_records)
        return self._offset


@component("PageIncrement")
class PageIncrement:
    def __init__(self, definition: dict, config: dict, **kwargs):
        self.page_size = eval_string(definition.get("page_size", 100), config)
        if isinstance(self.page_size, str):
            self.page_size = int(self.page_size)
        self._start_from = definition.get("start_from_page", 0)

    def next_token(self, response: requests.Response, last_records: list,
                   page_count: int) -> Optional[int]:
        if len(last_records) < self.page_size:
            return None
        return self._start_from + page_count + 1


@component("CursorPagination")
class CursorPagination:
    def __init__(self, definition: dict, config: dict, **kwargs):
        self._cursor_value_template = definition.get("cursor_value", "")
        self._stop_condition_template = definition.get("stop_condition", "")
        self.page_size = eval_string(definition.get("page_size", ""), config)
        self._config = config

    def next_token(self, response: requests.Response, last_records: list,
                   page_count: int) -> Optional[str]:
        try:
            resp_body = response.json()
        except Exception:
            resp_body = {}

        headers = dict(response.headers)
        last_record = last_records[-1] if last_records else {}

        context = {
            "response": resp_body,
            "headers": headers,
            "last_record": last_record,
            "last_page_size": len(last_records),
        }

        # Check stop condition
        if self._stop_condition_template:
            stop = eval_string(self._stop_condition_template, self._config, **context)
            if stop is True or stop == "True" or stop == "true":
                return None

        # Evaluate cursor
        cursor = eval_string(self._cursor_value_template, self._config, **context)
        if not cursor or cursor == "None" or cursor == "none":
            return None
        return str(cursor)
