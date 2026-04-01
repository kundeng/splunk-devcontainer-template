# Error handling and backoff components.

import re
import time
from typing import Any, List, Optional

import requests

from urc.interpolation import eval_string
from urc.registry import component


class RetryableError(Exception):
    """Raised when a request should be retried."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """Raised when a request should not be retried."""
    pass


# ── Backoff strategies ──


@component("ConstantBackoffStrategy")
class ConstantBackoffStrategy:
    """Always returns the same wait time."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        raw = definition.get("backoff_time_in_seconds", 5)
        self._backoff_time = float(eval_string(raw, config))

    def get_backoff_time(self, response: requests.Response, attempt: int) -> float:
        return self._backoff_time


@component("ExponentialBackoffStrategy")
class ExponentialBackoffStrategy:
    """Returns factor * 2^attempt."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._factor = float(definition.get("factor", 5))

    def get_backoff_time(self, response: requests.Response, attempt: int) -> float:
        return self._factor * (2 ** attempt)


@component("WaitTimeFromHeader")
class WaitTimeFromHeader:
    """Extracts wait time from a response header (e.g. Retry-After)."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._header = definition.get("header", "Retry-After")
        self._regex = definition.get("regex")
        self._max_wait = definition.get("max_waiting_time_in_seconds")

    def get_backoff_time(self, response: requests.Response, attempt: int) -> Optional[float]:
        """Return wait time from the header, or None if header is absent."""
        return self.get_wait_time(response)

    def get_wait_time(self, response: requests.Response) -> Optional[float]:
        """Extract wait time from the configured response header."""
        value = response.headers.get(self._header)
        if value is None:
            return None

        if self._regex:
            match = re.search(self._regex, value)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
            else:
                return None

        try:
            wait = float(value)
        except (ValueError, TypeError):
            return None

        if self._max_wait is not None and wait > self._max_wait:
            wait = float(self._max_wait)
        return wait


# ── Response filter ──


@component("HttpResponseFilter")
class HttpResponseFilter:
    """Matches a response against configurable criteria.

    Fields:
        action: One of SUCCESS, FAIL, RETRY, IGNORE.
        http_codes: List of integer HTTP status codes to match.
        error_message_contains: Substring to search for in the response body.
        predicate: Jinja2 expression evaluated with response context.
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._action = definition.get("action", "RETRY")
        self._http_codes: List[int] = definition.get("http_codes", [])
        self._error_message_contains: Optional[str] = definition.get("error_message_contains")
        self._predicate: Optional[str] = definition.get("predicate")
        self._config = config

    @property
    def action(self) -> str:
        return self._action

    def matches(self, response: requests.Response) -> bool:
        """Return True if the response matches this filter's criteria."""
        # Check HTTP status codes
        if self._http_codes and response.status_code not in self._http_codes:
            return False

        # Check error message substring in response body
        if self._error_message_contains is not None:
            try:
                body_text = response.text
            except Exception:
                body_text = ""
            if self._error_message_contains not in body_text:
                return False

        # Check Jinja2 predicate
        if self._predicate is not None:
            try:
                body_text = response.text
            except Exception:
                body_text = ""
            try:
                headers_dict = dict(response.headers)
            except Exception:
                headers_dict = {}
            result = eval_string(
                self._predicate,
                self._config,
                response={"body": body_text, "headers": headers_dict, "status_code": response.status_code},
            )
            if not result or result == self._predicate:
                return False

        return True


# ── Error handlers ──


@component("DefaultErrorHandler")
class DefaultErrorHandler:
    """Handles HTTP errors with configurable retry and backoff."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        self._max_retries = definition.get("max_retries", 5)

        # Build backoff strategy components
        self._backoff_strategies: list = []
        backoff_defs = definition.get("backoff_strategies", [])
        for bdef in backoff_defs:
            if isinstance(bdef, dict) and bdef.get("type"):
                self._backoff_strategies.append(create_component(bdef, config))

        # Fallback: exponential with factor 5 when no strategies configured
        if not self._backoff_strategies:
            self._backoff_strategies.append(
                ExponentialBackoffStrategy({"factor": 5}, config)
            )

        # Build response filter components
        self._filters: list = []
        filter_defs = definition.get("response_filters", [])
        for fdef in filter_defs:
            if isinstance(fdef, dict):
                if fdef.get("type"):
                    self._filters.append(create_component(fdef, config))
                else:
                    # Legacy: plain dict filters (kept for backwards compat)
                    self._filters.append(fdef)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, response: requests.Response) -> bool:
        """Check if the response indicates a retryable error."""
        # Check response filters first
        for filt in self._filters:
            if isinstance(filt, HttpResponseFilter):
                if filt.matches(response):
                    return filt.action == "RETRY"
            elif isinstance(filt, dict):
                # Legacy plain-dict filter
                codes = filt.get("http_codes", [])
                if codes and response.status_code in codes:
                    action = filt.get("action", "RETRY")
                    if action == "RETRY":
                        return True
                    return False

        # Default: retry on 429 and 5xx
        if response.status_code == 429:
            return True
        if response.status_code >= 500:
            return True
        return False

    def get_backoff_time(self, response: requests.Response, attempt: int) -> float:
        """Calculate wait time before retry.

        Iterates backoff strategies in order. WaitTimeFromHeader strategies
        are checked first (they may return None if the header is absent).
        The first strategy that returns a value wins.
        """
        for strategy in self._backoff_strategies:
            wait = strategy.get_backoff_time(response, attempt)
            if wait is not None:
                return float(wait)

        # Absolute fallback
        return 5.0 * (2 ** attempt)


@component("CompositeErrorHandler")
class CompositeErrorHandler:
    """Delegates to a list of error handlers; first match wins."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        self._max_retries = definition.get("max_retries", 5)
        self._error_handlers: list = []
        for handler_def in definition.get("error_handlers", []):
            if isinstance(handler_def, dict) and handler_def.get("type"):
                self._error_handlers.append(create_component(handler_def, config))

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, response: requests.Response) -> bool:
        """Iterate handlers in order; first match wins."""
        for handler in self._error_handlers:
            result = handler.should_retry(response)
            if result is not None:
                return result
        # Default fallback: retry on 429 / 5xx
        if response.status_code == 429:
            return True
        if response.status_code >= 500:
            return True
        return False

    def get_backoff_time(self, response: requests.Response, attempt: int) -> float:
        """Delegate to the first handler that provides a backoff time."""
        for handler in self._error_handlers:
            try:
                wait = handler.get_backoff_time(response, attempt)
                if wait is not None:
                    return float(wait)
            except Exception:
                continue
        return 5.0 * (2 ** attempt)
