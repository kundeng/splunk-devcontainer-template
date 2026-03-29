# Error handling and backoff components.

import logging
import time
from typing import Optional

import requests

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Raised when a request should be retried."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """Raised when a request should not be retried."""
    pass


@component("DefaultErrorHandler")
class DefaultErrorHandler:
    """Handles HTTP errors with configurable retry and backoff."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._max_retries = definition.get("max_retries", 5)
        self._backoff_time = 5.0  # default exponential base

        # Parse backoff strategies
        backoff_defs = definition.get("backoff_strategies", [])
        if backoff_defs:
            first = backoff_defs[0]
            btype = first.get("type", "")
            if btype == "ConstantBackoffStrategy":
                self._backoff_time = float(
                    eval_string(first.get("backoff_time_in_seconds", 5), config)
                )
            elif btype == "ExponentialBackoffStrategy":
                self._backoff_time = float(
                    eval_string(first.get("factor", 5), config)
                )

        # Parse response filters
        self._filters = definition.get("response_filters", [])

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, response: requests.Response) -> bool:
        """Check if the response indicates a retryable error."""
        # Check custom filters first
        for filt in self._filters:
            codes = filt.get("http_codes", [])
            if codes and response.status_code in codes:
                action = filt.get("action", "RETRY")
                if action == "RETRY":
                    return True
                elif action == "FAIL":
                    return False
                elif action == "IGNORE" or action == "SUCCESS":
                    return False

        # Default: retry on 429 and 5xx
        if response.status_code == 429:
            return True
        if response.status_code >= 500:
            return True
        return False

    def get_backoff_time(self, response: requests.Response, attempt: int) -> float:
        """Calculate wait time before retry."""
        # Respect Retry-After header
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

        # Exponential backoff: factor * 2^attempt
        return self._backoff_time * (2 ** attempt)
