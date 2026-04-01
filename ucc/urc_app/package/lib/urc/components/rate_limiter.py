# Rate limiting — token bucket + API budget per endpoint.
#
# Pure Python, stdlib only (time.monotonic + collections.deque).
# No native deps, no threading (Splunk inputs are single-threaded).
#
# Supported strategies:
#   - TokenBucket: classic token bucket with burst capacity
#   - MovingWindowRateLimiter: sliding window counter
#   - APIBudget: per-endpoint rate limits with configurable policies

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from urc.registry import component


# ── Token Bucket ──


class TokenBucket:
    """Classic token bucket algorithm.

    Allows bursts up to `max_tokens` with a steady refill rate.
    Uses time.monotonic() for clock — immune to wall-clock jumps.
    """

    def __init__(
        self,
        max_tokens: float,
        refill_rate: float,
        initial_tokens: Optional[float] = None,
    ):
        """
        Args:
            max_tokens: Maximum tokens (burst capacity).
            refill_rate: Tokens added per second.
            initial_tokens: Starting tokens (defaults to max_tokens).
        """
        self._max_tokens = float(max_tokens)
        self._refill_rate = float(refill_rate)
        self._tokens = float(initial_tokens if initial_tokens is not None else max_tokens)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, blocking if necessary.

        Returns:
            Actual wait time in seconds (0 if tokens were available).
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0

        # Calculate wait time
        deficit = tokens - self._tokens
        wait_time = deficit / self._refill_rate
        time.sleep(wait_time)
        self._refill()
        self._tokens -= tokens
        return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without blocking.

        Returns:
            True if tokens were available, False otherwise.
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """Return seconds to wait before tokens would be available (no consume)."""
        self._refill()
        if self._tokens >= tokens:
            return 0.0
        deficit = tokens - self._tokens
        return deficit / self._refill_rate

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


# ── Moving Window Rate Limiter ──


class MovingWindowRateLimiter:
    """Sliding window rate limiter using a deque of timestamps.

    Allows up to `max_requests` in any `window_seconds` period.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window = float(window_seconds)
        self._timestamps: Deque[float] = deque()

    def _purge_old(self) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.monotonic() - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def acquire(self) -> float:
        """Acquire a request slot, blocking if necessary.

        Returns:
            Actual wait time in seconds.
        """
        self._purge_old()
        if len(self._timestamps) < self._max_requests:
            self._timestamps.append(time.monotonic())
            return 0.0

        # Wait until the oldest request exits the window
        wait_time = self._timestamps[0] + self._window - time.monotonic()
        if wait_time > 0:
            time.sleep(wait_time)
        self._purge_old()
        self._timestamps.append(time.monotonic())
        return max(wait_time, 0.0)

    def try_acquire(self) -> bool:
        """Try to acquire without blocking."""
        self._purge_old()
        if len(self._timestamps) < self._max_requests:
            self._timestamps.append(time.monotonic())
            return True
        return False

    @property
    def available_slots(self) -> int:
        self._purge_old()
        return max(0, self._max_requests - len(self._timestamps))


# ── API Budget ──


class EndpointPolicy:
    """Rate limit policy for a single endpoint pattern."""

    def __init__(
        self,
        matcher: str,
        max_requests: int,
        window_seconds: float,
        strategy: str = "moving_window",
        burst_capacity: Optional[int] = None,
    ):
        self.matcher = matcher
        self.strategy = strategy

        if strategy == "token_bucket":
            refill_rate = max_requests / window_seconds
            self.limiter = TokenBucket(
                max_tokens=float(burst_capacity or max_requests),
                refill_rate=refill_rate,
            )
        else:
            self.limiter = MovingWindowRateLimiter(max_requests, window_seconds)

    def matches(self, url: str) -> bool:
        """Check if URL matches this policy's pattern.

        Supports:
        - Exact match
        - Prefix match (pattern ends with *)
        - Contains match (pattern starts and ends with *)
        """
        if self.matcher == "*":
            return True
        if self.matcher.endswith("*") and not self.matcher.startswith("*"):
            return url.startswith(self.matcher[:-1])
        if self.matcher.startswith("*") and self.matcher.endswith("*"):
            return self.matcher[1:-1] in url
        return url == self.matcher

    def acquire(self) -> float:
        return self.limiter.acquire()

    def try_acquire(self) -> bool:
        return self.limiter.try_acquire()


class APIBudget:
    """Per-endpoint rate limiting with configurable policies.

    Manifest schema:
        rate_limiter:
            type: APIBudget
            policies:
                - matcher: "https://api.example.com/v1/*"
                  max_requests: 100
                  window_seconds: 60
                  strategy: moving_window
                - matcher: "*"
                  max_requests: 1000
                  window_seconds: 60
                  strategy: token_bucket
                  burst_capacity: 50
    """

    def __init__(self, policies: Optional[List[dict]] = None):
        self._policies: List[EndpointPolicy] = []
        for p in (policies or []):
            self._policies.append(EndpointPolicy(
                matcher=p.get("matcher", "*"),
                max_requests=int(p.get("max_requests", 100)),
                window_seconds=float(p.get("window_seconds", 60)),
                strategy=p.get("strategy", "moving_window"),
                burst_capacity=p.get("burst_capacity"),
            ))

    def acquire(self, url: str) -> float:
        """Acquire rate limit tokens for a URL. Returns total wait time."""
        total_wait = 0.0
        for policy in self._policies:
            if policy.matches(url):
                wait = policy.acquire()
                total_wait += wait
        return total_wait

    def try_acquire(self, url: str) -> bool:
        """Try to acquire without blocking. Returns False if any policy would block."""
        for policy in self._policies:
            if policy.matches(url):
                if not policy.try_acquire():
                    return False
        return True


# ── Component registration ──


@component("APIBudget")
class APIBudgetComponent:
    """Registered component wrapper for APIBudget.

    Can be attached to an HttpRequester via the rate_limiter field.
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        policies = definition.get("policies", [])
        self._budget = APIBudget(policies)

    def acquire(self, url: str) -> float:
        return self._budget.acquire(url)

    def try_acquire(self, url: str) -> bool:
        return self._budget.try_acquire(url)


@component("TokenBucketRateLimiter")
class TokenBucketRateLimiter:
    """Simple token bucket rate limiter as a component.

    Manifest schema:
        rate_limiter:
            type: TokenBucketRateLimiter
            max_tokens: 10
            refill_rate: 2.0
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        max_tokens = float(definition.get("max_tokens", 10))
        refill_rate = float(definition.get("refill_rate", 1.0))
        self._bucket = TokenBucket(max_tokens, refill_rate)

    def acquire(self, url: str = "") -> float:
        return self._bucket.acquire()

    def try_acquire(self, url: str = "") -> bool:
        return self._bucket.try_acquire()


@component("MovingWindowRateLimiterComponent")
class MovingWindowRateLimiterComponent:
    """Simple moving window rate limiter as a component.

    Manifest schema:
        rate_limiter:
            type: MovingWindowRateLimiterComponent
            max_requests: 100
            window_seconds: 60
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        max_requests = int(definition.get("max_requests", 100))
        window_seconds = float(definition.get("window_seconds", 60))
        self._limiter = MovingWindowRateLimiter(max_requests, window_seconds)

    def acquire(self, url: str = "") -> float:
        return self._limiter.acquire()

    def try_acquire(self, url: str = "") -> bool:
        return self._limiter.try_acquire()
