"""Tests for urc.components.rate_limiter — rate limiting strategies.

CRITICAL: All time is mocked — no real delays.
"""

from unittest.mock import patch, MagicMock

from urc.components.rate_limiter import (
    TokenBucket,
    MovingWindowRateLimiter,
    APIBudget,
    EndpointPolicy,
)


# ── TokenBucket ──


@patch("urc.components.rate_limiter.time")
def test_token_bucket_acquire_when_available(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    bucket = TokenBucket(max_tokens=5, refill_rate=1.0)
    wait = bucket.acquire(1.0)
    assert wait == 0.0
    mock_time.sleep.assert_not_called()


@patch("urc.components.rate_limiter.time")
def test_token_bucket_acquire_when_empty(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()

    bucket = TokenBucket(max_tokens=2, refill_rate=1.0, initial_tokens=0)

    # First acquire: no tokens, must wait
    # After sleep, monotonic advances
    def advance_after_sleep(duration):
        mock_time.monotonic.return_value = duration

    mock_time.sleep.side_effect = advance_after_sleep

    wait = bucket.acquire(1.0)
    assert wait == 1.0  # deficit=1.0, rate=1.0 => wait=1.0
    mock_time.sleep.assert_called_once_with(1.0)


@patch("urc.components.rate_limiter.time")
def test_token_bucket_try_acquire_success(mock_time):
    mock_time.monotonic.return_value = 0.0
    bucket = TokenBucket(max_tokens=5, refill_rate=1.0)
    assert bucket.try_acquire(1.0) is True


@patch("urc.components.rate_limiter.time")
def test_token_bucket_try_acquire_fail(mock_time):
    mock_time.monotonic.return_value = 0.0
    bucket = TokenBucket(max_tokens=1, refill_rate=1.0, initial_tokens=0)
    assert bucket.try_acquire(1.0) is False


@patch("urc.components.rate_limiter.time")
def test_token_bucket_refill_after_time(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    bucket = TokenBucket(max_tokens=5, refill_rate=2.0, initial_tokens=0)

    # Advance time by 2 seconds => 4 tokens refilled
    mock_time.monotonic.return_value = 2.0
    assert bucket.try_acquire(3.0) is True  # 4 >= 3


@patch("urc.components.rate_limiter.time")
def test_token_bucket_refill_caps_at_max(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    bucket = TokenBucket(max_tokens=5, refill_rate=10.0, initial_tokens=0)

    mock_time.monotonic.return_value = 100.0
    # Even after a long time, tokens cap at max_tokens=5
    assert bucket.try_acquire(5.0) is True
    assert bucket.try_acquire(1.0) is False


# ── MovingWindowRateLimiter ──


@patch("urc.components.rate_limiter.time")
def test_moving_window_acquire_within_limit(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    limiter = MovingWindowRateLimiter(max_requests=3, window_seconds=10.0)
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0
    mock_time.sleep.assert_not_called()


@patch("urc.components.rate_limiter.time")
def test_moving_window_blocks_at_capacity(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    limiter = MovingWindowRateLimiter(max_requests=2, window_seconds=10.0)
    limiter.acquire()
    limiter.acquire()

    # Third request should block
    def advance_after_sleep(duration):
        mock_time.monotonic.return_value = 10.0 + duration

    mock_time.sleep.side_effect = advance_after_sleep
    mock_time.monotonic.return_value = 1.0  # 1s after start

    wait = limiter.acquire()
    # Oldest timestamp is 0.0, window is 10.0, current is 1.0
    # wait = 0.0 + 10.0 - 1.0 = 9.0
    assert wait >= 0.0
    mock_time.sleep.assert_called()


@patch("urc.components.rate_limiter.time")
def test_moving_window_expires(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    limiter = MovingWindowRateLimiter(max_requests=1, window_seconds=5.0)
    limiter.acquire()

    # After window expires, should be able to acquire again
    mock_time.monotonic.return_value = 6.0
    assert limiter.try_acquire() is True


@patch("urc.components.rate_limiter.time")
def test_moving_window_try_acquire(mock_time):
    mock_time.monotonic.return_value = 0.0
    limiter = MovingWindowRateLimiter(max_requests=1, window_seconds=10.0)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False


# ── EndpointPolicy ──


def test_endpoint_policy_exact_match():
    p = EndpointPolicy("https://api.test/v1/users", 10, 60.0)
    assert p.matches("https://api.test/v1/users") is True
    assert p.matches("https://api.test/v1/orders") is False


def test_endpoint_policy_prefix_match():
    p = EndpointPolicy("https://api.test/v1/*", 10, 60.0)
    assert p.matches("https://api.test/v1/users") is True
    assert p.matches("https://api.test/v1/orders") is True
    assert p.matches("https://api.test/v2/users") is False


def test_endpoint_policy_contains_match():
    p = EndpointPolicy("*api.test*", 10, 60.0)
    assert p.matches("https://api.test/v1/users") is True
    assert p.matches("https://other.com/api.test/x") is True
    assert p.matches("https://other.com/foo") is False


def test_endpoint_policy_wildcard():
    p = EndpointPolicy("*", 10, 60.0)
    assert p.matches("https://any.url/anywhere") is True
    assert p.matches("http://localhost") is True


# ── APIBudget ──


@patch("urc.components.rate_limiter.time")
def test_api_budget_url_matching(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    budget = APIBudget(
        policies=[
            {
                "matcher": "https://api.test/v1/*",
                "max_requests": 100,
                "window_seconds": 60,
                "strategy": "moving_window",
            },
            {
                "matcher": "*",
                "max_requests": 1000,
                "window_seconds": 60,
                "strategy": "moving_window",
            },
        ]
    )
    wait = budget.acquire("https://api.test/v1/users")
    assert wait == 0.0


@patch("urc.components.rate_limiter.time")
def test_api_budget_try_acquire(mock_time):
    mock_time.monotonic.return_value = 0.0
    budget = APIBudget(
        policies=[
            {"matcher": "*", "max_requests": 1, "window_seconds": 60},
        ]
    )
    assert budget.try_acquire("https://any.url") is True
    assert budget.try_acquire("https://any.url") is False


@patch("urc.components.rate_limiter.time")
def test_api_budget_multi_policy(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    budget = APIBudget(
        policies=[
            {"matcher": "https://api.test/*", "max_requests": 2, "window_seconds": 10},
            {"matcher": "*", "max_requests": 5, "window_seconds": 10},
        ]
    )
    # First two calls match both policies
    budget.acquire("https://api.test/users")
    budget.acquire("https://api.test/users")
    # Third call: prefix policy exhausted
    assert budget.try_acquire("https://api.test/users") is False
    # But a different URL matches only the wildcard
    assert budget.try_acquire("https://other.com/users") is True
