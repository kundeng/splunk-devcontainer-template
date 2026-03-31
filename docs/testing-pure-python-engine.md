# Testing a Pure-Python REST Engine: A Ground-Up Guide

> **Goal:** You need to test a Python application that makes HTTP requests to REST APIs,
> handles authentication, pagination, rate limiting, checkpointing, and runs inside
> Splunk's modular input framework. You want tests that run in under a second, need no
> network access, and catch real bugs.
>
> This doc explains every testing concept from scratch — what pytest fixtures are, how
> HTTP mocking works, why you mock time, and how to test against a framework (Splunk SDK)
> you can't install in CI. Every code example is from the actual test suite in
> `ucc/urc_app/tests/`, not hypothetical.
>
> **Audience:** Developers who can write Python but haven't used pytest mocking before,
> or anyone inheriting this project who needs to understand what the tests do and how
> to add new ones.

---

## Table of Contents

1. [The Problem: Why Can't We Just Call the APIs?](#1-the-problem)
2. [The Test Stack: Three Tools](#2-the-test-stack)
3. [pytest Basics: Fixtures, Not setUp/tearDown](#3-pytest-basics)
4. [HTTP Mocking with `responses`: Faking the Internet](#4-http-mocking)
5. [Testing Components in Isolation (Unit Tests)](#5-unit-tests)
6. [Mocking Time: Testing Rate Limiters Without Sleeping](#6-mocking-time)
7. [Testing Auth Flows That Call External Endpoints](#7-testing-auth)
8. [Integration Tests: Testing the Full Pipeline](#8-integration-tests)
9. [Testing Incremental Sync and Checkpointing](#9-testing-checkpointing)
10. [Testing Async Bulk Jobs (Create → Poll → Download)](#10-testing-async)
11. [Testing Against Splunk Without Splunk](#11-testing-splunk)
12. [The conftest.py File: Shared Setup for Every Test](#12-conftest)
13. [Fixtures Directory: Reusable Building Blocks](#13-fixtures)
14. [Adding Your Own Tests](#14-adding-tests)

---

## 1. The Problem: Why Can't We Just Call the APIs? {#1-the-problem}

The URC engine talks to REST APIs — GitHub, ServiceNow, Azure, anything with a URL.
You could test it by pointing it at real APIs. We do that too (see `test_manifests.py`).
But real-API tests have three fatal problems:

1. **They need credentials.** Your CI server doesn't have your GitHub PAT. A new
   developer doesn't have a ServiceNow instance. Tests that need secrets are tests
   that don't run.

2. **They're non-deterministic.** GitHub might return 50 repos today and 51 tomorrow.
   A rate limit might kick in. The API might be down. Your test passes Monday, fails
   Tuesday, and nobody trusts the test suite.

3. **They're slow.** Each HTTP request takes 100-500ms. The engine makes dozens of
   requests per collection. A 176-test suite hitting real APIs would take minutes.
   Ours runs in 0.19 seconds.

The solution: fake the HTTP layer. Your code calls `requests.get("https://api.example.com/users")`,
but instead of going to the internet, a library intercepts the call and returns whatever
JSON you configured. Your code can't tell the difference.

---

## 2. The Test Stack: Three Tools {#2-the-test-stack}

We use three things. That's it.

### pytest

The test runner. You write functions that start with `test_`, put `assert` statements
in them, and pytest finds and runs them. No classes needed, no `unittest.TestCase`
inheritance.

```bash
pip install pytest
python -m pytest ucc/urc_app/tests/ -v
```

### responses

A library that patches Python's `requests` library. When your code does
`requests.get("https://api.test/data")`, `responses` intercepts it and returns
whatever you told it to. The real HTTP stack never runs.

```bash
pip install responses
```

### unittest.mock (built into Python)

For everything that isn't HTTP: mocking `time.time()`, `time.sleep()`,
`time.monotonic()`, and the Splunk SDK. It's in the standard library — nothing
to install.

---

## 3. pytest Basics: Fixtures, Not setUp/tearDown {#3-pytest-basics}

If you've used `unittest`, you're used to classes with `setUp()` and `tearDown()`.
pytest uses **fixtures** instead. A fixture is a function that provides something
your test needs — a config dict, a database connection, a mock server.

### A simple fixture

```python
# conftest.py
import pytest

@pytest.fixture
def mock_config():
    return {
        "base_url": "https://mock-api.test",
        "api_key": "test-key-123",
        "username": "testuser",
        "password": "testpass",
    }
```

### Using it in a test

You don't call the fixture. You put its name as a parameter, and pytest injects it:

```python
# test_something.py
def test_bearer_auth_sets_header(mock_config):    # ← pytest sees "mock_config", calls the fixture
    auth = BearerAuth({"api_token": "{{ config['api_key'] }}"}, mock_config)
    session = requests.Session()
    auth.apply(session, mock_config)
    assert session.headers["Authorization"] == "Bearer test-key-123"
```

There's no `self`, no class, no inheritance. The function name starts with `test_`,
it takes a fixture parameter, and it uses `assert`. That's a complete test.

### The `autouse` fixture: runs for every test automatically

```python
# conftest.py
@pytest.fixture(autouse=True)
def activate_responses():
    with responses.RequestsMock() as rsps:
        yield rsps
```

`autouse=True` means this runs before *every test in the directory*, without you
asking for it. It activates the `responses` mock, yields control to your test,
then tears it down. If your test tries to make a real HTTP request that isn't
mocked, it raises `ConnectionError` — which is exactly what you want. No
accidental network calls.

### How `yield` works in fixtures

```python
@pytest.fixture
def my_fixture():
    # Setup: runs BEFORE the test
    resource = create_something()
    yield resource                  # ← test runs here, with 'resource' available
    # Teardown: runs AFTER the test
    resource.cleanup()
```

The `yield` splits the fixture into "before" and "after". For `activate_responses`,
the "before" activates mocking, and the "after" (implicit in `with`) deactivates it
and checks that all expected requests were made.

---

## 4. HTTP Mocking with `responses`: Faking the Internet {#4-http-mocking}

This is the core concept. Understanding `responses` unlocks 90% of the test suite.

### The simplest case: mock a GET endpoint

```python
def test_simple_get(activate_responses, mock_config):
    rsps = activate_responses

    # Tell responses: "when someone GETs this URL, return this JSON"
    rsps.add(
        rsps.GET,
        "https://mock-api.test/users",
        json=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        status=200,
    )

    # Now run the engine — it will "call" the URL and get our fake data
    results = list(collect(SIMPLE_GET_MANIFEST, mock_config))
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 2
    assert data_records[0]["name"] == "Alice"
```

What happens under the hood:

1. `rsps.add(...)` registers a rule: "GET to this URL → return this response"
2. `collect()` calls `requests.get("https://mock-api.test/users")` internally
3. `responses` intercepts the call, matches it against registered rules
4. The fake response (status 200, JSON body) is returned to `collect()`
5. `collect()` processes it as if it came from a real server

The URL in your manifest (`https://mock-api.test/users`) must match the URL in
`rsps.add()` exactly. If they don't match, `responses` raises `ConnectionError`.

### Dynamic responses with callbacks

Sometimes you need the response to vary based on the request — for example,
returning different pages based on an `offset` query parameter:

```python
def test_paginated_pipeline(activate_responses, mock_config):
    rsps = activate_responses
    all_items = [{"id": i, "name": f"item_{i}"} for i in range(25)]

    def paginated_callback(request):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(request.url).query)
        offset = int(qs.get("offset", ["0"])[0])
        limit = int(qs.get("limit", ["10"])[0])
        page = all_items[offset:offset + limit]
        return (200, {}, json.dumps({"data": page}))

    rsps.add_callback(
        rsps.GET,
        "https://mock-api.test/items",
        callback=paginated_callback,
        content_type="application/json",
    )

    results = list(collect(PAGINATED_OFFSET_MANIFEST, mock_config))
    data_records = [rec for _, rec, _ in results if rec]
    assert len(data_records) == 25  # 10 + 10 + 5
```

The callback receives the `request` object and returns a tuple of
`(status_code, headers_dict, body_string)`. You can inspect `request.url`,
`request.headers`, `request.body` — anything the engine sent.

This is how you simulate pagination: the engine sends `?offset=0&limit=10`,
gets 10 records, sends `?offset=10&limit=10`, gets 10 more, sends
`?offset=20&limit=10`, gets 5, sees less than page_size, stops.

### Simulating errors

```python
def register_rate_limited_api(rsps, base_url, fail_count):
    call_count = {"n": 0}

    def rate_limit_callback(request):
        call_count["n"] += 1
        if call_count["n"] <= fail_count:
            return (429, {"Retry-After": "1"}, json.dumps({"error": "rate limited"}))
        return (200, {}, json.dumps({"data": [{"id": 1}]}))

    rsps.add_callback(
        rsps.GET,
        f"{base_url}/rate-limited",
        callback=rate_limit_callback,
        content_type="application/json",
    )
```

The callback uses a closure (`call_count`) to track how many times it's been
called. The first N calls return HTTP 429 (Too Many Requests) with a
`Retry-After` header. After that, it returns 200. This tests that the engine's
retry logic actually works.

---

## 5. Testing Components in Isolation (Unit Tests) {#5-unit-tests}

The engine is built from small, registered components: authenticators, paginators,
decoders, transformations, etc. Each one can be tested independently because they
have a simple interface: instantiate with a definition dict and config, then call
the main method.

### Pattern: create → call → assert

Every unit test follows this shape:

```python
def test_keys_to_snake_camel_case():
    # 1. Create the component (no HTTP, no engine, no manifest)
    t = KeysToSnakeCase({}, {})

    # 2. Call its main method with test input
    result = t.transform({"firstName": "Alice", "lastName": "Bob"}, {})

    # 3. Assert the output
    assert result == {"first_name": "Alice", "last_name": "Bob"}
```

No fixtures needed. No mocking. Pure function in, data out. This is the
fastest kind of test to write and the easiest to debug.

### Testing with Jinja2 interpolation

Many components accept Jinja2 templates in their definition. The config
dict provides the variables:

```python
def test_add_fields_jinja_template(mock_config):
    t = AddFields(
        {
            "fields": [
                {"path": ["greeting"], "value": "Hello {{ config['username'] }}"},
            ]
        },
        mock_config,
    )
    record = {"id": 1}
    result = t.transform(record, mock_config)
    assert result["greeting"] == "Hello testuser"
```

The `{{ config['username'] }}` template is resolved at transform time using the
config dict. This is how manifests parameterize behavior without hardcoding values.

---

## 6. Mocking Time: Testing Rate Limiters Without Sleeping {#6-mocking-time}

Rate limiters use `time.monotonic()` to track elapsed time and `time.sleep()` to
wait. If you tested them with real time, a test that verifies "wait 60 seconds when
the bucket is empty" would take 60 seconds. With 13 rate limiter tests, you'd be
waiting over a minute for one file.

The solution: replace `time` with a mock that you control.

### How `@patch` works

```python
from unittest.mock import patch, MagicMock

@patch("urc.components.rate_limiter.time")
def test_token_bucket_acquire_when_available(mock_time):
    # mock_time replaces the `time` module inside rate_limiter.py
    mock_time.monotonic.return_value = 0.0    # "time is 0.0 seconds"
    mock_time.sleep = MagicMock()             # sleep does nothing

    bucket = TokenBucket(max_tokens=5, refill_rate=1.0)
    wait = bucket.acquire(1.0)

    assert wait == 0.0                        # tokens available, no wait
    mock_time.sleep.assert_not_called()       # never slept
```

`@patch("urc.components.rate_limiter.time")` does this: inside
`urc/components/rate_limiter.py`, every call to `time.monotonic()` or
`time.sleep()` now goes to your mock object instead of the real `time` module.

The string `"urc.components.rate_limiter.time"` is the **import path** — where
the `time` module is used, not where it's defined. This is critical: you patch
the name as seen by the code under test.

### Controlling the clock

For tests that need time to advance (like testing token refill):

```python
@patch("urc.components.rate_limiter.time")
def test_token_bucket_refill_after_time(mock_time):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()
    bucket = TokenBucket(max_tokens=5, refill_rate=2.0, initial_tokens=0)

    # Advance time by 2 seconds => 4 tokens refilled (rate=2.0/s * 2s)
    mock_time.monotonic.return_value = 2.0
    assert bucket.try_acquire(3.0) is True   # 4 available >= 3 needed
```

You set `mock_time.monotonic.return_value` to whatever "time" you want. The
bucket's `_refill()` method calculates `elapsed = now - last_refill` and adds
`elapsed * refill_rate` tokens. Since you control "now", you control how many
tokens get added.

### Making sleep advance time

For `acquire()` (which blocks), the test needs `sleep()` to actually move
the clock forward:

```python
@patch("urc.components.rate_limiter.time")
def test_token_bucket_acquire_when_empty(mock_time):
    mock_time.monotonic.return_value = 0.0

    def advance_after_sleep(duration):
        mock_time.monotonic.return_value = duration

    mock_time.sleep.side_effect = advance_after_sleep

    bucket = TokenBucket(max_tokens=2, refill_rate=1.0, initial_tokens=0)
    wait = bucket.acquire(1.0)

    assert wait == 1.0
    mock_time.sleep.assert_called_once_with(1.0)
```

`side_effect` is a function that runs when `sleep()` is called. Here, it
advances the monotonic clock by the sleep duration, so when `acquire()` calls
`_refill()` after sleeping, it sees that time has passed and adds tokens.

This entire test runs in microseconds. No actual sleeping.

---

## 7. Testing Auth Flows That Call External Endpoints {#7-testing-auth}

Some auth strategies make HTTP calls themselves — OAuth2 calls a token endpoint,
SessionToken calls a login endpoint. These tests combine component testing with
HTTP mocking.

### OAuth2: mock the token endpoint

```python
def test_oauth2_auth_fetches_token(mock_config, activate_responses):
    # 1. Mock the token endpoint
    activate_responses.add(
        responses.POST,
        "https://auth.test/token",
        json={"access_token": "oauth-token-xyz", "expires_in": 3600},
        status=200,
    )

    # 2. Create the OAuth2 authenticator
    session = requests.Session()
    auth = OAuth2Auth(
        {
            "client_id": "my-client",
            "client_secret": "my-secret",
            "token_refresh_endpoint": "https://auth.test/token",
            "grant_type": "client_credentials",
        },
        mock_config,
    )

    # 3. Apply — this triggers the token fetch internally
    auth.apply(session, mock_config)

    # 4. Assert the token was set
    assert session.headers["Authorization"] == "Bearer oauth-token-xyz"
```

When `auth.apply()` runs, it calls `requests.post("https://auth.test/token", ...)`
internally to get a token. The `responses` library intercepts this call and returns
the fake token response. The auth component processes it exactly as it would a
real response — extracting `access_token` from the JSON body and setting the
`Authorization` header.

The test proves: "Given a working token endpoint that returns this JSON, the
OAuth2 component correctly extracts the token and sets the Bearer header."

### Why two fixtures?

Notice the test takes both `mock_config` and `activate_responses`. That's because:
- `mock_config` provides the config dict (injected explicitly)
- `activate_responses` provides the mock HTTP layer (also `autouse`, but we need
  the reference to register URLs)

pytest resolves both fixtures before calling the test function.

---

## 8. Integration Tests: Testing the Full Pipeline {#8-integration-tests}

Unit tests verify components in isolation. Integration tests verify that the
components *work together* — that the engine reads a manifest, builds the right
components, makes HTTP calls, applies transformations, resolves `_time`, and
yields records with correct state.

### The integration test pattern

```python
def test_simple_get_pipeline(activate_responses, mock_config):
    rsps = activate_responses

    # 1. Mock the API endpoint
    rsps.add(rsps.GET, "https://mock-api.test/users",
             json=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
             status=200)

    # 2. Call the engine with a real manifest
    results = list(collect(SIMPLE_GET_MANIFEST, mock_config))

    # 3. Assert what came out
    data_records = [(name, rec, st) for name, rec, st in results if rec]
    assert len(data_records) == 2
    assert data_records[0][0] == "users"            # stream name
    assert data_records[0][1]["name"] == "Alice"    # record data
    for _, rec, _ in data_records:
        assert "_time" in rec                       # Splunk timestamp present
        assert isinstance(rec["_time"], float)      # epoch float
```

`SIMPLE_GET_MANIFEST` is a real YAML manifest (from `fixtures/sample_manifests.py`):

```yaml
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: users
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/users"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
```

The engine parses this YAML, builds a `SimpleRetriever` with an `HttpRequester`,
makes the GET request (intercepted by `responses`), extracts records, resolves
`_time`, and yields them. The test verifies the entire chain.

### What integration tests catch that unit tests don't

- Component wiring: does the engine pass `stream_slice` correctly to the requester?
- Pagination loop: does the retriever call the paginator and fetch all pages?
- State flow: does the cursor observe records and produce correct final state?
- `_time` injection: does the engine's timestamp resolver run after transformations?

---

## 9. Testing Incremental Sync and Checkpointing {#9-testing-checkpointing}

Incremental sync is where a stream remembers where it left off. On the first run,
it collects everything since `start_datetime`. On the second run, it passes the
previous cursor value and only collects new data.

### Testing the round-trip

```python
def test_state_round_trip(activate_responses):
    rsps = activate_responses
    call_count = {"n": 0}

    def callback(request):
        call_count["n"] += 1
        qs = parse_qs(urlparse(request.url).query)
        since = qs.get("since", [""])[0]
        if since and since >= "2024-05-01":
            return (200, {}, json.dumps([
                {"id": 10, "updated_at": "2024-06-01T00:00:00Z"},
            ]))
        return (200, {}, json.dumps([
            {"id": 1, "updated_at": "2024-05-01T00:00:00Z"},
        ]))

    rsps.add_callback(rsps.GET, "https://mock-api.test/events",
                       callback=callback, content_type="application/json")

    config = {"base_url": "https://mock-api.test"}

    # First run: no checkpoint
    results1 = list(collect(INCREMENTAL_MANIFEST, config))
    states1 = [st for _, _, st in results1 if st]
    final_state1 = states1[-1]
    assert final_state1["cursor_value"] == "2024-05-01T00:00:00Z"

    # Second run: pass the state back as checkpoint
    checkpoint = {"events": final_state1}
    results2 = list(collect(INCREMENTAL_MANIFEST, config, checkpoint=checkpoint))
    data2 = [rec for _, rec, _ in results2 if rec]
    assert len(data2) == 1
    assert data2[0]["id"] == 10  # only the new record

    states2 = [st for _, _, st in results2 if st]
    assert states2[-1]["cursor_value"] == "2024-06-01T00:00:00Z"
```

The callback changes its response based on the `since` parameter. On the first
call (`since=""`) it returns the old record. On the second call
(`since="2024-05-01"`) it returns the new record. This simulates an API that
supports cursor-based filtering.

### Testing per-partition state

When a stream has a partition router (e.g., one request per region), each partition
needs its own cursor. The test creates 12 partitions and verifies all 12 have
distinct cursor values in the final state:

```python
def test_many_partitions_fidelity(activate_responses):
    # ... (12 partitions, each returns a unique timestamp)

    final_state = all_states[-1]
    assert "partitions" in final_state
    assert len(final_state["partitions"]) == 12

    cursor_values = [pdata["cursor"]["cursor_value"]
                     for pdata in final_state["partitions"].values()]
    assert len(set(cursor_values)) == 12  # All unique
```

This catches serialization bugs — if the state manager accidentally reused the
same cursor for multiple partitions, you'd see duplicates.

---

## 10. Testing Async Bulk Jobs (Create → Poll → Download) {#10-testing-async}

Some APIs work asynchronously: you POST to create a job, poll a status endpoint
until it completes, then GET the results. The `AsyncRetriever` handles this
lifecycle. Testing it requires mocking three endpoints and controlling time.

```python
@patch("urc.components.async_retriever.time")
def test_happy_path(mock_time, activate_responses):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()

    rsps = activate_responses

    # 1. Create job
    rsps.add(rsps.POST, f"{BASE_URL}/jobs",
             json={"job_id": "j42", "status": "accepted"}, status=202)

    # 2. Poll: pending → processing → complete
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/status",
             json={"job_id": "j42", "status": "pending"}, status=200)
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/status",
             json={"job_id": "j42", "status": "processing"}, status=200)
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/status",
             json={"job_id": "j42", "status": "complete"}, status=200)

    # 3. Download results
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/results",
             json=[{"id": 1, "data": "result1"}, {"id": 2, "data": "result2"}],
             status=200)

    retriever = _build_async_retriever(config)
    records = list(retriever.read_records(config))

    assert len(records) == 2
```

Key detail: when you call `rsps.add()` multiple times for the same URL, `responses`
returns them **in order**. The first GET to `/jobs/j42/status` returns "pending",
the second returns "processing", the third returns "complete". This is how you
simulate a multi-step polling sequence without callbacks.

### Testing timeout

```python
@patch("urc.components.async_retriever.time")
def test_timeout(mock_time, activate_responses):
    call_count = {"n": 0}
    def monotonic_side_effect():
        call_count["n"] += 1
        return call_count["n"] * 2.0       # 2s per call

    mock_time.monotonic.side_effect = monotonic_side_effect
    mock_time.sleep = MagicMock()

    # Always return "processing" — never completes
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/jtimeout/status",
             json={"status": "processing"}, status=200)

    retriever = _build_async_retriever(config, max_poll_time_seconds=5)
    records = list(retriever.read_records(config))

    assert len(records) == 0    # timed out, no results
```

`side_effect` is a function that's called every time `monotonic()` is called. Each
call returns a time 2 seconds later than the last. After 3 calls (6 seconds), the
retriever exceeds `max_poll_time_seconds=5` and gives up.

---

## 11. Testing Against Splunk Without Splunk {#11-testing-splunk}

The engine runs inside Splunk's modular input framework. It reads credentials from
Splunk's encrypted password storage, saves checkpoints to KV Store, and writes
events with `_time`. But our CI server doesn't have Splunk installed.

The solution: mock the Splunk SDK with `MagicMock`.

### What is MagicMock?

`MagicMock` is a Python object that pretends to be anything. You can call any
method on it, access any attribute, subscript it — it just returns more
`MagicMock` objects. Then you configure specific behaviors:

```python
from unittest.mock import MagicMock

service = MagicMock()

# Make service.storage_passwords return a list of credential objects
cred = MagicMock()
cred.realm = "my_account"
cred.username = "api_user"
cred.clear_password = "api_secret"
service.storage_passwords = [cred]

# Make KV Store return a checkpoint
kv_data = MagicMock()
kv_data.query.return_value = [
    {"_key": "test_input", "state": '{"cursor_value": "2024-06-01"}'}
]
service.kvstore.__getitem__ = MagicMock(
    return_value=MagicMock(data=kv_data)
)
```

Now `service.storage_passwords` behaves like a real Splunk passwords collection,
and `service.kvstore["checkpoints"].data.query(...)` returns your fake checkpoint.

### The modular input test

```python
def test_first_run_saves_checkpoint(activate_responses):
    rsps = activate_responses
    rsps.add(rsps.GET, "https://mock-api.test/items",
             json=[
                 {"id": 1, "updated_at": "2024-05-01T00:00:00Z"},
                 {"id": 2, "updated_at": "2024-06-01T00:00:00Z"},
             ], status=200)

    service, kv_data = _build_mock_service(checkpoint_state=None)
    input_config = {"name": "test_input", "account": "my_account",
                     "base_url": "https://mock-api.test"}

    events, last_state = run_modular_input(service, input_config, MANIFEST)

    # Verify records were collected
    assert len(events) == 2

    # Verify checkpoint was saved to KV Store
    kv_data.update.assert_called_once()
    saved_state = json.loads(kv_data.update.call_args[0][1]["state"])
    assert saved_state["cursor_value"] == "2024-06-01T00:00:00Z"
```

This test verifies the complete lifecycle:
1. Credentials were read from `storage_passwords` (mocked)
2. Checkpoint was read from KV Store (mocked — returned empty)
3. Engine collected records (HTTP mocked via `responses`)
4. Events were produced with correct `_time`
5. Checkpoint was saved to KV Store (`kv_data.update.assert_called_once()`)

### Verifying what was saved

`assert_called_once()` checks that the mock was called exactly once.
`call_args` gives you the arguments it was called with:

```python
kv_data.update.assert_called_once()
call_args = kv_data.update.call_args
# call_args[0] = positional args, call_args[1] = keyword args
key = call_args[0][0]              # "test_input"
data = call_args[0][1]             # {"_key": "test_input", "state": "..."}
saved_state = json.loads(data["state"])
assert saved_state["cursor_value"] == "2024-06-01T00:00:00Z"
```

---

## 12. The conftest.py File: Shared Setup for Every Test {#12-conftest}

`conftest.py` is pytest's convention for shared fixtures. Every test file in the
same directory (and subdirectories) can use fixtures defined here without importing
them.

Our `conftest.py` does three things:

```python
import sys
from pathlib import Path

# 1. Add the engine lib to Python's import path
sys.path.insert(0, str(Path(__file__).parent.parent / "package" / "lib"))

# 2. Tell pytest to skip the manual test file
collect_ignore = ["test_manifests.py"]

import pytest
import responses

# 3. Fixtures
@pytest.fixture(autouse=True)
def activate_responses():
    with responses.RequestsMock() as rsps:
        yield rsps

@pytest.fixture
def mock_config():
    return {
        "base_url": "https://mock-api.test",
        "api_key": "test-key-123",
        "username": "testuser",
        "password": "testpass",
    }

@pytest.fixture
def empty_checkpoint():
    return {}
```

**Why `sys.path.insert`?** The engine code lives at
`ucc/urc_app/package/lib/urc/`. Python doesn't know to look there by default.
This line adds the `lib/` directory to the import path so `from urc.engine import collect`
works.

**Why `collect_ignore`?** `test_manifests.py` is a CLI tool, not a pytest test.
Its functions have parameters like `name` and `yaml_path` that aren't fixtures.
Without this ignore, pytest tries to collect it and fails.

---

## 13. Fixtures Directory: Reusable Building Blocks {#13-fixtures}

The `fixtures/` directory has three files that provide reusable test data:

### mock_apis.py — API scenario builders

Functions that register `responses` callbacks for common patterns:

```python
register_paginated_api(rsps, base_url, total_records=100, page_size=10)
register_rate_limited_api(rsps, base_url, fail_count=3)
register_bulk_job_api(rsps, base_url, poll_count=2)
register_csv_api(rsps, base_url, csv_text="id,name\n1,Alice")
register_error_sequence(rsps, base_url, status_codes=[500, 500, 200])
```

Use these when you need a standard API pattern. Write your own callbacks when
you need custom behavior.

### sample_manifests.py — YAML manifests for testing

Six inline YAML strings covering the common patterns:

- `SIMPLE_GET_MANIFEST` — no auth, no pagination, no incremental
- `PAGINATED_OFFSET_MANIFEST` — offset-based pagination
- `BEARER_AUTH_MANIFEST` — Bearer token authentication
- `INCREMENTAL_MANIFEST` — DatetimeBasedCursor
- `SUBSTREAM_MANIFEST` — parent → child with SubstreamPartitionRouter
- `TRANSFORM_MANIFEST` — AddFields + RemoveFields + KeysToSnakeCase chain

### sample_data.py — records and checkpoints

```python
SAMPLE_RECORDS = [
    {"id": i, "name": f"Item {i}", "created_at": f"2024-01-{i:02d}T00:00:00Z", ...}
    for i in range(1, 11)
]

SAMPLE_CHECKPOINT = {
    "events": {
        "cursor_field": "updated_at",
        "cursor_value": "2024-02-05T12:00:00Z",
    }
}
```

---

## 14. Adding Your Own Tests {#14-adding-tests}

### New component? Write a unit test.

1. Create `tests/unit/test_<component>.py`
2. Import the component class directly
3. Create it with a definition dict and config
4. Call its method with test input
5. Assert the output

```python
from urc.components.my_new_thing import MyNewThing

def test_my_new_thing_basic():
    t = MyNewThing({"option": "value"}, {})
    result = t.do_something({"input": "data"})
    assert result == {"expected": "output"}
```

### New API pattern? Write an integration test.

1. Add a mock scenario to `fixtures/mock_apis.py` (or use inline callbacks)
2. Add a sample manifest to `fixtures/sample_manifests.py`
3. Create `tests/integration/test_<scenario>.py`
4. Mock the API, call `collect()`, assert records

### New live API? Add a live test.

1. Add a manifest YAML to `tests/manifests/`
2. Add a config entry to `get_live_configs()` in `test_manifests.py`
3. Run with `python test_manifests.py --live <name>`

### Running tests

```bash
# All automated tests
python -m pytest ucc/urc_app/tests/ -v

# Just one file
python -m pytest ucc/urc_app/tests/unit/test_auth.py -v

# Just one test
python -m pytest ucc/urc_app/tests/unit/test_auth.py::test_bearer_auth_sets_header -v

# With print output visible
python -m pytest ucc/urc_app/tests/ -v -s
```
