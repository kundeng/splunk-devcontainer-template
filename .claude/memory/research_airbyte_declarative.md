# Research: Airbyte Declarative Low-Code Connector Framework

**Date**: 2026-03-27
**Purpose**: Inform the design of a Splunk UCC-based generic REST API connector app

---

## 1. Manifest Schema & Top-Level Structure

Airbyte declarative connectors are defined entirely in a `manifest.yaml` file. The declarative CDK interprets this YAML at runtime -- no Python code required for standard REST API patterns.

### Top-Level Properties

```yaml
version: "6.13.0"                    # Airbyte CDK version
type: DeclarativeSource              # Always DeclarativeSource

definitions:                         # Reusable component definitions (anchors)
  base_requester:
    type: HttpRequester
    url_base: "https://api.example.com"
    authenticator: ...

streams:                             # List of data streams
  - type: DeclarativeStream
    name: "users"
    primary_key: "id"
    retriever:
      type: SimpleRetriever
      requester:
        $ref: "#/definitions/base_requester"
        path: "/v1/users"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: ["data"]
      paginator: ...
    incremental_sync: ...
    schema_loader:
      type: InlineSchemaLoader
      schema:
        type: object
        properties:
          id: { type: integer }
          name: { type: string }

dynamic_streams: []                  # Streams discovered at runtime
concurrency_level:                   # Max concurrent async jobs
  type: ConcurrencyLevel
  default_concurrency: 10
  max_concurrency: 20
api_budget:                          # Rate limiting budget
  type: HTTPAPIBudget
  ...
max_concurrent_async_job_count: 5    # For report-based APIs

check:                               # Connection health check
  type: CheckStream
  stream_names: ["users"]

spec:                                # User-facing configuration schema
  type: Spec
  connection_specification:
    type: object
    required: ["api_key"]
    properties:
      api_key:
        type: string
        airbyte_secret: true
  advanced_auth: ...
```

### Architecture

- **DeclarativeStream**: Defines a data stream with retriever, schema, and optional incremental sync
- **SimpleRetriever**: Orchestrates requester, record_selector, paginator, and partition_router
- **HttpRequester**: Makes HTTP requests with auth, headers, params, body
- **RecordSelector**: Extracts records from responses
- **DefaultPaginator**: Handles multi-page responses
- **DatetimeBasedCursor**: Handles incremental/stateful collection

### `definitions` Section (DRY Pattern)

Components defined under `definitions` are referenced elsewhere via `$ref`:
```yaml
definitions:
  base_requester:
    type: HttpRequester
    url_base: "https://api.example.com/v2"
    authenticator:
      type: BearerAuthenticator
      api_token: "{{ config['api_key'] }}"

streams:
  - type: DeclarativeStream
    retriever:
      requester:
        $ref: "#/definitions/base_requester"
        path: "/users"
```

---

## 2. Authentication Types

All authentication is configured on the `HttpRequester.authenticator` property and shared across all streams in a connector.

### ApiKeyAuthenticator
Injects an API key into any request location (header, query param, or body).

```yaml
authenticator:
  type: ApiKeyAuthenticator
  api_token: "{{ config['api_key'] }}"
  inject_into:
    type: RequestOption
    inject_into: request_parameter    # or: header, path
    field_name: "x-api-key"
```

Legacy shorthand (header only):
```yaml
authenticator:
  type: ApiKeyAuthenticator
  header: "X-API-Key"
  api_token: "{{ config['api_key'] }}"
```

### BearerAuthenticator
Specialized ApiKeyAuthenticator that always sets `Authorization: Bearer {token}`.

```yaml
authenticator:
  type: BearerAuthenticator
  api_token: "{{ config['api_token'] }}"
```

### BasicHttpAuthenticator
Sets `Authorization: Basic base64(username:password)` per RFC 7617.

```yaml
authenticator:
  type: BasicHttpAuthenticator
  username: "{{ config['username'] }}"
  password: "{{ config['password'] }}"   # optional
```

### OAuthAuthenticator
Handles OAuth 2.0 token refresh (both `refresh_token` and `client_credentials` grants).

```yaml
authenticator:
  type: OAuthAuthenticator
  token_refresh_endpoint: "https://api.example.com/oauth/token"
  client_id: "{{ config['client_id'] }}"
  client_secret: "{{ config['client_secret'] }}"
  refresh_token: "{{ config['refresh_token'] }}"
  grant_type: "refresh_token"          # or "client_credentials"
  access_token_name: "access_token"    # field name in response
  expires_in_name: "expires_in"        # field name in response
  scopes:
    - "read"
    - "write"
  refresh_request_body:                # extra body params
    audience: "https://api.example.com"
  refresh_token_updater:               # for single-use refresh tokens
    refresh_token_name: "refresh_token"
    refresh_token_config_path: ["credentials", "refresh_token"]
    access_token_config_path: ["credentials", "access_token"]
```

### JwtAuthenticator
Generates JWT tokens automatically from a secret key.

```yaml
authenticator:
  type: JwtAuthenticator
  secret_key: "{{ config['secret_key'] }}"
  algorithm: "RS256"                   # HS256, RS256, ES256
  token_duration: 3600                 # seconds
  header_prefix: "Bearer"             # optional
  jwt_payload:                         # custom claims
    iss: "{{ config['issuer'] }}"
    sub: "{{ config['subject'] }}"
```

### SessionTokenAuthenticator
Performs a login request to obtain a session token, then uses it for subsequent requests.

```yaml
authenticator:
  type: SessionTokenAuthenticator
  login_requester:
    type: HttpRequester
    url_base: "https://api.example.com"
    path: "/auth/login"
    http_method: POST
    request_body_json:
      username: "{{ config['username'] }}"
      password: "{{ config['password'] }}"
  session_token_path: ["token"]        # DPath into login response
  expiration_duration: "PT1H"          # ISO 8601 duration
  request_authentication:
    type: SessionTokenRequestApiKeyAuthenticator
    inject_into:
      type: RequestOption
      inject_into: header
      field_name: "X-Session-Token"
```

### SelectiveAuthenticator
Chooses between multiple auth strategies based on user configuration.

```yaml
authenticator:
  type: SelectiveAuthenticator
  authenticator_selection_path: ["credentials", "auth_type"]
  authenticators:
    api_key:
      type: ApiKeyAuthenticator
      api_token: "{{ config['credentials']['api_key'] }}"
      header: "X-API-Key"
    oauth:
      type: OAuthAuthenticator
      client_id: "{{ config['credentials']['client_id'] }}"
      ...
```

### NoAuth
Implicit -- no authenticator property needed.

### Summary Table

| Type | Use Case | Key Properties |
|------|----------|---------------|
| ApiKeyAuthenticator | API key in header/param/body | api_token, inject_into, field_name |
| BearerAuthenticator | Bearer token auth | api_token |
| BasicHttpAuthenticator | Username/password basic auth | username, password |
| OAuthAuthenticator | OAuth 2.0 with token refresh | client_id/secret, refresh_token, grant_type |
| JwtAuthenticator | JWT signing | secret_key, algorithm, token_duration |
| SessionTokenAuthenticator | Login-based sessions | login_requester, session_token_path |
| SelectiveAuthenticator | User chooses auth method | authenticator_selection_path, authenticators |

---

## 3. Pagination Strategies

Pagination is configured on the `SimpleRetriever.paginator` property using `DefaultPaginator`.

### DefaultPaginator Structure

```yaml
paginator:
  type: DefaultPaginator
  page_size_option:                    # How to tell API the page size
    type: RequestOption
    inject_into: request_parameter
    field_name: "limit"
  page_token_option:                   # How to tell API which page
    type: RequestOption                # or RequestPath
    inject_into: request_parameter
    field_name: "offset"
  pagination_strategy:                 # One of the strategies below
    type: OffsetIncrement
    page_size: 100
```

### PageIncrement
Increments a page number with each request. Stops when a page has fewer records than `page_size`.

```yaml
paginator:
  type: DefaultPaginator
  page_size_option:
    type: RequestOption
    inject_into: request_parameter
    field_name: "page_size"
  pagination_strategy:
    type: PageIncrement
    page_size: 50
    start_from_page: 0                 # optional, default 0
  page_token_option:
    type: RequestOption
    inject_into: request_parameter
    field_name: "page"
```

Produces: `?page_size=50&page=0`, `?page_size=50&page=1`, ...

### OffsetIncrement
Increments offset by page_size. Stops when a page has fewer records than `page_size`.

```yaml
paginator:
  type: DefaultPaginator
  page_size_option:
    type: RequestOption
    inject_into: request_parameter
    field_name: "page_size"
  pagination_strategy:
    type: OffsetIncrement
    page_size: 100
    inject_on_first_request: true      # optional
  page_token_option:
    type: RequestOption
    inject_into: request_parameter
    field_name: "offset"
```

Produces: `?page_size=100&offset=0`, `?page_size=100&offset=100`, ...

### CursorPagination
Extracts a next-page token from the response (headers, body, or last record). Most flexible.

**Cursor from response body (next_url pattern):**
```yaml
paginator:
  type: DefaultPaginator
  pagination_strategy:
    type: CursorPagination
    cursor_value: "{{ response['paging']['next'] }}"
    stop_condition: "{{ response['paging']['next'] is none }}"
  page_token_option:
    type: RequestPath                  # Follow the URL directly
```

**Cursor from last record (keyset pagination):**
```yaml
paginator:
  type: DefaultPaginator
  pagination_strategy:
    type: CursorPagination
    cursor_value: "{{ last_records[-1]['id'] }}"
    page_size: 100
  page_token_option:
    type: RequestOption
    inject_into: request_parameter
    field_name: "after"
```

**Cursor from response headers:**
```yaml
paginator:
  type: DefaultPaginator
  pagination_strategy:
    type: CursorPagination
    cursor_value: "{{ headers['link']['next']['url'] }}"
  page_token_option:
    type: RequestPath
```

### NoPagination (implicit)
Omit the `paginator` property entirely if the API returns all records in one response.

### Summary Table

| Strategy | When to Use | Stop Condition |
|----------|-------------|----------------|
| PageIncrement | `?page=1&page=2` style | Fewer records than page_size |
| OffsetIncrement | `?offset=0&offset=100` style | Fewer records than page_size |
| CursorPagination | Next-page token/URL in response | Configurable stop_condition or null token |

---

## 4. Record Extraction / Response Parsing

Records are extracted from API responses using `RecordSelector` + `DpathExtractor`.

### DpathExtractor

Uses the Python `dpath` library (not JSONPath, not jq) to navigate JSON responses.

```yaml
record_selector:
  type: RecordSelector
  extractor:
    type: DpathExtractor
    field_path: ["data", "results"]     # response["data"]["results"]
```

**Common patterns:**

```yaml
# Response: {"results": [{...}, {...}]}
field_path: ["results"]

# Response: [{...}, {...}]  (root-level array)
field_path: []

# Response: {"data": {"items": [{...}]}}
field_path: ["data", "items"]

# Response: {"level1": {"*": {"records": [...]}}}  (wildcard)
field_path: ["level1", "*", "records"]

# Templated path
field_path: ["{{ parameters['data_field'] }}"]
```

### Record Filtering

Optional filter applied after extraction:

```yaml
record_selector:
  type: RecordSelector
  extractor:
    type: DpathExtractor
    field_path: ["results"]
  record_filter:
    type: RecordFilter
    condition: "{{ record['status'] == 'active' }}"
```

Filter condition has access to: `record`, `stream_interval`, `stream_slice`, `config`.

### Record Transformations

Applied after selection, before output:

```yaml
transformations:
  - type: AddFields
    fields:
      - path: ["source"]
        value: "api_v2"
      - path: ["extracted_at"]
        value: "{{ now_utc() }}"
  - type: RemoveFields
    field_pointers:
      - ["internal_id"]
      - ["metadata", "debug"]
```

### ResponseToFileExtractor
For CSV/file responses (no DPath):
```yaml
record_selector:
  type: RecordSelector
  extractor:
    type: ResponseToFileExtractor
```

---

## 5. Incremental Sync / Checkpointing

### DatetimeBasedCursor

The primary mechanism for incremental data collection. Partitions a time range into windows and tracks the cursor position.

```yaml
incremental_sync:
  type: DatetimeBasedCursor
  cursor_field: "updated_at"                # Field on each record
  cursor_datetime_formats:                   # Formats to PARSE from records
    - "%Y-%m-%dT%H:%M:%S.%fZ"
    - "%Y-%m-%dT%H:%M:%SZ"
  datetime_format: "%Y-%m-%dT%H:%M:%SZ"     # Format to SEND to API
  start_datetime: "2023-01-01T00:00:00Z"     # Or: "{{ config['start_date'] }}"
  end_datetime: "{{ now_utc() }}"            # Or omit for "now"
  step: "P30D"                               # ISO 8601 duration per window
  cursor_granularity: "PT1S"                 # Minimum time between windows
  lookback_window: "P3D"                     # Re-read 3 days before cursor
  start_time_option:                         # Inject start into request
    type: RequestOption
    field_name: "updated_after"
    inject_into: request_parameter
  end_time_option:                           # Inject end into request
    type: RequestOption
    field_name: "updated_before"
    inject_into: request_parameter
```

### Key Properties

| Property | Purpose |
|----------|---------|
| `cursor_field` | Record field used as bookmark (must be top-level) |
| `cursor_datetime_formats` | Accepted input datetime formats from records |
| `datetime_format` | Output format sent to API |
| `start_datetime` | Initial sync start (string or `MinMaxDatetime`) |
| `end_datetime` | Sync end (default: now) |
| `step` | Time window size (ISO 8601 duration: P1D, PT1H, P30D) |
| `cursor_granularity` | Precision between windows (PT1S, PT0.001S) |
| `lookback_window` | Re-read period before cursor to catch late-arriving data |
| `start_time_option` | RequestOption to inject window start |
| `end_time_option` | RequestOption to inject window end |
| `is_data_feed` | true = API returns ALL records, filter client-side |
| `is_client_side_incremental` | true = filter records locally by cursor |
| `global_substream_cursor` | true = single cursor for all partitions |

### Checkpointing Behavior

1. On first sync: reads from `start_datetime` to `end_datetime`
2. Emits state message after reading all records with cursor = last record's datetime
3. On next sync: reads from stored cursor value (minus lookback_window if set)
4. Records must be returned in ascending order by cursor_field

### IncrementingCountCursor
For APIs that use a sequential ID instead of datetime:

```yaml
incremental_sync:
  type: IncrementingCountCursor
  cursor_field: "id"
  start_value: 0
  start_value_option:
    type: RequestOption
    field_name: "since_id"
    inject_into: request_parameter
```

### MinMaxDatetime
Constrains start/end to a range:

```yaml
start_datetime:
  type: MinMaxDatetime
  datetime: "{{ config['start_date'] }}"
  min_datetime: "2020-01-01T00:00:00Z"
```

---

## 6. Request Construction & Templating

### HttpRequester Properties

```yaml
requester:
  type: HttpRequester
  url: "https://api.example.com/v2/{{ config['resource'] }}"   # Full URL (new)
  url_base: "https://api.example.com/v2"                        # Base (legacy)
  path: "/users/{{ stream_slice.user_id }}/posts"               # Path (legacy)
  http_method: "GET"                   # or "POST"
  authenticator: ...
  request_parameters:                  # Query string params
    status: "active"
    since: "{{ stream_interval.start_time }}"
    page_size: "{{ config['page_size'] or '100' }}"
  request_headers:                     # HTTP headers
    Accept: "application/json"
    X-Custom: "{{ config['custom_header'] }}"
  request_body_json:                   # JSON body (for POST)
    query: "{{ config['query'] }}"
    filters:
      start: "{{ stream_interval.start_time }}"
      end: "{{ stream_interval.end_time }}"
  request_body_data:                   # Form-encoded body
    grant_type: "client_credentials"
  error_handler: ...
```

### Jinja2 Interpolation Engine

All string values support `{{ }}` Jinja2 templates. The engine provides these context variables:

| Variable | Available In | Description |
|----------|-------------|-------------|
| `config` | Everywhere | User's connector configuration |
| `parameters` | Everywhere | `$parameters` from definitions |
| `stream_slice` | Requester, paginator | Current partition values |
| `stream_interval` | Requester | `.start_time` and `.end_time` |
| `stream_partition` | Requester | Current partition key-value |
| `next_page_token` | Paginator | Token from pagination strategy |
| `response` | Record selector, error handler | HTTP response body (decoded) |
| `headers` | Pagination, error handler | HTTP response headers |
| `last_records` | CursorPagination | Array of records from last page |
| `record` | Record filter, transformations | Current record being processed |

### Built-in Macros/Functions

| Macro | Example | Description |
|-------|---------|-------------|
| `now_utc()` | `{{ now_utc() }}` | Current UTC datetime |
| `today_utc()` | `{{ today_utc() }}` | Current UTC date |
| `timestamp(n)` | `{{ timestamp(1672531200) }}` | Unix timestamp to datetime |
| `max(a, b)` | `{{ max(2, 3) }}` | Maximum of two values |
| `day_delta(n)` | `{{ day_delta(-7) }}` | Date offset from today |
| `format_datetime(dt, fmt)` | `{{ format_datetime(config['date'], '%Y-%m-%d') }}` | Reformat datetime |
| `duration(s)` | `{{ duration('P1D') }}` | Parse ISO 8601 duration |

### Jinja2 Filters
- `{{ value | string }}` -- cast to string
- `{{ value | int }}` -- cast to integer
- `{{ value | length }}` -- length
- `{{ value | urlencode }}` -- URL-encode
- `{{ value | b64encode }}` -- Base64 encode

### Conditional Expressions
```yaml
request_parameters:
  include_deleted: "{{ config.get('include_deleted', false) }}"
  format: "{{ 'json' if config['format'] == 'json' else 'xml' }}"
```

---

## 7. Error Handling / Retry

### Default Behavior (no configuration needed)
- Retries HTTP 429 (rate limited) and 5XX (server errors)
- Up to 5 retries with exponential backoff (factor=5: 5s, 10s, 20s, 40s, 80s)
- All other HTTP errors cause immediate failure

### DefaultErrorHandler

```yaml
requester:
  error_handler:
    type: DefaultErrorHandler
    max_retries: 10
    backoff_strategies:
      - type: ExponentialBackoffStrategy
        factor: 5
    response_filters:
      - type: HttpResponseFilter
        action: RETRY
        http_codes: [429, 500, 502, 503]
      - type: HttpResponseFilter
        action: IGNORE
        http_codes: [404]
        error_message: "Resource not found, skipping"
      - type: HttpResponseFilter
        action: FAIL
        http_codes: [401, 403]
        failure_type: config_error
```

### HttpResponseFilter Actions

| Action | Behavior |
|--------|----------|
| `SUCCESS` | Treat as successful response |
| `RETRY` | Retry with backoff |
| `FAIL` | Stop and raise error |
| `IGNORE` | Skip silently, continue |
| `RATE_LIMITED` | Apply rate-limit backoff |
| `RESET_PAGINATION` | Reset paginator and retry |
| `REFRESH_TOKEN_THEN_RETRY` | Refresh OAuth token, then retry |

### Response Filter Matching

```yaml
# By HTTP status code
- http_codes: [429, 503]
  action: RETRY

# By error message substring
- error_message_contains: "rate limit exceeded"
  action: RETRY

# By Jinja2 predicate on response body
- predicate: "{{ 'temporarily_unavailable' in response.get('error', '') }}"
  action: RETRY

# Failure type classification
- http_codes: [401]
  action: FAIL
  failure_type: config_error          # or: system_error, transient_error
```

### Backoff Strategies

```yaml
# Exponential: factor * 2^attempt (5s, 10s, 20s, 40s, 80s)
- type: ExponentialBackoffStrategy
  factor: 5                            # seconds (or string template)

# Constant: fixed wait between retries
- type: ConstantBackoffStrategy
  backoff_time_in_seconds: 30

# From response header (e.g., Retry-After: 60)
- type: WaitTimeFromHeader
  header: "Retry-After"
  regex: "\\d+"                        # optional regex to extract

# Wait until a specific time from header
- type: WaitUntilTimeFromHeader
  header: "X-Rate-Limit-Reset"
  min_wait: 1                          # minimum seconds to wait
```

### CompositeErrorHandler
Chain multiple handlers (evaluated in sequence):

```yaml
error_handler:
  type: CompositeErrorHandler
  error_handlers:
    - type: DefaultErrorHandler
      response_filters:
        - http_codes: [429]
          action: RETRY
      backoff_strategies:
        - type: WaitTimeFromHeader
          header: "Retry-After"
    - type: DefaultErrorHandler
      response_filters:
        - http_codes: [500, 502, 503]
          action: RETRY
      backoff_strategies:
        - type: ExponentialBackoffStrategy
          factor: 5
```

---

## 8. Stream Slicing / Partitioning

Stream slicing enables partitioned data collection -- fetching data for each partition separately.

### ListPartitionRouter
Iterates over a static or config-driven list of values.

```yaml
retriever:
  type: SimpleRetriever
  partition_router:
    type: ListPartitionRouter
    cursor_field: "repository"
    values:
      - "repo-a"
      - "repo-b"
      - "repo-c"
    # Or from config:
    # values: "{{ config['repositories'] }}"
    request_option:
      type: RequestOption
      field_name: "repo"
      inject_into: request_parameter
```

Access in templates: `{{ stream_slice.repository }}`

### SubstreamPartitionRouter
Creates parent-child relationships. For each record in the parent stream, fetches child records.

```yaml
definitions:
  organizations_stream:
    type: DeclarativeStream
    name: "organizations"
    retriever:
      requester:
        path: "/orgs"
      record_selector:
        extractor:
          type: DpathExtractor
          field_path: ["data"]

streams:
  - type: DeclarativeStream
    name: "repositories"
    retriever:
      type: SimpleRetriever
      requester:
        path: "/orgs/{{ stream_slice.org_id }}/repos"
      record_selector:
        extractor:
          type: DpathExtractor
          field_path: ["data"]
      partition_router:
        type: SubstreamPartitionRouter
        parent_stream_configs:
          - type: ParentStreamConfig
            stream:
              $ref: "#/definitions/organizations_stream"
            parent_key: "id"            # Field from parent record
            partition_field: "org_id"   # Name in stream_slice
            incremental_dependency: true
            extra_fields:               # Additional parent fields to carry
              - path: ["org_name"]
                value: "{{ record['name'] }}"
```

### Combining Multiple Routers (Cartesian Product)
Multiple routers create all permutations:

```yaml
partition_router:
  - type: ListPartitionRouter
    cursor_field: "status"
    values: ["open", "closed"]
  - type: ListPartitionRouter
    cursor_field: "priority"
    values: ["high", "medium", "low"]
```

This produces 6 partitions: (open, high), (open, medium), ..., (closed, low).

### Partition Router + Incremental Sync
Can be combined -- each partition gets its own cursor state:

```yaml
retriever:
  partition_router:
    type: SubstreamPartitionRouter
    parent_stream_configs:
      - stream: "#/definitions/projects_stream"
        parent_key: "id"
        partition_field: "project_id"
incremental_sync:
  type: DatetimeBasedCursor
  cursor_field: "updated_at"
  global_substream_cursor: false       # Per-partition state (default)
  # global_substream_cursor: true      # Single shared cursor (less state)
```

---

## 9. PyAirbyte

PyAirbyte is a Python library that lets you run Airbyte connectors (including declarative ones) directly in Python without the Airbyte platform.

### Installation
```bash
pip install airbyte
```

### Core API

```python
import airbyte as ab
from pathlib import Path

# Load a published connector
source = ab.get_source(
    "source-github",
    config={"access_token": "...", "repositories": ["org/repo"]},
    install_if_missing=True,
)

# Load a custom declarative manifest
source = ab.get_source(
    "source-custom-api",
    source_manifest=Path("manifest.yaml"),    # Path, dict, or URL
    config={"api_key": "...", "base_url": "https://api.example.com"},
)

# Or pass manifest as a dict
import yaml
with open("manifest.yaml") as f:
    manifest = yaml.safe_load(f)
source = ab.get_source(
    "source-custom-api",
    source_manifest=manifest,
    config={...},
)
```

### get_source() Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Connector identifier |
| `config` | dict | Connector configuration |
| `source_manifest` | bool/dict/Path/str | Declarative YAML definition |
| `streams` | str/list | Stream names to select ("*" for all) |
| `install_if_missing` | bool | Auto-install connector |
| `docker_image` | bool/str | Run in Docker |
| `pip_url` | str | Custom pip install URL |

### Reading Data

```python
# Verify connection
source.check()

# Select streams
source.select_all_streams()
# or: source.select_streams(["users", "orders"])

# Available streams
print(source.get_available_streams())

# Read with caching (DuckDB by default)
result = source.read()
for record in result["users"]:
    print(record)

# Read without caching (generator)
for record in source.get_records("users"):
    print(record)
```

### ReadResult API

```python
result = source.read()
result.streams                    # Available stream names
result["users"]                   # Access stream dataset
result["users"].to_pandas()       # Convert to DataFrame
result.processed_records          # Count of records
result.cache                      # Underlying cache object
```

### Cache Types
- **DuckDBCache** (default) -- local file-based
- **BigQueryCache** -- Google BigQuery
- **PostgresCache** -- PostgreSQL
- **SnowflakeCache** -- Snowflake
- **MotherDuckCache** -- MotherDuck cloud

### Relevance to Splunk UCC
PyAirbyte proves that a manifest-driven approach works for Python-based data collection. The `source_manifest` parameter shows how a YAML/JSON definition can be loaded and interpreted at runtime. This is directly analogous to what a Splunk UCC modular input could do: load a REST API definition from a configuration and execute it.

---

## 10. Connection Specification (spec)

The `spec` section defines the user-facing configuration form. It uses JSON Schema (draft-07) with Airbyte-specific extensions.

### Basic Structure

```yaml
spec:
  type: Spec
  connection_specification:
    $schema: "http://json-schema.org/draft-07/schema#"
    title: "My API Source"
    description: "Connector for My API"
    type: object
    required:
      - api_key
      - base_url
    properties:
      api_key:
        type: string
        title: "API Key"
        description: "Your API key"
        airbyte_secret: true
        order: 0
      base_url:
        type: string
        title: "Base URL"
        description: "API base URL"
        default: "https://api.example.com/v2"
        order: 1
      start_date:
        type: string
        title: "Start Date"
        description: "UTC date in YYYY-MM-DD format"
        pattern: "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
        pattern_descriptor: "YYYY-MM-DD"
        order: 2
```

### Airbyte-Specific Extensions

| Extension | Type | Purpose |
|-----------|------|---------|
| `airbyte_secret` | boolean | Mask field in UI and API |
| `order` | integer | Control field display order |
| `group` | string | Group fields into cards |
| `always_show` | boolean | Prevent collapsing optional fields |
| `airbyte_hidden` | boolean | Hide from UI, allow API config |
| `multiline` | boolean | Multi-line text input |
| `pattern_descriptor` | string | Human-readable format hint |
| `display_type` | string | "dropdown" or "radio" for oneOf |

### Supported Field Types

```yaml
# String
api_key:
  type: string

# Integer
page_size:
  type: integer
  default: 100
  minimum: 1
  maximum: 1000

# Boolean
include_deleted:
  type: boolean
  default: false

# Enum (dropdown)
environment:
  type: string
  enum: ["production", "sandbox"]
  default: "production"

# Array
tags:
  type: array
  items:
    type: string

# Object (nested)
advanced:
  type: object
  properties:
    timeout:
      type: integer
```

### oneOf Conditions (Conditional Fields)

Used for authentication method selection and similar patterns:

```yaml
credentials:
  type: object
  title: "Authentication"
  display_type: "dropdown"           # or "radio"
  oneOf:
    - title: "API Key"
      required: ["auth_type", "api_key"]
      properties:
        auth_type:
          type: string
          const: "api_key"
          order: 0
        api_key:
          type: string
          title: "API Key"
          airbyte_secret: true
          order: 1
    - title: "OAuth 2.0"
      required: ["auth_type", "client_id", "client_secret"]
      properties:
        auth_type:
          type: string
          const: "oauth2"
          order: 0
        client_id:
          type: string
          title: "Client ID"
          order: 1
        client_secret:
          type: string
          title: "Client Secret"
          airbyte_secret: true
          order: 2
    - title: "Basic Auth"
      required: ["auth_type", "username", "password"]
      properties:
        auth_type:
          type: string
          const: "basic"
          order: 0
        username:
          type: string
          order: 1
        password:
          type: string
          airbyte_secret: true
          order: 2
```

**oneOf Rules:**
1. Each item must be `type: object`
2. A discriminator field with `const` must exist in all items (same property name)
3. The `const` value must be unique per option
4. Forbidden JSON Schema keywords: `not`, `anyOf`, `patternProperties`, `allOf`, `if/then/else`

### Grouped Fields

```yaml
connection_specification:
  type: object
  groups:
    - id: "auth"
      title: "Authentication"
    - id: "connection"
      title: "Connection Settings"
  properties:
    api_key:
      type: string
      group: "auth"
      order: 0
    base_url:
      type: string
      group: "connection"
      order: 0
```

### OAuth Flow in Spec

```yaml
spec:
  connection_specification:
    type: object
    properties:
      client_id:
        type: string
        airbyte_secret: true
      client_secret:
        type: string
        airbyte_secret: true
      access_token:
        type: string
        airbyte_secret: true
  advanced_auth:
    auth_flow_type: "oauth2.0"
    oauth_config_specification:
      oauth_connector_input_specification:
        consent_url: "https://provider.com/oauth/authorize?client_id={{client_id_value}}&redirect_uri={{redirect_uri_value}}&scope={{scope_value}}&state={{state_value}}"
        access_token_url: "https://provider.com/oauth/token"
        scope: "read write"
        access_token_headers:
          Content-Type: "application/x-www-form-urlencoded"
        access_token_params:
          grant_type: "authorization_code"
      complete_oauth_output_specification:
        properties:
          access_token:
            type: string
            path_in_connector_config: ["access_token"]
      complete_oauth_server_input_specification:
        properties:
          client_id:
            type: string
          client_secret:
            type: string
      complete_oauth_server_output_specification:
        properties:
          client_id:
            type: string
            path_in_connector_config: ["client_id"]
          client_secret:
            type: string
            path_in_connector_config: ["client_secret"]
```

### OAuth Template Variables
- `{{client_id_value}}` -- Client ID
- `{{client_secret_value}}` -- Client secret
- `{{redirect_uri_value}}` -- Redirect URI
- `{{state_value}}` -- CSRF state
- `{{auth_code_value}}` -- Authorization code
- `{{scope_value}}` -- Requested scopes

### OAuth Template Filters
- `{{ value | urlencode }}` -- URL encoding
- `{{ value | b64encode }}` -- Base64 encoding
- `{{ value | codechallengeS256 }}` -- PKCE S256 challenge

---

## Design Pattern Takeaways for Splunk UCC

### What Airbyte Gets Right

1. **Declarative-first**: ~80% of their connectors are YAML-only, proving the pattern works at scale
2. **Composable components**: Auth, pagination, record extraction, error handling are independent, pluggable pieces
3. **Jinja2 templating**: Enables dynamic config without code -- `{{ config['key'] }}` patterns map well to Splunk UCC's `{{field_name}}` patterns
4. **oneOf for auth selection**: Their spec uses JSON Schema `oneOf` with `const` discriminators -- identical to how UCC `globalConfig.json` uses conditional display
5. **DPath for record extraction**: Simple dotted-path extraction is more approachable than JSONPath or jq
6. **Cursor-based checkpointing**: DatetimeBasedCursor with lookback_window is robust for incremental collection
7. **Partition routing**: SubstreamPartitionRouter for parent-child APIs and ListPartitionRouter for multi-tenant collection

### Mapping to Splunk UCC globalConfig.json

| Airbyte Concept | UCC Equivalent |
|-----------------|----------------|
| `spec.connection_specification` | `globalConfig.json` pages/tabs |
| `oneOf` with `const` discriminator | UCC entity `options` with `autoCompleteFields` |
| `airbyte_secret` | `encrypted: true` |
| `order` | `field` ordering in entity array |
| `group` | UCC tabs/groups |
| `streams[].name` | Input types / data input names |
| `authenticator` | Account configuration in UCC |
| `paginator` | Custom pagination logic in input helper |
| `record_selector` | Response handler in modular input |
| `incremental_sync` | Checkpoint/state via `KVStoreCheckpointer` |
| `partition_router` | Multiple input instances or parameterized collection |

### Key Differences to Account For

1. **Splunk runs inputs continuously** (polling interval) vs. Airbyte's batch sync model
2. **Splunk needs HEC/index output** vs. Airbyte's destination abstraction
3. **Splunk UCC has no built-in Jinja2** -- need custom interpolation or use Python string formatting
4. **State management**: Splunk uses KV store checkpointing vs. Airbyte's state messages
5. **No built-in DPath in Splunk** -- need to include dpath library or use custom extraction

---

## Sources

- [YAML Reference](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/reference)
- [Understanding the YAML File](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/yaml-overview)
- [Low-code CDK Overview](https://docs.airbyte.com/platform/connector-development/config-based/low-code-cdk-overview)
- [Authentication](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/authentication)
- [Pagination](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/pagination)
- [Record Selector](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/record-selector)
- [Incremental Syncs](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/incremental-syncs)
- [String Interpolation](https://docs.airbyte.com/platform/connector-development/config-based/advanced-topics/string-interpolation)
- [Error Handling](https://docs.airbyte.com/platform/connector-development/connector-builder-ui/error-handling)
- [Partition Router](https://docs.airbyte.com/platform/connector-development/config-based/understanding-the-yaml-file/partition-router)
- [OAuth Authentication](https://docs.airbyte.com/platform/connector-development/config-based/advanced-topics/oauth)
- [Connector Specification Reference](https://docs.airbyte.com/platform/connector-development/connector-specification-reference)
- [PyAirbyte](https://docs.airbyte.com/developers/pyairbyte)
- [PyAirbyte API Reference](https://airbytehq.github.io/PyAirbyte/airbyte.html)
- [Declarative API Connectors (DeepWiki)](https://deepwiki.com/airbytehq/airbyte/3.2-declarative-api-connectors)
