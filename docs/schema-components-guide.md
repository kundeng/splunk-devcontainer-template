# Airbyte Declarative Schema — Component Guide for URC

A practical reference for every component type available in the URC builder. Each component is auto-generated from Airbyte's `declarative_component_schema.yaml` and rendered as a schema-driven form in the UI.

**Audience:** Developers configuring REST API streams via YAML manifests or the builder UI.
**Last updated:** 2026-04-02

---

## How It Works

The URC engine uses Airbyte's declarative connector framework. Instead of writing Python code for each API, you write a YAML manifest that describes _how_ to talk to the API. The engine reads this manifest at runtime and does the rest.

The builder UI renders forms from this same schema — every dropdown, field, and option maps directly to a manifest YAML property. When you save a stream, the form state is serialized into YAML and stored in Splunk's `inputs.conf`.

```
┌──────────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Airbyte YAML Schema │────▶│  form-schema.ts   │────▶│  Builder UI      │
│  (declarative_       │     │  (auto-generated) │     │  (schema-driven  │
│   component_schema)  │     │  53 component     │     │   forms)         │
│                      │     │  types, 11 cats   │     │                  │
└──────────────────────┘     └───────────────────┘     └──────────────────┘
         │                                                       │
         │                                                       ▼
         │                                              ┌──────────────────┐
         └─────────────────────────────────────────────▶│  YAML manifest   │
                                                        │  (inputs.conf)   │
                                                        └──────────────────┘
                                                                 │
                                                                 ▼
                                                        ┌──────────────────┐
                                                        │  URC Engine      │
                                                        │  (Python runtime)│
                                                        └──────────────────┘
```

---

## 1. Record Selectors (Extractors)

These define how individual records are extracted from the API response body.

### DpathExtractor
**The most common choice.** Uses dot-path notation to navigate into the JSON response and extract an array of records.

```yaml
# API returns: { "data": { "items": [ {record1}, {record2}, ... ] } }
extractor:
  type: DpathExtractor
  field_path:
    - data
    - items
```

**Examples:**
| API Response Shape | field_path |
|---|---|
| `[ {record}, ... ]` (top-level array) | `[]` (empty — the root IS the array) |
| `{ "results": [...] }` | `["results"]` |
| `{ "data": { "users": [...] } }` | `["data", "users"]` |
| `{ "response": { "body": { "records": [...] } } }` | `["response", "body", "records"]` |

### RecordFilter
Filters records based on a condition. Use after extraction to remove unwanted records.

```yaml
extractor:
  type: RecordFilter
  condition: "{{ record['status'] == 'active' }}"
```

### RecordSelector
Wraps an extractor with optional filtering and schema normalization.

```yaml
recordSelector:
  type: RecordSelector
  extractor:
    type: DpathExtractor
    field_path: ["results"]
  record_filter:
    condition: "{{ record['deleted'] != true }}"
```

---

## 2. Paginators

How to retrieve multiple pages of results from paginated APIs.

### NoPagination
API returns all results in a single response. No pagination needed.

```yaml
paginator:
  type: NoPagination
```

### CursorPagination
The most flexible. Follows a cursor/link in each response to get the next page.

```yaml
# GitHub-style: response has a "Link" header with next page URL
paginator:
  type: CursorPagination
  cursor_value: "{{ headers['link']['next'] }}"
  stop_condition: "{{ not headers.get('link', {}).get('next') }}"
```

```yaml
# API returns: { "next_cursor": "abc123", "data": [...] }
paginator:
  type: CursorPagination
  cursor_value: "{{ response['next_cursor'] }}"
  stop_condition: "{{ not response.get('next_cursor') }}"
```

### OffsetIncrement
Classic offset-based pagination. Increments offset by page size.

```yaml
# GET /api/users?offset=0&limit=100, then offset=100, offset=200...
paginator:
  type: OffsetIncrement
  page_size: 100
```

### PageIncrement
Page-number based. Increments page number by 1.

```yaml
# GET /api/users?page=1, then page=2, page=3...
paginator:
  type: PageIncrement
  page_size: 50
  start_from_page: 1
```

### DefaultPaginator
Full-featured paginator with explicit page size, strategy, and injection configuration. Use when you need fine-grained control over how pagination parameters are sent.

```yaml
paginator:
  type: DefaultPaginator
  page_size_option:
    inject_into: request_parameter
    field_name: per_page
  pagination_strategy:
    type: OffsetIncrement
    page_size: 100
  page_token_option:
    inject_into: request_parameter
    field_name: offset
```

---

## 3. Incremental Sync (Cursors)

Track where you left off so subsequent runs only fetch new/updated records.

### DatetimeBasedCursor
The most common. Tracks a datetime field and only fetches records newer than the last sync.

```yaml
# ServiceNow: track sys_updated_on, send as sysparm_query
incrementalSync:
  type: DatetimeBasedCursor
  cursor_field: sys_updated_on
  datetime_format: "%Y-%m-%d %H:%M:%S"
  start_datetime: "2024-01-01T00:00:00Z"
  cursor_granularity: PT1S
  step: P30D
```

```yaml
# GitHub: track updated_at, send as query parameter
incrementalSync:
  type: DatetimeBasedCursor
  cursor_field: updated_at
  datetime_format: "%Y-%m-%dT%H:%M:%SZ"
  start_datetime: "{{ config.get('start_date', '2024-01-01T00:00:00Z') }}"
```

### IncrementingCountCursor
For APIs that use a numeric ID or sequence number instead of timestamps.

```yaml
incrementalSync:
  type: IncrementingCountCursor
  cursor_field: id
  start_value: 0
```

---

## 4. Partition Routers (Substreams)

Split collection into multiple partitions. This is how you implement parent-child relationships.

### SubstreamPartitionRouter
**The key pattern for related data.** First collect a parent stream, then iterate over each parent record to collect child records.

```yaml
# Example: GitHub repos → issues per repo
# Parent stream collects repos, child stream collects issues for each repo
partition_router:
  type: SubstreamPartitionRouter
  parent_stream_configs:
    - type: ParentStreamConfig
      parent_key: full_name          # field from parent record
      partition_field: repository    # name in the child's URL template
      stream:
        type: DeclarativeStream
        name: repos
        retriever:
          type: SimpleRetriever
          requester:
            type: HttpRequester
            url: "https://api.github.com/user/repos"
            http_method: GET

# Child stream URL uses the partition field:
# url: "https://api.github.com/repos/{{ stream_partition['repository'] }}/issues"
```

**Real-world substream patterns:**
| Parent | Child | parent_key | partition_field |
|---|---|---|---|
| `/repos` | `/repos/{repo}/issues` | `full_name` | `repository` |
| `/projects` | `/projects/{id}/tasks` | `id` | `project_id` |
| `/accounts` | `/accounts/{id}/contacts` | `id` | `account_id` |
| `/sites` | `/sites/{name}/devices` | `name` | `site_name` |

### ListPartitionRouter
Iterate over a static or config-driven list of values.

```yaml
# Collect from multiple regions
partition_router:
  type: ListPartitionRouter
  values: ["us-east-1", "us-west-2", "eu-west-1"]
  cursor_field: region
  # URL becomes: /api/metrics?region={{ stream_partition['region'] }}
```

```yaml
# Values from config
partition_router:
  type: ListPartitionRouter
  values: "{{ config['environments'] }}"
  cursor_field: env
```

### GroupingPartitionRouter
Groups partition values into batches for APIs that accept multiple IDs per request.

---

## 5. Transformations

Modify records after extraction, before indexing into Splunk.

### AddFields
Add computed or static fields to each record.

```yaml
transformations:
  - type: AddFields
    fields:
      - path: ["_collection_time"]
        value: "{{ now_utc() }}"
      - path: ["_source_api"]
        value: "github"
```

### RemoveFields
Remove unwanted fields (PII, large blobs, internal IDs).

```yaml
transformations:
  - type: RemoveFields
    field_pointers:
      - ["password_hash"]
      - ["internal_notes"]
      - ["raw_body"]
```

### KeysToSnakeCase
Convert camelCase keys to snake_case (common when indexing into Splunk).

```yaml
transformations:
  - type: KeysToSnakeCase
```

### KeysToLower
Lowercase all field names.

### KeysReplace
Regex-based key renaming.

```yaml
transformations:
  - type: KeysReplace
    old: "^sys_"
    new: "servicenow_"
```

### FlattenFields / DpathFlattenFields
Flatten nested objects into dot-notation keys.

```yaml
# { "user": { "name": "Alice", "email": "a@b.com" } }
# becomes: { "user.name": "Alice", "user.email": "a@b.com" }
transformations:
  - type: FlattenFields
```

---

## 6. Error Handlers

Configure retry behavior when API calls fail.

### DefaultErrorHandler
Retries on 429 (rate limit) and 5xx errors with exponential backoff.

```yaml
error_handler:
  type: DefaultErrorHandler
  max_retries: 5
  backoff_strategies:
    - type: ExponentialBackoffStrategy
      factor: 2
```

### ConstantBackoffStrategy
Wait a fixed number of seconds between retries.

```yaml
backoff_strategies:
  - type: ConstantBackoffStrategy
    backoff_time_in_seconds: 30
```

### WaitTimeFromHeader
Read the retry-after time from a response header (common with rate-limited APIs).

```yaml
backoff_strategies:
  - type: WaitTimeFromHeader
    header: Retry-After
```

### HttpResponseFilter
Match specific error responses and decide whether to retry, fail, or ignore.

```yaml
response_filters:
  - type: HttpResponseFilter
    action: RETRY
    http_codes: [429, 503]
  - type: HttpResponseFilter
    action: IGNORE
    http_codes: [404]
```

---

## 7. Decoders

How to decode the raw API response body. Default is JSON.

### JsonDecoder / JsonlDecoder
Standard JSON or newline-delimited JSON. Most APIs use JSON — this is the default.

### CsvDecoder
For APIs that return CSV data.

```yaml
decoder:
  type: CsvDecoder
  delimiter: ","
  encoding: utf-8
```

### XmlDecoder
For SOAP or XML-based APIs.

### GzipDecoder / ZipfileDecoder
For APIs that return compressed responses.

```yaml
decoder:
  type: GzipDecoder
  decoder:
    type: JsonDecoder
```

---

## 8. Putting It All Together

Here's a complete manifest for collecting GitHub issues with pagination, incremental sync, and substreams:

```yaml
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: github_issues
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://api.github.com/repos/{{ stream_partition['repository'] }}/issues"
        http_method: GET
        error_handler:
          type: DefaultErrorHandler
          max_retries: 3
          backoff_strategies:
            - type: WaitTimeFromHeader
              header: Retry-After
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
      paginator:
        type: CursorPagination
        cursor_value: "{{ headers['link']['next'] }}"
        stop_condition: "{{ not headers.get('link', {}).get('next') }}"
      partition_router:
        type: SubstreamPartitionRouter
        parent_stream_configs:
          - type: ParentStreamConfig
            parent_key: full_name
            partition_field: repository
            stream:
              type: DeclarativeStream
              name: repos
              retriever:
                type: SimpleRetriever
                requester:
                  type: HttpRequester
                  url: "https://api.github.com/user/repos"
                  http_method: GET
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: updated_at
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
    transformations:
      - type: AddFields
        fields:
          - path: ["_collected_at"]
            value: "{{ now_utc() }}"
```

---

## Quick Reference

| Category | Common Choice | When to Use |
|---|---|---|
| **Extractor** | DpathExtractor | Almost always — navigate into JSON response |
| **Pagination** | NoPagination | API returns everything at once |
| | CursorPagination | API provides next-page cursor/link |
| | OffsetIncrement | API uses offset + limit |
| | PageIncrement | API uses page numbers |
| **Incremental** | DatetimeBasedCursor | Track updates by timestamp |
| | IncrementingCountCursor | Track by numeric ID |
| **Partition** | (none) | Single endpoint, no splitting |
| | SubstreamPartitionRouter | Parent-child (repos → issues) |
| | ListPartitionRouter | Iterate over a list of values |
| **Transform** | AddFields | Add metadata fields |
| | RemoveFields | Strip sensitive data |
| | KeysToSnakeCase | Normalize key format |
| **Error** | DefaultErrorHandler | Standard retry with backoff |
| **Decoder** | JsonDecoder | Most APIs (default) |
| | CsvDecoder | CSV endpoints |

---

## Schema Generation

The form-schema.ts file is auto-generated:

```bash
# Regenerate from latest Airbyte schema
cd react/packages/urc-builder
node scripts/generate-form-schema.js
```

This reads `declarative_component_schema.yaml` and `urc_extensions_schema.yaml`, producing TypeScript arrays of `ComponentFormDef` objects. The builder UI renders forms dynamically from these definitions — no component types are hardcoded.

Current counts: **53 component types** across **11 categories** (authenticator, paginator, extractor, transformation, error_handler, cursor, decoder, partition_router, event_timestamp, retriever, requester).
