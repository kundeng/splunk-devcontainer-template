# URC Phase 2 — Feature Completeness and UI

## Introduction

Fill feature gaps identified from competitive analysis (Baboonbones REST Input, LukeMurphey Web Input, Airbyte Connector Builder) and add UI enhancements. Goal: URC should be the definitive REST API data collection solution for Splunk.

## Competitive Gap Requirements

### R1: Missing HTTP Features

**User Story:** As a Splunk admin, I want full HTTP method support, proxy configuration, SSL certificate options, and cookie persistence so I can connect to any REST API regardless of its network requirements.

#### Acceptance Criteria
1. Support HTTP methods: GET, POST, PUT, PATCH, DELETE, HEAD
2. HTTPS proxy support (host, port, username, password) — use UCC proxy settings tab
3. SSL certificate verification toggle + custom CA bundle path
4. Cookie persistence across requests within a collection cycle
5. Configurable request timeout (default 60s, user-configurable)

### R2: Missing Auth Types

**User Story:** As a Splunk admin, I want Digest auth, JWT auth, and Session Token auth so I can connect to APIs that use these authentication methods.

#### Acceptance Criteria
1. HTTP Digest authentication
2. JWT authentication (sign tokens with configurable algorithm, headers, payload)
3. Session Token authentication (login request → extract token → inject into subsequent requests)
4. Custom auth handler support via Python plugin (@component decorator)

### R3: Missing Components — Transformations

**User Story:** As a data engineer, I want to transform records before indexing (add fields, remove fields, flatten, rename keys) so that the data arrives in Splunk in the format I need.

#### Acceptance Criteria
1. AddFields — add computed fields from Jinja2 expressions
2. RemoveFields — remove specified field paths
3. KeysToLower — lowercase all keys
4. KeysToSnakeCase — convert camelCase to snake_case
5. FlattenFields — flatten nested objects

### R4: Missing Components — Substream Partition Router

**User Story:** As a data engineer collecting from APIs with parent-child relationships (e.g., ServiceNow tables referencing other tables, GitHub repos → issues), I want to iterate over parent records and fetch child records for each.

#### Acceptance Criteria
1. SubstreamPartitionRouter — iterate over parent stream records, extract a key, use it to partition child stream requests
2. ListPartitionRouter — iterate over a static or config-driven list of values
3. Support nested partition contexts (e.g., org → repo → issue)

### R5: Missing Components — Decoders

**User Story:** As a data engineer, I want to collect data from APIs that return CSV, XML, JSONL, or compressed responses, not just JSON.

#### Acceptance Criteria
1. JSON decoder (current, default)
2. JSONL decoder (newline-delimited JSON)
3. CSV decoder (configurable delimiter, encoding)
4. XML decoder (convert to dict via xmltodict)
5. GZip decoder (decompress then parse with inner decoder)

### R6: Network Resilience

**User Story:** As a Splunk admin, I want robust error handling with configurable retry, rate limiting awareness, and proper backoff so that transient API issues don't cause data loss.

#### Acceptance Criteria
1. CompositeErrorHandler — chain multiple error handlers
2. HttpResponseFilter — match responses by status code, error message, or predicate
3. WaitTimeFromHeader — extract retry timing from response headers (Retry-After)
4. Exponential and constant backoff strategies (existing, enhance)
5. Response action types: RETRY, FAIL, IGNORE, SUCCESS

## UI Enhancement Requirements

### R7: YAML Editor with Syntax Highlighting

**User Story:** As a Splunk admin, I want a proper YAML editor with syntax highlighting when editing the manifest, instead of a plain textarea.

#### Acceptance Criteria
1. Replace manifest textarea with CodeMirror 6 custom control
2. YAML syntax highlighting and auto-indent
3. Error markers for invalid YAML
4. Seamless integration with UCC form save flow

### R8: Test Connection and Data Preview

**User Story:** As a Splunk admin, I want to test my API connection and preview sample records before enabling the input, so I can verify the manifest works correctly.

#### Acceptance Criteria
1. "Test Connection" button in the input form that calls a custom REST endpoint
2. Custom REST endpoint performs a dry-run collection (first page only)
3. Shows success/failure status and sample records inline
4. Displays clear error messages if the manifest or credentials are wrong

### R9: Connector Builder-Style Guided UI (Phase 2b)

**User Story:** As a Splunk admin who doesn't know YAML, I want a guided form interface (similar to Airbyte Connector Builder) that lets me configure API connections visually with a stream list sidebar, per-stream config panel, and test panel.

#### Acceptance Criteria
1. Sidebar: list of configured streams, add/remove
2. Per-stream panel: URL, auth, pagination, record extraction sections
3. Test panel: send request, show response, show extracted records
4. Dual mode: form view and YAML view with seamless switching
5. Progressive disclosure: show advanced options (error handling, transformations) only when expanded

## Schema Extension Requirements

### R10: Web Scraping Support (LukeMurphey parity)

**User Story:** As a data engineer, I want to collect data from web pages using CSS selectors, not just REST APIs, so I can monitor web content changes.

#### Acceptance Criteria
1. New component type: HtmlExtractor with CSS selector support
2. Optional: XPath selector support
3. Response type: HTML (parse with BeautifulSoup or similar)
4. This extends the Airbyte schema with our own custom types

## Non-Functional Requirements

### Performance
- Support APIs returning up to 100,000 records per collection cycle
- Parallel stream collection within a single input

### Compatibility
- All features must work on Splunk Enterprise 10.2+ (Python 3.9)
- Pure Python only — no C extensions
- AppInspect compliant packaging
