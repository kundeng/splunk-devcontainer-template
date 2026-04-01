# Requirements: URC Builder UI — Stream Dashboard & Schema-Driven Editor

## Introduction

A custom React UI that complements UCC's auto-generated configuration pages. Two screens: a **Stream Dashboard** (list all streams, status, tags, bulk actions) and a **Stream Builder** (tabbed, schema-driven form for creating/editing stream configurations). Sits alongside UCC's existing Accounts, Logging, and Proxy pages — does not replace them.

The UI must detect unsupported manifest configurations (e.g., OAuth2 PKCE, JWT RS256) **before** runtime and surface clear warnings. Debug logging must be toggleable per-stream, enriching AOP instrumentation output without changing component code.

## Alignment with Product Vision

URC's goal is a **generic REST API connector for Splunk** where non-developers can configure API data collection through a visual interface. The current UCC-generated UI has a raw YAML textarea for the manifest — unusable for the target audience. This spec delivers the visual form that makes URC accessible to Splunk admins, not just Python developers.

## Requirements

### R1: Stream Dashboard (Screen 1)

**User Story:** As a Splunk admin, I want a single view of all configured REST API streams so that I can see what's running, what's failing, and manage streams in bulk.

#### Acceptance Criteria

1. WHEN the user navigates to the "Streams" page THEN the system SHALL display a table of all configured URC inputs with columns: Name, Tags, Account, Status, Last Run, Records (last run), and Interval.
2. WHEN inputs are loaded THEN the system SHALL read from UCC's standard `GET /services/data/inputs/urc_app_input` endpoint — no custom REST endpoints for listing.
3. WHEN the user clicks a tag filter THEN the table SHALL show only streams matching that tag.
4. WHEN the user clicks a stream row THEN the system SHALL navigate to the Stream Builder (Screen 2) in edit mode for that stream.
5. WHEN the user clicks "+ New Stream" THEN the system SHALL navigate to the Stream Builder in create mode.
6. WHEN the user selects multiple streams THEN bulk actions SHALL be available: Enable, Disable, Delete, Set Tags.
7. WHEN the dashboard loads THEN summary cards SHALL show: total streams, active count, error count.
8. WHEN a stream has errors from its last run THEN the Status column SHALL show an error indicator with the error type from the last `action=error` structured log.

### R2: Stream Builder (Screen 2) — Tabbed Editor

**User Story:** As a Splunk admin, I want a guided, tabbed form to configure a REST API stream so that I don't need to write YAML by hand.

#### Acceptance Criteria

1. WHEN the user opens the Stream Builder THEN it SHALL display five tabs: Connection, Data Mapping, Splunk Output, Schedule & Tags, and Test & Preview.
2. WHEN the user navigates between tabs THEN form state SHALL be preserved (no data loss on tab switch).
3. WHEN the user clicks Save THEN the system SHALL serialize the form state into an Airbyte declarative manifest YAML and POST it to UCC's standard input endpoint.
4. WHEN the user opens an existing stream for editing THEN the system SHALL parse the stored manifest YAML and populate all form fields.
5. WHEN the manifest contains an unsupported component type THEN the form SHALL display an inline warning identifying the specific unsupported feature and suggesting alternatives.

### R3: Connection Tab

**User Story:** As a Splunk admin, I want to select an account and configure the API connection without understanding YAML auth blocks.

#### Acceptance Criteria

1. WHEN the Connection tab loads THEN it SHALL show: Account (dropdown from UCC accounts), Base URL (text), HTTP Method (GET/POST dropdown).
2. WHEN the user selects an account THEN a read-only summary of the account's auth type SHALL be displayed (e.g., "API Key via X-API-Key header").
3. WHEN the user changes the Account dropdown THEN the system SHALL NOT modify the account itself — account management happens in UCC's Configuration page.

### R4: Data Mapping Tab — Schema-Driven Forms

**User Story:** As a Splunk admin, I want to configure record extraction, pagination, and incremental sync through dropdown menus and form fields so that I don't need to understand Airbyte manifest syntax.

#### Acceptance Criteria

1. WHEN the Data Mapping tab loads THEN it SHALL show collapsible sections: Record Selector, Pagination, Incremental Sync, Transformations, Error Handling.
2. WHEN the user selects a component type from a dropdown (e.g., Pagination Strategy → "Offset Increment") THEN the form SHALL render fields specific to that type, derived from the declarative component schema.
3. WHEN a field has an enum in the schema THEN it SHALL render as a dropdown with the enum values as options.
4. WHEN a field has a description in the schema THEN the description SHALL appear as help text below the field.
5. WHEN the user selects a component type that is not in the engine's registry of 50 supported types THEN the form SHALL show a warning: "This component type is not currently supported by the URC engine."
6. WHEN the manifest references OAuth2 with PKCE fields OR JWT with RS256/ES256 algorithm THEN the form SHALL show a specific warning explaining the limitation.

### R5: Splunk Output Tab

**User Story:** As a Splunk admin, I want to configure where data lands in Splunk (index, sourcetype, source) and which field becomes the event timestamp, especially when the data has multiple datetime candidates.

#### Acceptance Criteria

1. WHEN the Splunk Output tab loads THEN it SHALL show: Index (dropdown from Splunk indexes), Sourcetype (text with default `urc:api:json`), Source override (optional text), Host override (optional text).
2. WHEN the Index dropdown loads THEN it SHALL populate from Splunk's `GET /services/data/indexes` endpoint.
3. WHEN a test fetch has been run (Tab 5) THEN the Timestamp section SHALL display all detected datetime fields from the sample records with radio buttons for selection.
4. WHEN the user selects a timestamp field THEN a preview SHALL show the parsed value for 2-3 sample records to confirm correct parsing.
5. IF no test fetch has been run THEN the Timestamp section SHALL show: "Run a test fetch to detect available timestamp fields" with a link to Tab 5.

### R6: Schedule & Tags Tab

**User Story:** As a Splunk admin, I want to set the collection interval, apply tags for organization, and enable/disable the stream.

#### Acceptance Criteria

1. WHEN the Schedule tab loads THEN it SHALL show: Interval (number input, 10-86400 seconds), Tags (comma-separated text input), Enabled/Disabled toggle.
2. WHEN the user enters tags THEN the system SHALL store them as a comma-separated string in the UCC input's `tags` field in `inputs.conf`.
3. WHEN the user saves a stream with tags THEN the Stream Dashboard (Screen 1) SHALL be able to filter by those tags.

### R7: Test & Preview Tab

**User Story:** As a Splunk admin, I want to test my API configuration and see sample data before saving, so that I can verify the connection works and the data looks right.

#### Acceptance Criteria

1. WHEN the user clicks "Test Connection" THEN the system SHALL POST to `/services/urc_app/test_connection` with the current form state serialized as manifest YAML.
2. WHEN the test succeeds THEN the panel SHALL display: status, record count, sample records (JSON), and detected field list with types.
3. WHEN the test fails THEN the panel SHALL display the error message with actionable context (e.g., "401 Unauthorized — check account credentials in Configuration → Accounts").
4. WHEN the test succeeds THEN the system SHALL cache the result in component state so the user can navigate to other tabs without re-fetching.
5. WHEN the manifest uses a non-GET HTTP method (POST, DELETE, PUT) THEN the Test button SHALL show a warning: "This test will make a [METHOD] request to the API. Proceed?" before executing.
6. WHEN test results are available THEN timestamp candidate detection SHALL run and populate the Splunk Output tab's timestamp picker.

### R8: Manifest Validation & Capability Detection (Backend)

**User Story:** As a Splunk admin, I want the UI to warn me about unsupported configurations before I save, so that I don't discover problems at runtime.

#### Acceptance Criteria

1. WHEN a manifest is validated THEN the system SHALL check every component `type` field against the engine's `REGISTRY` and return structured results: `[{path, type, severity, message}]`.
2. WHEN a component type is not in `REGISTRY` THEN the result SHALL include `severity: "error"` with message "Unknown component type '{type}'".
3. WHEN a JWT authenticator specifies algorithm RS256, RS384, RS512, ES256, ES384, ES512 THEN the result SHALL include `severity: "warning"` with message explaining the `cryptography` library dependency.
4. WHEN an OAuth2 authenticator includes `code_challenge_method` or PKCE-related fields THEN the result SHALL include `severity: "warning"` with message "OAuth2 PKCE flow is not supported".
5. WHEN validation runs THEN it SHALL NOT make any HTTP requests — it is purely structural/schema analysis.
6. WHEN the test_connection endpoint is called THEN validation SHALL run first, and validation errors SHALL be returned before any network call.

### R9: Debug Logging Toggle

**User Story:** As a Splunk admin troubleshooting a specific stream, I want to turn on debug logging for just that stream so that I get detailed diagnostics without flooding logs from all streams.

#### Acceptance Criteria

1. WHEN the user enables debug mode for a stream THEN the system SHALL store `debug=1` in the input's configuration stanza.
2. WHEN the modular input handler runs a stream with `debug=1` THEN the Python logger for that stream SHALL be set to DEBUG level.
3. WHEN debug is enabled THEN AOP after-hooks SHALL emit additional fields: url, stream_slice, response_status, response_size — at DEBUG level.
4. WHEN debug is enabled THEN AOP before-hooks SHALL emit method entry with bound arguments at DEBUG level.
5. WHEN debug is disabled (default) THEN the additional DEBUG-level fields SHALL NOT be formatted or emitted (logger level check before string formatting).
6. WHEN the user views the Test & Preview tab with debug enabled THEN structured log lines from the test run SHALL be displayed in a log viewer panel.

### R10: Tags Field in UCC Configuration

**User Story:** As a Splunk admin, I want to tag streams for organization so that I can filter and group them in the dashboard.

#### Acceptance Criteria

1. WHEN a new input is created or edited THEN the form SHALL include a "Tags" text field accepting comma-separated values.
2. WHEN tags are saved THEN they SHALL be stored in `inputs.conf` via UCC's standard REST endpoint (the `tags` field is declared in `globalConfig.json`).
3. WHEN the Stream Dashboard loads THEN it SHALL extract unique tags from all inputs and render them as filter options.

## Non-Functional Requirements

### Code Architecture and Modularity

- **React monorepo structure:** `@splunk/urc-builder` component library (reusable) + `@splunk/urc-app` page wrappers — existing scaffold preserved.
- **State management:** React Context + useReducer for form state. No Redux — the app is not complex enough to justify it.
- **Schema-driven rendering:** Form sections generated from `declarative_component_schema.yaml` at build time (static import), not runtime YAML parsing.
- **Separation of concerns:** Form serialization (form state ↔ manifest YAML) in a dedicated `manifest-serializer.ts` module, independent of React components.

### Performance

- Stream Dashboard SHALL load and render 100+ streams in under 2 seconds.
- Schema-driven form rendering SHALL not block the UI thread (lazy-load heavy sections).
- Test results SHALL be cached in component state to avoid redundant API calls.

### Security

- All REST calls SHALL use Splunk session tokens (inherited from the page context). No custom auth.
- Manifest YAML SHALL be sanitized before display to prevent XSS via malicious field values.
- Encrypted account fields (api_key, password, bearer_token, client_secret) SHALL never appear in the React UI — only the account name and auth type summary.

### Reliability

- Form state SHALL survive tab switching within the builder without data loss.
- If the UCC REST endpoint is unreachable, the dashboard SHALL show a clear error state, not a blank page.
- Validation errors SHALL be non-blocking warnings (user can still save), except for truly invalid YAML (blocking error).

### Usability

- Navigation: Streams (custom) → Configuration (UCC-generated) → Search (Splunk built-in). Streams is the default landing page.
- The builder tabs follow a left-to-right "story" (Connection → Data → Output → Schedule → Test) but are not a wizard — any tab is accessible at any time.
- All form fields SHALL have help text derived from the Airbyte schema descriptions.
- Light and dark Splunk themes SHALL both be supported (using `@splunk/themes` variables).
