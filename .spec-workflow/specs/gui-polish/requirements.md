# Requirements: GUI Polish

## Introduction

The URC app's React-based dashboard and stream builder have a functional MVP, but lack the table interactivity, form safety, and operational feedback that make a Splunk app feel production-ready. This spec covers ~18 improvements spanning dashboard UX, builder UX, operational visibility, and bulk/import-export workflows. Fine-grained collection metrics (event counts, throughput, checkpoint cursors) are deferred to a native Splunk Dashboard v2 XML panel rather than custom React, since Splunk's search-driven dashboards handle time-series data natively.

## Alignment with Product Vision

These improvements close the gap between "it works" and "I'd trust it in production." Tags become a real organizational tool, the table behaves like a proper data grid, the builder protects work-in-progress, and operators get immediate signal on stream health without digging through logs.

---

## Requirements

### Dashboard & Stream Management

#### REQ-1: Column Sorting

**User Story:** As a Splunk admin, I want to click column headers to sort the stream table, so that I can quickly find streams by name, status, account, or interval.

**Acceptance Criteria:**
1. WHEN user clicks a column header (Name, Status, Tags, Account, Interval) THEN the table SHALL sort rows by that column in ascending order
2. WHEN user clicks the same column header again THEN the sort SHALL toggle to descending order
3. WHEN a sort is active THEN the column header SHALL display a directional indicator (arrow)
4. IF no sort is active THEN the table SHALL display rows in creation order (current default)

---

#### REQ-2: Search / Text Filter

**User Story:** As a Splunk admin, I want to search streams by name or URL, so that I can find a specific stream in a large list.

**Acceptance Criteria:**
1. WHEN user types in the search box THEN the table SHALL filter to rows where name OR base URL contains the search text (case-insensitive)
2. WHEN the search box is cleared THEN the full list SHALL be restored
3. WHEN search is active alongside tag/status filters THEN all filters SHALL combine with AND logic

---

#### REQ-3: Functional Tag Filtering & Tag Management

**User Story:** As a Splunk admin, I want tags to be a first-class organizational tool, so that I can group and manage streams by project, team, or environment.

**Acceptance Criteria:**
1. WHEN user selects one or more tag chips THEN the table SHALL show only streams that have at least one selected tag (OR within tags, AND with other filters)
2. WHEN tags are displayed in the table THEN each tag chip SHALL be clickable to add that tag to the active filter
3. WHEN user edits a stream THEN the tag input SHALL offer autocomplete from existing tags across all streams
4. WHEN no streams use a given tag THEN that tag SHALL disappear from the filter bar

---

#### REQ-4: Table Pagination

**User Story:** As a Splunk admin managing many streams, I want the table paginated, so that the page stays responsive.

**Acceptance Criteria:**
1. WHEN more than 25 streams exist THEN the table SHALL paginate with 25 rows per page (configurable: 10/25/50)
2. WHEN paginated THEN navigation controls (prev/next, page numbers) SHALL appear below the table
3. WHEN filters are applied THEN pagination SHALL reset to page 1
4. IF fewer than the page-size threshold exist THEN pagination controls SHALL be hidden

---

#### REQ-5: Stream Cloning

**User Story:** As a Splunk admin, I want to duplicate an existing stream, so that I can quickly create similar configurations without starting from scratch.

**Acceptance Criteria:**
1. WHEN user selects "Clone" from a stream's action menu THEN the builder SHALL open pre-filled with that stream's configuration
2. WHEN cloning THEN the stream name SHALL be set to `{original}-copy` and the name field SHALL be editable
3. WHEN cloning THEN the mode SHALL be 'create' (not 'edit') so the original is not modified

---

#### REQ-6: Error Status in Dashboard

**User Story:** As a Splunk admin, I want to see which streams have recent errors, so that I can triage problems without searching logs.

**Acceptance Criteria:**
1. WHEN a stream's last collection run produced errors THEN its status badge SHALL show "Error" (red) instead of "Active"
2. WHEN the "Error" status filter is selected THEN only streams with recent errors SHALL be shown
3. WHEN user hovers or clicks an error badge THEN a tooltip or popover SHALL show the last error message and timestamp
4. IF no error state is available for a stream THEN the status SHALL remain "Active" or "Disabled" as before

*Implementation note:* Error state can be read from the app's structured log index or a KV store health record written by the collection engine.

---

#### REQ-7: Stream Grouping

**User Story:** As a Splunk admin, I want to visually group streams by account or tag, so that I can see related streams together.

**Acceptance Criteria:**
1. WHEN user selects a "Group by" option (Account or Tag) THEN the table SHALL display grouped sections with a header row per group
2. WHEN grouped THEN each group header SHALL show the group name and stream count
3. WHEN no grouping is active THEN the table SHALL display as a flat list (default)

---

### Builder & Edit Experience

#### REQ-8: Unsaved Changes Warning

**User Story:** As a Splunk admin editing a stream, I want to be warned before losing unsaved changes, so that I don't accidentally discard work.

**Acceptance Criteria:**
1. WHEN user has modified any field and attempts to navigate away (back button, browser nav, "Cancel") THEN a confirmation dialog SHALL appear
2. WHEN user confirms leave THEN navigation SHALL proceed and changes SHALL be discarded
3. WHEN user cancels leave THEN the user SHALL remain on the builder with changes intact
4. IF no fields have been modified THEN navigation SHALL proceed without a dialog

---

#### REQ-9: Draft Auto-Save

**User Story:** As a Splunk admin filling out a long stream configuration, I want my in-progress work saved automatically, so that a browser crash doesn't lose everything.

**Acceptance Criteria:**
1. WHEN the builder form state changes THEN the current state SHALL be persisted to browser localStorage within 2 seconds (debounced)
2. WHEN the builder loads in 'create' mode AND a draft exists in localStorage THEN a banner SHALL offer to restore the draft
3. WHEN user successfully saves a stream THEN the draft SHALL be cleared from localStorage
4. WHEN user explicitly discards the draft (via banner dismiss) THEN localStorage SHALL be cleared

---

#### REQ-10: Real-Time Field Validation

**User Story:** As a Splunk admin, I want form fields validated as I type, so that I catch mistakes early instead of at test time.

**Acceptance Criteria:**
1. WHEN user leaves (blurs) a required field that is empty THEN an inline error SHALL appear immediately
2. WHEN user enters a base URL THEN the field SHALL validate URL format on blur and show an error if invalid
3. WHEN user enters an interval THEN the field SHALL validate the range (10-86400) on blur
4. WHEN user enters a stream name THEN the field SHALL validate allowed characters (alphanumeric, underscore, hyphen) on blur
5. IF a field has an inline error AND user corrects it THEN the error SHALL clear on the next blur/change

---

#### REQ-11: Stream Name Uniqueness Check

**User Story:** As a Splunk admin, I want to know if a stream name is already taken before I try to save, so that I get a clear message instead of a cryptic backend error.

**Acceptance Criteria:**
1. WHEN user enters a stream name in create mode THEN the system SHALL check it against existing stream names
2. IF the name already exists THEN an inline error "A stream with this name already exists" SHALL appear
3. WHEN in edit mode THEN the name field SHALL be read-only (names are immutable after creation)

---

#### REQ-12: Cross-Field Validation

**User Story:** As a Splunk admin, I want the builder to warn me about inconsistent configurations, so that I don't create streams that will fail at collection time.

**Acceptance Criteria:**
1. WHEN HTTP method is POST/PUT/PATCH and no request body is configured THEN a warning SHALL appear on the Connection tab
2. WHEN incremental sync is configured but no cursor field is mapped THEN a warning SHALL appear on the Data Mapping tab
3. WHEN pagination is configured but record selector path is empty THEN a warning SHALL appear on the Data Mapping tab

---

#### REQ-13: Transformation Preview

**User Story:** As a Splunk admin configuring data transformations, I want to see a before/after preview, so that I can verify my configuration produces the expected output.

**Acceptance Criteria:**
1. WHEN test data is available AND transformations are configured THEN the Test & Preview tab SHALL show a "Before" and "After" panel
2. WHEN user modifies transformation config THEN the after panel SHALL update to reflect the new configuration
3. IF no test data is available THEN the preview SHALL show a message prompting the user to run a test first

---

#### REQ-14: Template String Preview

**User Story:** As a Splunk admin using Jinja-style template fields (e.g., URL paths with `{{ }}`), I want a live preview showing the resolved value, so that I can verify my templates are correct.

**Acceptance Criteria:**
1. WHEN a template-type field contains `{{ }}` expressions THEN a preview line SHALL render below the input showing the resolved value using sample context
2. WHEN the template references an unknown variable THEN the preview SHALL highlight the unresolved variable in a warning color
3. IF no context is available THEN the preview SHALL show the raw template with placeholders indicated

---

### Accounts & Configuration

#### REQ-15: Account Health Indicator

**User Story:** As a Splunk admin, I want to see at a glance whether each account's credentials are valid, so that I don't have to run a test stream to discover expired tokens.

**Acceptance Criteria:**
1. WHEN the dashboard or account dropdown loads THEN each account SHALL show a health icon (green check, yellow warning, grey unknown)
2. WHEN an account's last test connection returned a 401/403 THEN the health icon SHALL show warning/red
3. WHEN user hovers the health icon THEN a tooltip SHALL show "Last verified: {timestamp}" or "Never tested"

---

### Operational Visibility (React UI)

#### REQ-16: Last-Run / Error Indicator

**User Story:** As a Splunk admin, I want to see when each stream last ran and whether it succeeded, so that I can spot stale or broken streams.

**Acceptance Criteria:**
1. WHEN streams are listed THEN each row SHALL show a "Last Run" column with relative timestamp ("2 min ago", "3 hours ago", or "Never")
2. WHEN the last run had errors THEN the Last Run cell SHALL show an error icon with count
3. WHEN user clicks the last-run cell THEN a popover SHALL show the last 3 run results (timestamp, status, record count or error message)

---

### Operational Visibility (Splunk Dashboard v2)

#### REQ-17: Collection Metrics Dashboard

**User Story:** As a Splunk admin, I want a native Splunk dashboard showing collection throughput, errors over time, checkpoint status, and per-stream health, so that I can monitor the app using standard Splunk tools.

**Acceptance Criteria:**
1. WHEN user navigates to the "Collection Metrics" view THEN a Splunk Dashboard v2 (JSON) dashboard SHALL load
2. The dashboard SHALL include panels for:
   - Event ingestion rate over time (timechart, per stream)
   - Error rate and error types over time
   - Top errors by stream (table)
   - Stream health summary (single-value panels: total active, total errored, total disabled)
   - Last successful collection per stream (table with relative timestamps)
3. WHEN user selects a time range THEN all panels SHALL respect the time picker
4. The dashboard SHALL use the app's internal index (`_internal` or a dedicated metrics index) as the data source

---

### Import / Export / Bulk

#### REQ-18: Import/Export Stream Configurations

**User Story:** As a Splunk admin, I want to export stream configs as YAML/JSON and import them on another instance, so that I can share, back up, and migrate configurations.

**Acceptance Criteria:**
1. WHEN user clicks "Export" on a single stream THEN a YAML file containing the stream's manifest and metadata SHALL be downloaded
2. WHEN user selects multiple streams and clicks "Export Selected" THEN a single YAML file with all selected stream configs SHALL be downloaded
3. WHEN user clicks "Import" THEN a file picker SHALL accept a YAML/JSON file
4. WHEN importing THEN the system SHALL validate the file and show a preview of streams to be created
5. IF a stream name in the import conflicts with an existing stream THEN the user SHALL be prompted to skip, rename, or overwrite
6. WHEN import completes THEN a summary SHALL show how many streams were created/skipped/failed

---

## Non-Functional Requirements

### Code Architecture and Modularity
- Each feature SHALL be implemented as an isolated, testable module (hook, component, or utility)
- Shared logic (filtering, sorting, validation) SHALL live in dedicated hooks or utility files, not inline in components
- All new components SHALL use Splunk React UI primitives for visual consistency

### Testing Strategy
- Every feature SHALL have unit tests written before or alongside the implementation (test-a-little, develop-a-little)
- Dashboard context logic (filtering, sorting, grouping) SHALL have dedicated hook/utility tests
- Form validation rules SHALL have unit tests covering valid and invalid inputs
- Component tests SHALL use React Testing Library with Splunk UI mocks where needed
- Target: >80% line coverage on new code

### Performance
- Table sorting, filtering, and pagination SHALL operate on the client-side dataset without additional API calls (streams are typically under 500)
- localStorage auto-save SHALL be debounced to avoid excessive writes
- Dashboard SHALL render initial view within 2 seconds on a dataset of 100 streams

### Security
- Import/export SHALL sanitize stream names and field values to prevent XSS
- localStorage drafts SHALL NOT store account credentials or tokens
- Exported YAML SHALL NOT include sensitive account credentials

### Usability
- All new interactive elements SHALL have keyboard accessibility (tab navigation, enter/space activation)
- Error messages SHALL be actionable ("Base URL must start with http:// or https://", not "Invalid input")
- The Splunk Dashboard v2 panel SHALL work with Splunk's built-in dark and light themes
