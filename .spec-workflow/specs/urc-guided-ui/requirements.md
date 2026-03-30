# URC Guided UI — Connector Builder for Splunk

## Introduction

Build a guided, zero-training connector configuration UI for the URC (Universal REST Client) Splunk add-on. The UI helps users create, test, and deploy REST API connector manifests without writing YAML by hand. It should feel like a native Splunk experience — intuitive enough that any Splunk admin can configure a new API data source in under 5 minutes, with inline hints, smart defaults, and a live test panel.

The UI generates Airbyte-compatible manifest YAML behind the scenes. Advanced users can switch to a raw YAML editor at any time. The existing UCC-generated Configuration page (accounts, logging, proxy) remains untouched.

## Alignment with Product Vision

URC's mission is to be the definitive REST API data collection solution for Splunk. The backend engine (38 components, Pydantic v1 validation, Airbyte manifest format) is working end-to-end. The guided UI is what turns a powerful-but-expert-only tool into something any Splunk admin can use. Without it, the manifest YAML textarea is a wall of complexity. With it, URC becomes the "Splunk DB Connect, but for REST APIs."

## Architecture Context

### Reference Implementation
CIMplicity AI App (Apache 2.0, Splunkbase #7945) demonstrates the hybrid pattern:
- UCC handles settings (accounts, logging, proxy) as auto-generated config tabs
- A separate Splunk page hosts a full React wizard (5 steps, 3-column layout)
- Navigation: custom page as default view, UCC `configuration` as second nav item
- Build: UCC generates backend → copy into React monorepo → Webpack → package
- REST: `PersistentServerConnectionApplication` handlers with `passSystemAuth=true`

### Existing URC Backend
- 38 registered components (auth, pagination, extraction, transformations, decoders, error handlers, partition routers)
- Test Connection endpoint at `/services/urc_app/test_connection` (POST: manifest + account → sample records)
- Manifest validation via Pydantic v1 models
- KV Store checkpointing for incremental sync
- Jinja2 interpolation for `{{ config['key'] }}` references

### Existing React Infrastructure
- Monorepo at `react/` with Webpack 5, Babel 7, TypeScript 5.8
- `@splunk/react-ui` v5.9.0, `@splunk/react-page` v8.2.1, `@splunk/themes` v1.6.0
- `styled-components` v5.3.10
- CIMplicity reference at `.ref/cimplicity-ai-app/`

## Requirements

### R1: Schema-Driven Stream Configuration

**User Story:** As a Splunk admin, I want to configure a REST API connection through a visual form that covers all supported component types, so that I don't need to learn YAML manifest syntax.

#### Acceptance Criteria
1. WHEN user opens the Connector Builder page THEN system SHALL display collapsible card sections for each configuration area, with fields auto-rendered from the declarative component JSON Schema
2. WHEN a field has a `oneOf`/`anyOf` schema (e.g., authenticator, paginator, retriever type) THEN system SHALL show a dropdown selector with descriptions for each option, and conditionally render the selected type's fields
3. WHEN a stream uses `AsyncRetriever` THEN system SHALL show sub-sections for creation, polling, and download requesters with their respective fields
4. WHEN any component type in the schema is configurable THEN system SHALL render a working form for it — either in a curated card (common types) or in the auto-rendered "Advanced" catch-all section
5. WHEN a schema property has a `description` THEN system SHALL display it as help text on the field
6. IF user has not completed required fields (per schema `required` array) THEN system SHALL highlight them with actionable validation messages
7. WHEN the schema adds new component types or fields (upstream updates) THEN the UI SHALL render them automatically via `SchemaFormRemainingFields` without code changes

### R2: Multi-Stream Support

**User Story:** As a data engineer, I want to configure multiple API streams in one connector (e.g., users, repos, issues from GitHub), so that related data sources are managed together.

#### Acceptance Criteria
1. WHEN user opens the builder THEN system SHALL show a stream list sidebar where streams can be added, removed, reordered, and renamed
2. WHEN user adds a new stream THEN system SHALL create it with sensible defaults (GET method, JSON decoder, no pagination) and focus the configuration panel on it
3. WHEN user configures a SubstreamPartitionRouter THEN system SHALL show a parent stream picker from existing streams and explain the parent→child relationship with a visual hint
4. WHEN user removes a stream THEN system SHALL confirm the action and warn if other streams reference it as a parent

### R3: Live Test Panel

**User Story:** As a Splunk admin, I want to test my API configuration before saving, so that I can verify the connection works and see what data will be collected.

#### Acceptance Criteria
1. WHEN user clicks "Test Connection" THEN system SHALL call the existing `/services/urc_app/test_connection` endpoint with the current form state serialized as manifest YAML
2. WHEN test succeeds THEN system SHALL display sample records in a table view with field names as columns
3. WHEN test succeeds THEN system SHALL also show the raw JSON response and the HTTP request details (URL, headers, params) in separate tabs
4. WHEN test fails THEN system SHALL display the error message with actionable guidance (e.g., "401 Unauthorized — check your API key in the Account tab")
5. IF test is running THEN system SHALL show a spinner and allow cancellation
6. WHEN test returns records THEN system SHALL show record count and auto-detect field paths for the record_selector if the user hasn't configured one yet

### R4: YAML Editor Mode

**User Story:** As a power user, I want to view and edit the raw YAML manifest, so that I can make advanced configurations not covered by the form or paste in manifests from other sources.

#### Acceptance Criteria
1. WHEN user clicks "YAML" toggle THEN system SHALL display the current configuration as formatted YAML in a syntax-highlighted editor
2. WHEN user edits YAML and switches back to "Form" mode THEN system SHALL parse the YAML and update the form fields accordingly, showing validation errors for invalid YAML
3. WHEN YAML contains syntax errors THEN system SHALL show inline error markers with line numbers
4. WHEN user pastes an external manifest THEN system SHALL validate it against the Pydantic schema and highlight unsupported or unknown fields
5. IF YAML and form are out of sync (user edited both) THEN system SHALL use the most recently edited source as truth and notify the user

### R5: Inline Help and Guidance

**User Story:** As a first-time user, I want contextual help at every step, so that I can configure a connector without reading external documentation.

#### Acceptance Criteria
1. WHEN user hovers over or focuses any form field THEN system SHALL display a tooltip or help panel explaining the field's purpose, expected format, and an example value
2. WHEN user opens a configuration section THEN system SHALL show a brief intro paragraph explaining what that section does in plain English (e.g., "Pagination tells URC how to fetch all pages of results when the API returns data in chunks")
3. WHEN user selects an auth type THEN system SHALL show a help card with a link-style "Learn more" that explains when to use this auth type (e.g., "Most APIs that give you a long-lived token use Bearer auth")
4. WHEN user is on the record extraction step THEN system SHALL show an example API response (from test results if available) alongside the field path input, so they can visually map where records live
5. IF user leaves a section empty (e.g., no pagination) THEN system SHALL show a non-blocking hint: "No pagination configured — URC will fetch only the first page of results"

### R6: Template Gallery

**User Story:** As a Splunk admin, I want to start from a pre-built template for common APIs (GitHub, ServiceNow, Infoblox, Azure, generic REST), so that I don't have to configure everything from scratch.

#### Acceptance Criteria
1. WHEN user creates a new connector THEN system SHALL offer a "Start from Template" option alongside "Start from Scratch"
2. WHEN user selects a template THEN system SHALL pre-populate all form fields with the template's configuration, leaving only credentials empty
3. WHEN template is loaded THEN system SHALL highlight which fields the user still needs to fill in (marked with "You" badge or similar)
4. System SHALL include templates for at least: Generic REST API, GitHub, ServiceNow Table API, Infoblox WAPI, Azure Management API
5. WHEN user modifies a template THEN system SHALL not prevent changes — templates are starting points, not locked configurations

### R7: Save and Deploy Flow

**User Story:** As a Splunk admin, I want to save my connector configuration as a Splunk modular input, so that it runs on a schedule and collects data into my chosen index.

#### Acceptance Criteria
1. WHEN user clicks "Save" THEN system SHALL serialize the form state to manifest YAML and create/update a Splunk input via the UCC REST API
2. WHEN saving THEN system SHALL prompt for: input name, collection interval, target index, and sourcetype — with smart defaults
3. WHEN save succeeds THEN system SHALL show a success message with a link to the Inputs list and a link to search the target index
4. IF input name already exists THEN system SHALL offer to update the existing input or create a new one
5. WHEN user returns to edit an existing input THEN system SHALL load the manifest YAML back into the form, round-tripping correctly

### R8: Responsive Layout and Navigation

**User Story:** As a Splunk admin, I want the builder to feel like a native Splunk page that integrates with the existing URC add-on navigation.

#### Acceptance Criteria
1. System SHALL render the builder as a Splunk page using `@splunk/react-page` layout with proper theme support (light/dark)
2. System SHALL appear in the URC app navigation alongside the existing Configuration page (nav order: Builder as default, Configuration second)
3. System SHALL use `@splunk/react-ui` components exclusively for all form controls, buttons, cards, and layout — no custom CSS for standard patterns
4. System SHALL use a responsive layout that works on screens 1280px and wider
5. System SHALL follow Splunk design language: card-based sections, consistent spacing via theme tokens, standard button placement (primary right, secondary left)

## Non-Functional Requirements

### Code Architecture and Modularity
- **CIMplicity pattern**: Separate component library package (`urc-builder-home`) from Splunk app wrapper (`urc-builder-app`), following the proven monorepo structure
- **Single Responsibility**: Each wizard section (auth, pagination, extraction, etc.) is its own React component
- **Shared utilities**: REST helper module (`api.ts`) for all Splunk/UCC endpoint calls, following CIMplicity's `postToEndpoint` pattern
- **Manifest serialization**: Bidirectional form-state ↔ YAML conversion as a pure module (no UI dependencies) for testability

### Performance
- Builder page SHALL load in under 3 seconds on a standard Splunk instance
- Test connection SHALL show results within the API's response time + 2 seconds overhead
- YAML editor SHALL handle manifests up to 500 lines without lag
- Form state changes SHALL not cause full-page re-renders

### Security
- All credential fields SHALL use UCC's encrypted storage — the builder form never stores or displays raw credentials
- REST calls to Splunk endpoints SHALL include CSRF tokens via `@splunk/splunk-utils` helpers
- The builder SHALL not send manifest content to any external service
- Template gallery SHALL be local (bundled with the app), not fetched from the internet

### Reliability
- Form state SHALL persist in browser sessionStorage to survive accidental page refreshes
- YAML parsing errors SHALL not crash the builder — show error state and allow recovery
- If test_connection endpoint is unavailable, the builder SHALL still allow form editing and saving

### Usability
- Every form field SHALL have a label, help text, and placeholder example
- Error messages SHALL be actionable ("Enter your API base URL, e.g., https://api.github.com" not "Required field")
- The builder SHALL work without training — a Splunk admin who has never seen URC should be able to configure a JSONPlaceholder connector by following the UI hints alone
- Color and iconography SHALL follow Splunk theme tokens — no hardcoded colors
- All interactive elements SHALL be keyboard-accessible
