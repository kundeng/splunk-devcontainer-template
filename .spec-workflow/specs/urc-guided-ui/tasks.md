# Tasks — URC Guided UI

## Phase 1: Foundation — Package Scaffold + State + Serializer

- [x] 1.1. Scaffold React packages (urc-builder + urc-app) + Splunk nav/view integration
  - Implemented via `task ucc:add-react APP_NAME=urc_app` which runs:
    1. `react/bin/splunk-create-noninteractive.mjs component UrcBuilder` — scaffolds component library
    2. `react/bin/splunk-create-noninteractive.mjs app UrcApp` — scaffolds Splunk app wrapper
    3. `react/bin/adapt-for-ucc.mjs` — applies 6 UCC overlay transformations (webpack, deps, entry point, nav.xml, view XML, HTML template)
  - Also created: `@splunk/create` as devDep, non-interactive scaffolding tooling, `ucc:add-react` Taskfile task
  - Updated golden path docs, .gitignore for react source tracking
  - _Note: Original task 1.2 (nav/view integration) was completed as part of this task by the adapt-for-ucc.mjs script_
  - Success: `cd react && yarn workspace @splunk/urc-app build` produces stage/appserver/static/pages/builder.js ✓

- [ ] 1.3. Implement BuilderState interfaces and useBuilderState hook
  - Files: react/packages/urc-builder/src/state/BuilderState.ts, react/packages/urc-builder/src/state/useBuilderState.ts
  - Define TypeScript interfaces (StreamConfig, BuilderState, TestResult) per design doc
  - Implement React state hook: addStream, removeStream, setField, mode toggle, validation
  - Session storage persistence for crash recovery
  - Purpose: Single source of truth for all UI state before building any components
  - _Leverage: design.md Data Models section_
  - _Requirements: R1, R2, R4_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React/TypeScript developer specializing in state management | Task: Create BuilderState.ts with all TypeScript interfaces from the design doc Data Models section, then implement useBuilderState.ts hook with addStream, removeStream, setField(path, value), toYaml, fromYaml, mode toggle, and sessionStorage persistence. | Restrictions: No external state libraries (no Redux, Zustand, etc.) — use React useState + useCallback. Keep it simple. The hook must be testable without DOM. | _Leverage: .spec-workflow/specs/urc-guided-ui/design.md (Data Models section) | Success: Hook compiles, addStream/removeStream work, setField updates nested paths, sessionStorage round-trips correctly._

- [ ] 1.4. Implement manifestSerializer (form ↔ YAML bidirectional conversion)
  - Files: react/packages/urc-builder/src/state/manifestSerializer.ts, react/packages/urc-builder/tests/manifestSerializer.test.ts
  - Pure functions: formStateToYaml(state) → YAML string, yamlToFormState(yaml) → BuilderState
  - Must handle all 38 component types, $ref definitions, $parameters
  - Round-trip fidelity: form → YAML → form should be lossless for all supported fields
  - Purpose: Core logic that makes form ↔ YAML toggle work. Must be rock-solid before building UI.
  - _Leverage: ucc/urc_app/package/lib/urc/manifest.py (Python reference), ucc/urc_app/tests/manifests/*.yaml (test fixtures)_
  - _Requirements: R4_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript developer with YAML serialization expertise | Task: Implement manifestSerializer.ts with formStateToYaml and yamlToFormState functions. Use js-yaml library. Test with all 5 manifest fixtures in ucc/urc_app/tests/manifests/ — parse each YAML, convert to form state, convert back to YAML, verify structural equivalence. | Restrictions: Pure functions only — no React, no DOM. Must handle all auth types, pagination strategies, partition routers, decoders, transformations, error handlers. Field ordering in output YAML should match Airbyte conventions. | _Leverage: ucc/urc_app/tests/manifests/*.yaml, ucc/urc_app/package/lib/urc/manifest.py | Success: All 5 test manifests round-trip without data loss. Unit tests pass for each component type._

## Phase 2: Core Layout + Stream Sidebar

- [ ] 2.1. Implement ConnectorBuilder root component (3-column grid)
  - Files: react/packages/urc-builder/src/ConnectorBuilder.tsx, react/packages/urc-builder/src/ConnectorBuilder.styles.ts
  - CSS Grid layout: 240px | 1fr | 280px columns, 1fr | auto rows
  - Placeholder slots for StreamSidebar, main content area, HelpPanel, TestPanel
  - Form/YAML mode toggle in sidebar
  - Splunk theme integration via styled-components
  - Purpose: Visual shell that all other components plug into
  - _Leverage: .ref/cimplicity-ai-app/packages/ci-mplicity-home/src/CiMplicityHome.jsx (MainGrid pattern)_
  - _Requirements: R8_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React/TypeScript frontend developer with Splunk UI experience | Task: Build the ConnectorBuilder root component with 3-column CSS Grid layout following the ASCII mockup in the design doc. Use styled-components with @splunk/themes tokens. Include placeholder divs for sidebar, main, help, and test panel areas. Add a Form/YAML toggle using @splunk/react-ui RadioBar. | Restrictions: Use @splunk/react-ui components and @splunk/themes tokens exclusively — no hardcoded colors or spacing. Must support light and dark Splunk themes. | _Leverage: .ref/cimplicity-ai-app/packages/ci-mplicity-home/src/CiMplicityHome.jsx (lines 109-115 MainGrid) | Success: Builder renders in Splunk with correct 3-column layout, adapts to light/dark theme, toggle switches between form and YAML placeholder._

- [ ] 2.2. Implement StreamSidebar (stream list + add/remove/rename)
  - Files: react/packages/urc-builder/src/panels/StreamSidebar.tsx
  - List of stream names with active highlight, click to select
  - Add stream button (creates with defaults), remove with confirmation, inline rename
  - Template gallery button
  - Purpose: Left panel navigation for multi-stream connectors
  - _Leverage: @splunk/react-ui List, Button, Modal for confirm dialog_
  - _Requirements: R2_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React UI developer | Task: Build StreamSidebar component showing stream names as a clickable list. Active stream is highlighted. Add button creates a new stream with defaults, remove button shows Modal confirmation. Inline rename on double-click. Use @splunk/react-ui components (List, Button, Modal, Text). | Restrictions: No drag-and-drop reordering yet — just add/remove/select/rename. Keep it simple. | _Leverage: @splunk/react-ui List, Button, Modal, Text | Success: Can add 3 streams, select between them, rename one, remove one with confirmation._

- [ ] 2.3. Implement HelpPanel (contextual help sidebar)
  - Files: react/packages/urc-builder/src/panels/HelpPanel.tsx, react/packages/urc-builder/src/utils/helpContent.ts
  - Right panel showing context-sensitive help for the currently focused field/section
  - Help content catalog: one entry per form field with description, example, and tip
  - Quick Start card at top when nothing is focused
  - Purpose: Zero-training usability — every field explains itself
  - _Leverage: @splunk/react-ui Card, Typography, Message_
  - _Requirements: R5_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: UX-focused React developer | Task: Build HelpPanel that shows contextual help based on a focusedField prop. Create helpContent.ts catalog with entries for every form field (endpoint URL, auth type, pagination strategy, etc.). Each entry has: title, description, example, tip. Show Quick Start card when nothing focused. | Restrictions: Help text must be in plain English aimed at Splunk admins, not developers. No jargon like "Jinja2 interpolation" — say "dynamic value from your Account config". | _Leverage: @splunk/react-ui Card, Typography, Heading | Success: Help panel updates when focusedField changes. All sections have help content. Quick Start shows by default._

## Phase 3: Configuration Sections

- [ ] 3.1. Implement EndpointSection
  - Files: react/packages/urc-builder/src/sections/EndpointSection.tsx
  - URL input with placeholder, HTTP method dropdown, request headers key-value editor, request params key-value editor
  - Optional request body (JSON/form/GraphQL) with textarea
  - Purpose: First section users see — configure where to call
  - _Leverage: @splunk/react-ui ControlGroup, Text, Select, TextArea, FormRows_
  - _Requirements: R1.1_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React form developer | Task: Build EndpointSection as a collapsible Accordion panel. Include: URL text input with placeholder "https://api.example.com/v1/items", HTTP method Select (GET/POST/PUT/PATCH/DELETE), dynamic key-value editors for headers and params using FormRows, optional body textarea that shows conditionally for POST/PUT/PATCH. Wire onChange to propagate values up. | Restrictions: Follow SectionProps interface from design doc. Use ControlGroup for every field with label + help text. | _Leverage: @splunk/react-ui Accordion, ControlGroup, Text, Select, TextArea, FormRows | Success: Can configure URL, method, add 2 headers, add 2 params. Values propagate to parent via onChange._

- [ ] 3.2. Implement AuthSection
  - Files: react/packages/urc-builder/src/sections/AuthSection.tsx
  - Auth type radio/select with plain-English descriptions for each type
  - Conditional fields per auth type (Bearer: token; Basic: username+password; OAuth2: client_id+secret+token_url; etc.)
  - Auto-populate with `{{ config['field'] }}` interpolation references
  - Preview of resulting auth header
  - Purpose: Authentication is the #1 config users struggle with — make it obvious
  - _Leverage: @splunk/react-ui RadioBar, ControlGroup, Text, Message_
  - _Requirements: R1.2, R5.3_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React form developer with auth knowledge | Task: Build AuthSection with auth type selector showing all 8 types (None, API Key, Bearer, Basic, OAuth2, Digest, JWT, Session Token). Each type has a 1-line plain-English description. Conditional fields show/hide based on selection. Default values use config interpolation syntax. Show preview of the resulting Authorization header. | Restrictions: Do not implement actual auth logic — this is a form builder that generates manifest YAML. Field defaults should reference config interpolation. | _Leverage: @splunk/react-ui RadioBar, ControlGroup, Text, Message, Accordion | Success: Select Bearer → shows token field with default config interpolation reference. Select OAuth2 → shows client_id, client_secret, token_url. Preview updates._ Select OAuth2 → shows client_id, client_secret, token_url. Preview updates._

- [ ] 3.3. Implement PaginationSection
  - Files: react/packages/urc-builder/src/sections/PaginationSection.tsx
  - Visual picker: None / Offset / Page Number / Cursor — each with plain-English explanation
  - Auto-populated defaults per strategy (page_size, param names)
  - "How it works" preview showing example request sequence
  - Purpose: Pagination is the second hardest concept — visual picker makes it intuitive
  - _Leverage: @splunk/react-ui RadioBar, ControlGroup, Text, Number, Card_
  - _Requirements: R1.3, R5.2_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer with API knowledge | Task: Build PaginationSection with RadioBar for strategy selection. Each option has a card-style description ("Offset — API uses offset + limit params, like 'give me items 100-200'"). Show strategy-specific fields with smart defaults. Include a "How it works" Card showing example request sequence (Request 1: ?offset=0&limit=100, Request 2: ?offset=100&limit=100, etc.). | Restrictions: Defaults must match common API patterns. Cursor pagination needs cursor_value and stop_condition template fields. | _Leverage: @splunk/react-ui RadioBar, Card, ControlGroup, Text, Number, Typography | Success: Select Offset → shows page_size=100, offset_param, limit_param. "How it works" preview updates dynamically._

- [ ] 3.4. Implement ExtractionSection
  - Files: react/packages/urc-builder/src/sections/ExtractionSection.tsx
  - Field path input with dot-notation hint (e.g., "data.items" or "result")
  - API response preview panel (populated from test results when available)
  - Visual highlighting of where records live in the response
  - Optional record filter condition
  - Purpose: Help users find records in nested API responses
  - _Leverage: @splunk/react-ui ControlGroup, Text, Code, TextArea_
  - _Requirements: R1.4, R3.6, R5.4_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build ExtractionSection with field_path text input and optional record filter. When testResponse prop is available, show the JSON response in a Code block with the field_path highlighted. Auto-suggest field_path from test response structure (find first array-valued key). | Restrictions: JSON preview must be read-only. Auto-suggest is a hint, not auto-applied. | _Leverage: @splunk/react-ui ControlGroup, Text, Code, Message | Success: Type "result" in field_path → preview highlights response.result array. Auto-suggest works when test data is available._

- [ ] 3.5. Implement IncrementalSection
  - Files: react/packages/urc-builder/src/sections/IncrementalSection.tsx
  - Enable/disable toggle with explanation ("only fetch new/updated records since last run")
  - Type picker: Datetime cursor vs Count cursor
  - Conditional fields: cursor_field, start_datetime, datetime_format for datetime; cursor_field, start_value for count
  - Purpose: Incremental sync is powerful but conceptually hard — make it opt-in with clear guidance
  - _Leverage: @splunk/react-ui Switch, ControlGroup, Text, Select, RadioBar_
  - _Requirements: R1.5_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build IncrementalSection with Switch toggle (default off). When enabled, show RadioBar for type (Datetime / Count). Datetime shows cursor_field, start_datetime, datetime_format with common format presets. Count shows cursor_field, start_value. Include Message explaining "URC will track the last value seen and only fetch newer records on the next run." | Restrictions: Datetime format presets should cover ISO 8601, Unix timestamp, and common API formats. | _Leverage: @splunk/react-ui Switch, RadioBar, ControlGroup, Text, Select, Message | Success: Toggle on → select Datetime → fill cursor_field="updated_at", format="%Y-%m-%dT%H:%M:%SZ". Values propagate correctly._

- [ ] 3.6. Implement TransformSection
  - Files: react/packages/urc-builder/src/sections/TransformSection.tsx
  - Add/remove transformation list using FormRows
  - Type dropdown for each: AddFields, RemoveFields, KeysToLower, KeysToSnakeCase, FlattenFields
  - Conditional fields per type (AddFields: field path + value pairs; RemoveFields: field path list; etc.)
  - Purpose: Post-extraction data shaping
  - _Leverage: @splunk/react-ui FormRows, Select, ControlGroup, Text_
  - _Requirements: R1.6_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build TransformSection with dynamic list of transformations using FormRows (add/remove). Each row has a type Select dropdown. Show conditional fields based on type: AddFields → field path + value pairs; RemoveFields → field path list; KeysToLower/KeysToSnakeCase → no extra fields; FlattenFields → separator. | Restrictions: Keep it simple — no drag-and-drop reorder. FormRows handles add/remove. | _Leverage: @splunk/react-ui FormRows, Select, ControlGroup, Text | Success: Can add AddFields with 2 field/value pairs, add KeysToLower (no extra config), remove a transform._

- [ ] 3.7. Implement ErrorHandlerSection
  - Files: react/packages/urc-builder/src/sections/ErrorHandlerSection.tsx
  - Max retries number input, backoff strategy selector, retry/fail status code lists
  - Smart defaults: 5 retries, exponential backoff, retry on 429/500/502/503
  - Purpose: Most users will use defaults — section exists for power users
  - _Leverage: @splunk/react-ui ControlGroup, Number, Select, Text_
  - _Requirements: R1.7_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build ErrorHandlerSection with sensible defaults pre-filled (5 retries, exponential backoff, retry on [429, 500, 502, 503]). Max retries as Number input, backoff strategy as Select (Constant/Exponential/Header), retry and fail status codes as comma-separated text inputs. | Restrictions: Defaults should be filled in so most users never need to touch this section. | _Leverage: @splunk/react-ui Accordion, ControlGroup, Number, Select, Text | Success: Section renders with all defaults filled. User can change max_retries and status codes._

## Phase 4: Test Panel + YAML Editor

- [ ] 4.1. Implement api.ts REST helper module
  - Files: react/packages/urc-builder/src/utils/api.ts
  - Functions: testConnection, listAccounts, listIndexes, saveInput, loadInput, listInputs
  - Uses @splunk/splunk-utils for createRESTURL and getDefaultFetchInit (CSRF tokens)
  - Error handling with typed error responses
  - Purpose: All backend communication in one module, following CIMplicity's pattern
  - _Leverage: .ref/cimplicity-ai-app/packages/ci-mplicity-home/src/utils/api.js_
  - _Requirements: R3, R7_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript developer with Splunk REST API experience | Task: Create api.ts with typed functions for all Splunk REST interactions. Follow CIMplicity's api.js pattern using createRESTURL and getDefaultFetchInit from @splunk/splunk-utils. Handle errors with typed responses. | Restrictions: Must include CSRF tokens on all POST/PUT/DELETE. Must handle 401 (redirect to login) and timeout gracefully. Do not use axios — use native fetch. | _Leverage: .ref/cimplicity-ai-app/packages/ci-mplicity-home/src/utils/api.js | Success: testConnection calls /services/urc_app/test_connection with correct headers. listAccounts returns Account[]. All functions have TypeScript types._

- [ ] 4.2. Implement TestPanel (bottom docked, tabbed results)
  - Files: react/packages/urc-builder/src/panels/TestPanel.tsx
  - Account selector dropdown, Test Connection button with spinner
  - TabLayout: Records (table view), Request (formatted HTTP request), Response (raw JSON), Schema (auto-inferred fields)
  - Auto-detect field_path from response and offer to apply
  - Collapsible (toggle visibility)
  - Purpose: Verify config works before saving — the core UX differentiator
  - _Leverage: @splunk/react-ui TabLayout, Table, Select, Button, WaitSpinner, Code_
  - _Requirements: R3_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build TestPanel as a bottom-docked collapsible panel. Include account Select dropdown, Test Connection button, and TabLayout with 4 tabs: Records (Table component), Request (Code block showing URL+headers+params), Response (Code block with raw JSON), Schema (auto-inferred field list). When test returns, analyze response to suggest field_path. | Restrictions: Use @splunk/react-ui Table for records — not a custom table. Limit displayed records to 10 rows. Show record count badge on Records tab. | _Leverage: @splunk/react-ui TabLayout, Table, Select, Button, WaitSpinner, Code, Badge | Success: Test button calls API, spinner shows during request, results display in all 4 tabs, field_path auto-suggestion works._

- [ ] 4.3. Implement YamlEditor (CodeMirror 6)
  - Files: react/packages/urc-builder/src/editor/YamlEditor.tsx
  - CodeMirror 6 with @codemirror/lang-yaml, line numbers, bracket matching
  - Splunk theme integration (light/dark via @codemirror/theme-one-dark conditional)
  - Inline error markers from YAML parse errors
  - Debounced onChange (300ms)
  - Purpose: Power user escape hatch + YAML paste support
  - _Leverage: @codemirror/lang-yaml, @codemirror/view, @codemirror/state_
  - _Requirements: R4_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer with CodeMirror experience | Task: Build YamlEditor component wrapping CodeMirror 6 with YAML language support, line numbers, bracket matching, and error markers. Integrate with Splunk theme (use one-dark for dark mode, default for light). Debounce onChange at 300ms. Show validation errors as inline decorations. | Restrictions: Install @codemirror/lang-yaml, @codemirror/view, @codemirror/state, @codemirror/theme-one-dark as dependencies. Do not use CodeMirror 5. | _Leverage: @codemirror/lang-yaml | Success: Editor renders YAML with syntax highlighting, errors show inline, onChange fires debounced, theme matches Splunk light/dark._

## Phase 5: Templates + Save + Integration

- [ ] 5.1. Implement TemplateGallery (modal with template cards)
  - Files: react/packages/urc-builder/src/dialogs/TemplateGallery.tsx, react/packages/urc-builder/src/utils/templates.ts
  - Modal with card grid: Generic, GitHub, ServiceNow, Infoblox, Azure, JSONPlaceholder
  - Each card: icon, name, auth type badge, pagination type badge, brief description
  - Select applies template to form state, "Start from Scratch" creates empty
  - templates.ts: Bundled manifest YAML strings from ucc/urc_app/tests/manifests/
  - Purpose: Fastest path to a working connector — start from a known-good template
  - _Leverage: @splunk/react-ui Modal, Card, CardLayout, Badge, Button_
  - _Requirements: R6_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer | Task: Build TemplateGallery as a Modal with CardLayout grid. Create templates.ts importing the 5 test manifest YAML files as strings plus a Generic blank template. Each Card shows template name, auth type badge, pagination type badge, and 1-line description. Selecting a card parses YAML into form state. | Restrictions: Templates are static strings bundled at build time — no runtime fetch. Use the existing manifest files from ucc/urc_app/tests/manifests/. | _Leverage: ucc/urc_app/tests/manifests/*.yaml, @splunk/react-ui Modal, Card, CardLayout, Badge | Success: Modal opens, shows 6 template cards. Selecting GitHub template populates form with BearerAuth + CursorPagination + SubstreamPartitionRouter._

- [ ] 5.2. Implement SaveDialog (modal with input config + deploy)
  - Files: react/packages/urc-builder/src/dialogs/SaveDialog.tsx
  - Modal: input name, account selector, interval, index selector, sourcetype
  - Smart defaults: sourcetype auto-generated from stream names (urc:{name}), interval 300
  - Save button calls api.saveInput, shows success with link to Inputs list
  - Edit mode: pre-fills from existing input config
  - Purpose: Bridge from builder to Splunk modular input creation
  - _Leverage: @splunk/react-ui Modal, ControlGroup, Text, Select, Number, Button, Message_
  - _Requirements: R7_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React developer with Splunk knowledge | Task: Build SaveDialog modal with fields for input name, account (dropdown from listAccounts), interval (Number, default 300), index (dropdown from listIndexes), sourcetype (auto-generated from stream names). Save button serializes form to manifest YAML, calls saveInput API. Show success Message with link to the UCC inputs page. | Restrictions: Must validate input name (alphanumeric + underscore only). Must handle duplicate name error gracefully. | _Leverage: @splunk/react-ui Modal, ControlGroup, Text, Select, Number, Button, Message | Success: Fill form → Save → input created in Splunk → success message with link._

- [ ] 5.3. Implement build pipeline Taskfile tasks
  - Files: Taskfile.yml (modify existing)
  - Add tasks: ucc:build-ui, ucc:merge-ui, ucc:build-full
  - ucc:build-ui: cd react && yarn workspace @splunk/urc-app build
  - ucc:merge-ui: copy builder.js, builder.html, nav/views XML into ucc/output/urc_app/
  - ucc:build-full: ucc:build → ucc:build-ui → ucc:merge-ui
  - Purpose: One-command build for the complete app
  - _Leverage: existing Taskfile.yml UCC tasks_
  - _Requirements: R8_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: DevOps/build engineer | Task: Add three tasks to Taskfile.yml: ucc:build-ui (webpack build of urc-app package), ucc:merge-ui (copy React build output into UCC output directory, overlay nav.xml), ucc:build-full (runs ucc:build + ucc:build-ui + ucc:merge-ui in sequence). | Restrictions: Must not break existing ucc:build, ucc:link, ucc:dev tasks. Merge must preserve all existing UCC-generated files. | _Leverage: existing Taskfile.yml tasks (ucc:build, ucc:link, ucc:dev) | Success: `task ucc:build-full` produces a complete app package with both UCC config pages and the builder page._

- [ ] 5.4. End-to-end integration test
  - Verify: task ucc:build-full succeeds, app loads in Splunk, builder page renders, select template, test connection, save input, verify input in Splunk
  - Fix any integration issues discovered
  - Purpose: Prove the full flow works before declaring the spec complete
  - _Requirements: All_
  - _Prompt: Implement the task for spec urc-guided-ui, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA engineer | Task: Run the full integration flow: task ucc:build-full, verify app loads in Splunk dev container, navigate to builder page, select JSONPlaceholder template, test connection (should return 20 records), save as a new input, verify input appears in UCC inputs list, search for events in the target index. Document any issues found and fix them. | Restrictions: Must use the dev Splunk container (task dev:up). Do not skip any steps. | _Leverage: task ucc:build-full, task dev:up, task dev:refresh | Success: Complete flow works: build → load → template → test → save → events in Splunk._
