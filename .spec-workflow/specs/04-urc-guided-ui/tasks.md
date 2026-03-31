# Tasks — URC Guided UI (Schema-Driven)

## Phase 1: Foundation — Scaffold + State + Schema Engine

- [x] 1.1. Scaffold React packages (urc-builder + urc-app) + Splunk nav/view integration
  - Implemented via `task ucc:add-react APP_NAME=urc_app` which runs:
    1. `react/bin/splunk-create-noninteractive.mjs component UrcBuilder` — scaffolds component library
    2. `react/bin/splunk-create-noninteractive.mjs app UrcApp` — scaffolds Splunk app wrapper
    3. `react/bin/adapt-for-ucc.mjs` — applies 6 UCC overlay transformations (webpack, deps, entry point, nav.xml, view XML, HTML template)
  - Also created: `@splunk/create` as devDep, non-interactive scaffolding tooling, `ucc:add-react` Taskfile task
  - Updated golden path docs, .gitignore for react source tracking
  - Success: `cd react && yarn workspace @splunk/urc-app build` produces stage/appserver/static/pages/builder.js

- [ ] 1.2. Bundle JSON Schema as static asset
  - Files: react/packages/urc-builder/src/schema/schema.json, build script to convert YAML→JSON
  - Convert declarative_component_schema.yaml to JSON at build time
  - Import as `import schema from './schema/schema.json'` in urc-builder
  - Purpose: Schema drives all form rendering — must be available at runtime without fetch
  - _Requirements: R1_

- [ ] 1.3. Implement useBuilderState hook
  - Files: react/packages/urc-builder/src/state/useBuilderState.ts
  - Manifest dict IS the form state — no intermediate StreamConfig types
  - Methods: setField(path, value), addStream, removeStream, mode toggle
  - js-yaml.dump(manifest) ↔ js-yaml.load(yamlString) for form/YAML sync
  - Session storage persistence for crash recovery
  - _Requirements: R1, R2, R4_

- [ ] 1.4. Implement SchemaFormControl (the core field renderer)
  - Files: react/packages/urc-builder/src/schema/SchemaFormControl.tsx, schemaUtils.ts
  - Auto-renders any field from its JSON Schema definition
  - Maps: string→Text, enum→Select, boolean→Switch, number→Number, object→KeyValueEditor, array→ArrayEditor
  - Reads description, required, default, examples from schema
  - Wraps each field in ControlGroup with label + help text
  - _Requirements: R1, R5_

- [ ] 1.5. Implement OneOfSelector (polymorphic field renderer)
  - Files: react/packages/urc-builder/src/schema/OneOfSelector.tsx
  - Dropdown for oneOf/anyOf fields (auth types, pagination strategies, retriever types, decoders)
  - On type selection, renders sub-schema fields via recursive SchemaFormControl
  - Shows type descriptions in dropdown options
  - _Requirements: R1.2, R1.3_

- [ ] 1.6. Implement SchemaFormRemainingFields (catch-all)
  - Files: react/packages/urc-builder/src/schema/SchemaFormRemainingFields.tsx
  - Given schema + set of already-placed paths, renders everything else
  - Collapsible "Advanced" section — nothing hidden
  - _Requirements: R1.4, R1.7_

## Phase 2: Layout + Cards

- [ ] 2.1. Implement ConnectorBuilder root component (3-column grid)
  - Files: react/packages/urc-builder/src/ConnectorBuilder.tsx
  - CSS Grid: 240px | 1fr | 280px, with bottom test panel
  - Slots for StreamSidebar, card area, HelpPanel, TestPanel
  - Form/YAML mode toggle
  - _Requirements: R8_

- [ ] 2.2. Implement StreamSidebar
  - Files: react/packages/urc-builder/src/panels/StreamSidebar.tsx
  - Stream list with active highlight, add/remove/rename
  - Form|YAML toggle, Templates button
  - _Requirements: R2_

- [ ] 2.3. Implement HelpPanel (schema-driven)
  - Files: react/packages/urc-builder/src/panels/HelpPanel.tsx
  - Shows description from focused field's schema — no hand-written helpContent.ts
  - Quick Start card when nothing focused
  - _Requirements: R5_

- [ ] 2.4. Implement curated cards (layout grouping)
  - Files: react/packages/urc-builder/src/cards/EndpointCard.tsx, AuthCard.tsx, PaginationCard.tsx, ExtractionCard.tsx, IncrementalCard.tsx, TransformCard.tsx, ErrorHandlingCard.tsx, AdvancedCard.tsx
  - Each card declares which schema paths it places, delegates to SchemaFormControl
  - AdvancedCard uses SchemaFormRemainingFields for the catch-all
  - _Requirements: R1_

## Phase 3: Widget Overrides + Test Panel

- [ ] 3.1. Implement widget overrides (JinjaInput, KeyValueEditor, ArrayEditor)
  - Files: react/packages/urc-builder/src/schema/JinjaInput.tsx, KeyValueEditor.tsx, ArrayEditor.tsx, widgetRegistry.ts
  - JinjaInput: template-aware text input with Jinja interpolation hints
  - KeyValueEditor: for object-typed fields (headers, params)
  - ArrayEditor: for array-typed fields with add/remove (FormRows)
  - widgetRegistry: maps schema patterns to custom widgets (~15 overrides)
  - _Requirements: R1, R5_

- [ ] 3.2. Implement api.ts REST helper module
  - Files: react/packages/urc-builder/src/utils/api.ts
  - Functions: testConnection, listAccounts, listIndexes, saveInput, loadInput
  - Uses @splunk/splunk-utils for CSRF tokens and REST URLs
  - _Requirements: R3, R7_

- [ ] 3.3. Implement TestPanel
  - Files: react/packages/urc-builder/src/panels/TestPanel.tsx
  - Account selector, Test button, TabLayout: Records/Request/Response/Schema
  - Auto-detect field_path from response
  - _Requirements: R3_

## Phase 4: YAML Editor + Templates + Save

- [ ] 4.1. Implement YamlEditor (CodeMirror 6)
  - Files: react/packages/urc-builder/src/editor/YamlEditor.tsx
  - YAML syntax highlighting, inline error markers, Splunk theme integration
  - Debounced onChange → js-yaml.load → updates manifest state
  - _Requirements: R4_

- [ ] 4.2. Implement TemplateGallery
  - Files: react/packages/urc-builder/src/dialogs/TemplateGallery.tsx, utils/templates.ts
  - Modal with cards: Generic, GitHub, ServiceNow, Infoblox, Azure, JSONPlaceholder
  - Templates bundled from ucc/urc_app/tests/manifests/*.yaml
  - _Requirements: R6_

- [ ] 4.3. Implement SaveDialog
  - Files: react/packages/urc-builder/src/dialogs/SaveDialog.tsx
  - Input name, account, interval, index, sourcetype fields
  - Calls api.saveInput, shows success with link to Inputs list
  - _Requirements: R7_

## Phase 5: Build Pipeline + E2E

- [ ] 5.1. Implement build pipeline Taskfile tasks
  - Files: Taskfile.yml (modify existing)
  - Add tasks: ucc:build-ui, ucc:merge-ui, ucc:build-full
  - ucc:merge-ui copies stage/ artifacts into UCC output
  - _Requirements: R8_

- [ ] 5.2. End-to-end integration test
  - Verify: task ucc:build-full, app loads in Splunk, builder page renders
  - Select template, test connection, save input, verify in Splunk
  - _Requirements: All_
