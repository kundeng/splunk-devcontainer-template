# Research: Airbyte Connector Builder UI

Date: 2026-03-30
Source: Airbyte Platform repo (airbyte-webapp/src/area/connectorBuilder), Airbyte docs

## Overall Architecture

Three-panel layout inside a single-page app:
1. **Left Sidebar** (~200px) - Navigation + UI/YAML toggle
2. **Center Panel** (resizable, main) - Either visual form Builder OR Monaco YAML editor
3. **Right Panel** (resizable, ~33% default) - StreamTestingPanel (test + results)

Panels use `<ResizablePanels>` component with configurable flex ratios and min widths.

## Top-Level Component Hierarchy

```
ConnectorBuilderEditPage
  ConnectorBuilderFormManagementStateProvider   (modal state)
    ConnectorBuilderLocalStorageProvider         (persisted preferences)
      ConnectorBuilderResolveProvider            (manifest resolution)
        ConnectorBuilderSchemaProvider           (JSON Schema for form)
          SchemaForm (react-hook-form + zod)     (entire form state)
            ConnectorBuilderFormStateProvider     (builder-specific state)
              ConnectorBuilderTestReadProvider    (test read API state)
                MenuBar                          (top bar: name, publish, etc.)
                Panels                           (ResizablePanels)
                  LEFT:  mode=yaml ? YamlManifestEditor : Builder
                  RIGHT: StreamTestingPanel
```

## State Management

- **Single react-hook-form** wraps the entire `BuilderState`:
  ```ts
  interface BuilderState {
    name: string;
    mode: "ui" | "yaml";
    yaml: string;
    customComponentsCode?: string;
    view: { type: "global" } | { type: "inputs" } | { type: "components" } | StreamId;
    streamTab: "requester" | "schema" | "polling" | "download";
    testStreamId: StreamId;
    testingValues: object;
    manifest: ConnectorManifest;   // THE source of truth
    generatedStreams: Record<string, DeclarativeStream[]>;
  }
  ```
- The `manifest` field IS the declarative YAML manifest (as JSON). Form controls bind directly to `manifest.streams[i].retriever.requester.url` etc.
- Schema is derived from the Airbyte `declarative_component_schema.yaml` (OpenAPI-like JSON Schema), loaded at build time.
- `useBuilderWatch(path)` = thin wrapper around react-hook-form's `watch()`.

## Dual-Mode (Form <-> YAML) Switching

- **UiYamlToggleButton**: Simple two-segment toggle ("UI" | "YAML"), lives in the Sidebar.
- `toggleUI(newMode)` in ConnectorBuilderFormState triggers the switch:
  - **UI -> YAML**: Converts manifest JSON to YAML string, sets `mode: "yaml"`.
  - **YAML -> UI**: Parses YAML, resolves manifest via API call, updates form state, sets `mode: "ui"`.
  - Disabled while resolving (shows tooltip "Resolving stream list...").
- **YamlManifestEditor**: Monaco editor with `js-yaml` validation. On YAML change, debounced (500ms) `resolveManifest()` call updates the `manifest` form state in the background. Markers shown for YAML syntax errors.
- When in YAML mode, the center panel shows `YamlManifestEditor` (with optional `components.py` tab for custom Python components). The sidebar still renders but shows TestingValuesMenu instead of stream list.

## Left Sidebar (BuilderSidebar)

When in UI mode, the sidebar contains:
1. **UI/YAML Toggle** (top)
2. **Connector Name** input
3. **Base connector info** (if forked)
4. **Saving indicator**
5. **Navigation buttons** (vertical list):
   - Global Configuration (icon: parameters)
   - User Inputs (icon: user, shows count)
   - Custom Components (icon: wrench, conditional)
6. **Streams** section:
   - Header: "Streams (N)" + Add Stream button
   - List of `StreamViewButton` items (click to select, error/warning indicators)
7. **Dynamic Streams** section (if enabled):
   - Similar pattern, with expandable generated stream children

Each `ViewSelectButton` sets `view` in form state, which controls what the center panel renders.

## Center Panel - Builder (UI Mode)

`Builder` component renders based on `view.type`:
- `"global"` -> **GlobalConfigView**: Cards for `check`, `concurrency_level`, `api_budget`
- `"inputs"` -> **InputsView**: User input parameter management
- `"components"` -> **ComponentsView**: Custom Python code editor
- `"stream"` / `"generated_stream"` -> **StreamConfigView**
- `"dynamic_stream"` -> **DynamicStreamConfigView**

### StreamConfigView (the main configuration)

Has sub-tabs at the top: "Stream Configuration" | "Schema" (for sync streams), plus "Polling" and "Download" tabs for async streams.

**"Requester" tab** renders a vertical stack of Cards (StreamCard components):

1. **URL & Record Selection Card**:
   - `retriever.requester.url` (or url_base + path)
   - `retriever.requester.http_method`
   - `retriever.decoder`
   - `retriever.record_selector` (with DpathExtractor field_path)
   - `primary_key`

2. **Authentication Card**:
   - `retriever.requester.authenticator` (polymorphic: ApiKey, Bearer, Basic, OAuth, JWT, etc.)
   - Non-advanced fields shown by default, rest collapsible

3. **Request Options Card**:
   - `request_parameters`, `request_headers`, `request_body_json`, `request_body_data`

4. **Pagination Card**:
   - `retriever.paginator` (polymorphic: offset, cursor, page-based)

5. **Incremental Sync Card**:
   - `incremental_sync` (DatetimeBasedCursor with cursor_field, formats, start/end, step, granularity)

6. **Partitioning Card**:
   - `retriever.partition_router` (SubstreamPartitionRouter, ListPartitionRouter, etc.)

7. **Error Handling Card**:
   - `retriever.requester.error_handler`

8. **Transformations Card**:
   - `transformations` array

9. **Collapsed/Advanced Card**:
   - `SchemaFormRemainingFields` for any fields not explicitly placed above

**"Schema" tab** renders an inline JSON schema editor with auto-import from detected schema.

**All form controls are schema-driven** via `<SchemaFormControl path="...">`. The JSON Schema from `declarative_component_schema.yaml` defines field types, enums, oneOf polymorphism, defaults, required fields, etc. Custom field overrides (`overrideByObjectField`, `overrideByFieldSchema`) handle special cases like Jinja template inputs, OAuth client ID, cursor fields.

## Right Panel - StreamTestingPanel

### Top bar:
- **StreamSelector** dropdown (pick which stream to test)
- **AdvancedTestSettings** (record/page/slice limits, custom state injection)

### StreamTester (main content):
- **StreamTestButton**: "Test" button (with stale indicator, queue/cancel support)
- **ResizablePanels** (horizontal, stacked):
  1. **ResultDisplay** (primary):
     - SliceSelector (if multiple slices/partitions)
     - **PageDisplay** -> **TabbedDisplay** with these tabs:
       - **Records** (count) - JSON view or Table view toggle (RecordTable)
       - **Request** - formatted HTTP request (url, headers, params)
       - **Response** - formatted HTTP response body
       - **Schema** - SchemaDiffView (inferred vs declared, with conflict indicator)
       - **State** - cursor/checkpoint state with "Import State" button
     - Paginator (if multiple pages)
  2. **LogsDisplay** (collapsible splitter, shows info/warning/error counts)
  3. **AuxiliaryRequestsDisplay** (collapsible splitter, for auth token requests etc.)

### TabbedDisplay implementation:
- Uses `@headlessui/react` Tab components
- Simple `TabGroup > TabList > Tab` + `TabPanels > TabPanel` pattern
- `TabData` interface: `{ key, title, content, data-testid }`

## AI Assistant (Assist)

Located in `Builder/Assist/` directory.

### Architecture:
- **AssistButton** / **AssistConfigButton**: UI triggers
- **assist.ts**: Custom hooks that call Airbyte's assist API endpoints via proxy
- Input: `docs_url` or `openapi_spec_url` + current manifest context
- Output: `BuilderAssistManifestResponse` with `ManifestUpdate` containing all field values

### API Endpoints (each returns manifest_update):
- `find_url_base` - discovers base URL
- `find_auth` - identifies auth method
- `create_stream` - generates full stream config
- `find_stream_pagination` - detects pagination pattern
- `find_stream_metadata` - finds metadata (primary key, etc.)
- `find_response_structure` - detects record selector
- `find_request_parameters` - identifies required params
- `find_stream_incremental_sync` - configures cursor
- `find_streams` - discovers available streams from docs

### Workflow:
1. User provides API docs URL or OpenAPI spec URL
2. AI runs server-side, returns partial manifest updates
3. `convertToAssistFormValuesSync()` merges AI output into manifest structure
4. Form fields are populated automatically
5. User reviews and can trigger per-section assists (auth, pagination, etc.)

## Key Technology Choices

- **React Hook Form** for all form state (single form wrapping entire builder)
- **Monaco Editor** for YAML editing (with js-yaml validation + error markers)
- **@headlessui/react** for tabs and other UI primitives
- **JSON Schema-driven forms** (`SchemaFormControl`) - forms auto-generated from OpenAPI/JSON Schema
- **ResizablePanels** for the split layout
- **Zod** for additional validation overrides
- **Jinja template syntax** in string fields (interpolation_context in schema)
- **react-intl** for i18n
- **@tanstack/react-query** for API calls (test reads, assist, resolve)
- **Web Worker** (SchemaWorker.ts) for background schema processing

## Applicability to Splunk UCC Custom UI

### What maps well to UCC custom tab/page:
- The **stream configuration card layout** (vertical stack of collapsible sections) fits inside a UCC custom tab
- **Test panel with tabs** (Request/Response/Records/Schema) can be a standalone React component
- **Stream selector** in sidebar maps to UCC's built-in input list
- **JSON/Table toggle** for records display is straightforward

### What needs adaptation:
- **Dual-mode (Form <-> YAML)**: UCC has no YAML concept. Could do Form <-> JSON toggle instead, or skip entirely
- **Three-panel resizable layout**: UCC pages are single-panel. Could use CSS grid within a custom page, or put test panel in a slide-out drawer
- **Schema-driven forms from manifest schema**: UCC uses globalConfig.json for its forms. Our component schema would need its own form renderer, or we hand-code forms
- **Monaco editor**: Heavy dependency (~2MB). Consider CodeMirror 6 or a simple textarea for JSON editing
- **AI Assistant**: Requires server-side AI proxy. Out of scope for Splunk add-on initially, but the per-section assist pattern (assist button next to each config card) is a good UX pattern to copy

### Realistic scope for Splunk UCC custom UI:
1. **Stream list** in UCC's native left nav (one "input" per stream)
2. **Stream config form** as a custom UCC tab (cards for URL, auth, pagination, record extraction, incremental sync, error handling)
3. **Test panel** as a second custom tab or embedded panel (Records/Request/Response/Schema tabs)
4. **Form <-> JSON toggle** using a lightweight editor (CodeMirror or textarea)
5. **No Monaco, no AI assist, no resizable panels** in v1
6. **Schema validation** via Zod or JSON Schema on the client side
