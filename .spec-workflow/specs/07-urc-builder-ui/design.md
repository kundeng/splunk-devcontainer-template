# Design: URC Builder UI — Stream Dashboard & Schema-Driven Editor

## Overview

Two custom React screens (Stream Dashboard + Stream Builder) that complement UCC's auto-generated Configuration/Inputs pages. Built in the existing `react/` monorepo using `@splunk/react-ui` v5.9.0 components. Backend changes are minimal: activate the dead `validate.py`, enrich `emit()` with a `level` parameter, and add a `tags` field to `globalConfig.json`.

The React pages communicate with Splunk exclusively through UCC's standard REST endpoints plus the existing custom `test_connection` endpoint. No new REST endpoints are required.

## Code Reuse Analysis

### Existing Components to Leverage

- **`react/packages/urc-builder/`** — existing component library scaffold. Currently a placeholder counter component. We replace its contents with the real builder components. The monorepo plumbing (webpack, yarn workspaces, Splunk integration) is already working.
- **`react/packages/urc-app/`** — existing Splunk app wrapper. We add new page entry points under `src/main/webapp/pages/`. Webpack auto-discovers them.
- **`@splunk/react-ui` v5.9.0** — 88 components available. Key ones we'll use: Table, TabLayout, Select, Text (input), TextArea, Switch, ControlGroup, Button, Message, Modal, CollapsiblePanel, WaitSpinner, Heading, ColumnLayout, Chip, Menu, Tooltip, Code (for JSON preview).
- **`@splunk/themes`** — variables and mixins for light/dark theme support. Already imported in existing styled-components.
- **`@splunk/splunk-utils`** — `createRESTURL()` for building Splunk REST API URLs with proper proxying, `getUserTheme()` for theme detection.
- **`ucc/urc_app/package/lib/urc/validate.py`** — existing dead code with Pydantic schema validation. We activate it and extend with capability detection.
- **`ucc/urc_app/package/lib/urc/registry.py`** — `REGISTRY` dict of 50 supported types. Validation reads this to check manifest types.
- **`ucc/urc_app/package/bin/urc_test_connection.py`** — existing REST endpoint for dry-run collection. We extend it to include validation results and support a `mode=validate` parameter for validation-only calls.
- **`ucc/urc_app/schema/declarative_component_schema.yaml`** — component schema with titles, descriptions, enums, defaults, required fields. Consumed at build time to generate form configuration.

### Integration Points

- **UCC REST endpoints** — all CRUD operations use UCC's standard `AdminExternalHandler` endpoints. No custom REST handlers for configuration storage.
- **`/services/data/inputs/urc_app_input`** — GET (list/read), POST (create), PUT (update), DELETE for stream management.
- **`/services/urc_app_account`** — GET (list accounts) for populating account dropdowns.
- **`/services/data/indexes`** — GET (list indexes) for index dropdown.
- **`/services/urc_app/test_connection`** — POST for test/validate (extended with mode parameter).
- **`/services/urc_app_settings`** — GET log level (for debug state display).

## Architecture

### Navigation & Page Structure

```
URC App Navigation (default.xml)
├── streams     ← NEW: Stream Dashboard (Screen 1)
├── builder     ← MODIFIED: Stream Builder (Screen 2), accessed via streams row click
├── configuration ← UCC-generated (Accounts, Logging, Proxy) — untouched
└── search       ← Splunk built-in — untouched
```

The "streams" page is the default landing page. "builder" is accessed via navigation from streams (with `?input=<name>` query param for edit, no param for create). Both are separate webpack entry points that load independently.

### Component Architecture

```
react/packages/urc-builder/src/
├── index.ts                          # Public exports
├── types.ts                          # Shared TypeScript interfaces
├── schema/
│   └── form-schema.ts                # Build-time generated form config from YAML schema
│
├── context/
│   ├── BuilderContext.tsx             # React Context + useReducer for builder form state
│   └── DashboardContext.tsx           # React Context for dashboard filters/selection
│
├── services/
│   ├── splunk-api.ts                 # REST client wrapping createRESTURL + fetch
│   ├── manifest-serializer.ts        # Form state ↔ Airbyte manifest YAML (bidirectional)
│   └── manifest-validator.ts         # Client-side pre-validation (type checking vs schema)
│
├── dashboard/
│   ├── StreamDashboard.tsx           # Screen 1: table + filters + summary cards
│   ├── StreamTable.tsx               # Table component with sorting, row actions
│   ├── TagFilter.tsx                 # Tag filter bar (chips + dropdown)
│   ├── SummaryCards.tsx              # Active/Error/Total stream counts
│   └── BulkActions.tsx              # Enable/Disable/Delete/Tag selected streams
│
├── builder/
│   ├── StreamBuilder.tsx             # Screen 2: tabbed form container
│   ├── tabs/
│   │   ├── ConnectionTab.tsx         # Tab 1: Account, Base URL, HTTP method
│   │   ├── DataMappingTab.tsx        # Tab 2: Schema-driven form sections
│   │   ├── SplunkOutputTab.tsx       # Tab 3: Index, sourcetype, timestamp picker
│   │   ├── ScheduleTagsTab.tsx       # Tab 4: Interval, tags, enable/disable, debug toggle
│   │   └── TestPreviewTab.tsx        # Tab 5: Test button, results, log viewer
│   │
│   ├── form/
│   │   ├── SchemaSection.tsx         # Renders a collapsible section from schema definition
│   │   ├── SchemaField.tsx           # Single field renderer (dispatches by type)
│   │   ├── TypeSelector.tsx          # Dropdown that switches component type + re-renders fields
│   │   ├── NestedObjectField.tsx     # Renders $ref nested objects recursively
│   │   └── ValidationBanner.tsx      # Inline warning/error display for unsupported types
│   │
│   └── test/
│       ├── TestPanel.tsx             # Test execution + result display
│       ├── RecordPreview.tsx         # JSON tree view of sample records
│       ├── FieldDetector.tsx         # Detected fields with types
│       └── TimestampPicker.tsx       # Timestamp candidate selection from test results
│
└── styles/
    └── shared.ts                     # Shared styled-components using @splunk/themes
```

### Page Entry Points (Webpack auto-discovered)

```
react/packages/urc-app/src/main/webapp/pages/
├── builder/
│   ├── index.tsx                     # Existing — mounts StreamBuilder
│   └── Styles.ts
└── streams/                          # NEW
    ├── index.tsx                     # Mounts StreamDashboard
    └── Styles.ts
```

### State Management

**BuilderContext** — React Context + `useReducer` for the stream builder form:

```typescript
interface BuilderState {
  mode: 'create' | 'edit';
  inputName: string | null;
  
  // Tab 1: Connection
  account: string;
  baseUrl: string;
  httpMethod: 'GET' | 'POST';
  
  // Tab 2: Data Mapping (schema-driven, dynamic shape)
  streams: StreamConfig[];  // array of stream definitions
  
  // Tab 3: Splunk Output
  index: string;
  sourcetype: string;
  sourceOverride: string;
  hostOverride: string;
  timestampField: string | null;
  timestampFormat: string;
  
  // Tab 4: Schedule & Tags
  interval: number;
  tags: string;
  enabled: boolean;
  debug: boolean;
  
  // Tab 5: Test (ephemeral, not saved)
  testResult: TestResult | null;
  testLoading: boolean;
  
  // Validation
  validationResults: ValidationResult[];
  isDirty: boolean;
}

type BuilderAction =
  | { type: 'SET_FIELD'; path: string; value: any }
  | { type: 'LOAD_INPUT'; input: SavedInput }
  | { type: 'SET_TEST_RESULT'; result: TestResult }
  | { type: 'SET_VALIDATION'; results: ValidationResult[] }
  | { type: 'RESET' };
```

**DashboardContext** — lighter context for dashboard filters:

```typescript
interface DashboardState {
  inputs: InputSummary[];
  selectedTags: string[];
  selectedStatus: 'all' | 'active' | 'error' | 'disabled';
  selectedRows: string[];
  loading: boolean;
}
```

### Data Flow

```
Stream Dashboard                        Stream Builder
┌─────────────┐                        ┌─────────────────────┐
│ GET inputs   │──────────────────────▶│ Parse manifest YAML  │
│ from UCC API │                       │ into form state      │
│              │◀── navigate ──────────│ (manifest-serializer)│
│ Display      │                       │                      │
│ table +      │                       │ Edit in schema-driven│
│ filters      │                       │ form                 │
└─────────────┘                        │                      │
                                       │ On Save:             │
                                       │ Serialize form state │
                                       │ → manifest YAML      │
                                       │ → POST to UCC API    │
                                       └─────────────────────┘
                                              │
                                              ▼ On Test:
                                       ┌─────────────────────┐
                                       │ POST to              │
                                       │ test_connection      │
                                       │ endpoint             │
                                       │                      │
                                       │ Returns: validation  │
                                       │ results + sample     │
                                       │ records              │
                                       └─────────────────────┘
```

## Components and Interfaces

### Component 1: `manifest-serializer.ts`

- **Purpose:** Bidirectional conversion between form state and Airbyte declarative manifest YAML.
- **Interfaces:**
  - `formStateToManifest(state: BuilderState): string` — serializes form state to YAML string
  - `manifestToFormState(yaml: string, defaults: Partial<BuilderState>): BuilderState` — parses YAML and populates form state
- **Dependencies:** `js-yaml` for YAML parse/stringify
- **Design notes:** The serializer is the single source of truth for how form fields map to manifest paths. Example mapping:
  - `state.streams[0].retriever.requester.url` → `streams[0].retriever.requester.url`
  - `state.streams[0].pagination.type` → `streams[0].retriever.paginator.pagination_strategy.type`
  - `state.account` + `state.baseUrl` are NOT in the manifest — they're stored as separate UCC input fields

### Component 2: `form-schema.ts` (build-time generated)

- **Purpose:** Static TypeScript module generated from `declarative_component_schema.yaml` at build time. Provides the form configuration that `SchemaSection` and `SchemaField` consume.
- **Interfaces:**
  ```typescript
  interface FormFieldDef {
    key: string;           // property name in schema
    label: string;         // from schema title
    helpText: string;      // from schema description
    type: 'text' | 'number' | 'boolean' | 'enum' | 'object' | 'array' | 'template';
    required: boolean;
    defaultValue?: any;
    enumValues?: { value: string; label: string }[];
    examples?: string[];
    nestedFields?: FormFieldDef[];       // for $ref objects
    interpolationContext?: string[];      // marks field as Jinja2 template
  }
  
  interface ComponentFormDef {
    type: string;           // e.g. "ApiKeyAuthenticator"
    category: string;       // "authenticator" | "pagination" | "extractor" | etc.
    label: string;          // from schema title
    description: string;    // from schema description
    fields: FormFieldDef[];
    supported: boolean;     // checked against REGISTRY
  }
  
  // Exported: grouped by category
  export const AUTHENTICATORS: ComponentFormDef[];
  export const PAGINATORS: ComponentFormDef[];
  export const EXTRACTORS: ComponentFormDef[];
  export const TRANSFORMATIONS: ComponentFormDef[];
  export const ERROR_HANDLERS: ComponentFormDef[];
  export const INCREMENTAL_CURSORS: ComponentFormDef[];
  export const DECODERS: ComponentFormDef[];
  ```
- **Generation:** A Node.js script reads the YAML schema at build time, processes definitions, and writes `form-schema.ts`. Runs as a `prebuild` step in package.json. This avoids runtime YAML parsing and gives us TypeScript type safety.

### Component 3: `splunk-api.ts`

- **Purpose:** Thin REST client wrapping Splunk's `createRESTURL()` with proper session tokens and error handling.
- **Interfaces:**
  ```typescript
  // All return typed promises, handle Splunk's XML/JSON response formats
  listInputs(): Promise<InputSummary[]>
  getInput(name: string): Promise<SavedInput>
  createInput(data: InputPayload): Promise<void>
  updateInput(name: string, data: InputPayload): Promise<void>
  deleteInput(name: string): Promise<void>
  toggleInput(name: string, disabled: boolean): Promise<void>
  listAccounts(): Promise<AccountSummary[]>
  listIndexes(): Promise<string[]>
  testConnection(manifest: string, accountName: string, baseUrl: string): Promise<TestResult>
  validateManifest(manifest: string): Promise<ValidationResult[]>
  ```
- **Dependencies:** `@splunk/splunk-utils` for `createRESTURL`, browser `fetch` API.
- **Design notes:** Splunk REST API returns XML by default; we add `output_mode=json` to all calls. Session token is inherited from the page context via cookies (Splunk handles this). The module handles Splunk's response envelope format (`{ entry: [...] }`).

### Component 4: `SchemaSection.tsx` + `SchemaField.tsx`

- **Purpose:** Generic, recursive form renderer that takes a `ComponentFormDef` and renders appropriate Splunk UI form fields.
- **Interfaces:**
  - `<SchemaSection def={componentFormDef} value={object} onChange={handler} validationResults={results} />`
  - `<SchemaField def={fieldDef} value={any} onChange={handler} />`
- **Rendering rules:**
  - `type: 'text'` → `<Text />` (from @splunk/react-ui)
  - `type: 'number'` → `<Number />` 
  - `type: 'boolean'` → `<Switch />`
  - `type: 'enum'` → `<Select />` with options from `enumValues`
  - `type: 'template'` → `<Text />` with a "{{ }}" indicator showing it supports Jinja2 interpolation
  - `type: 'object'` → `<CollapsiblePanel>` with recursive `SchemaField` for each nested field
  - `type: 'array'` → list with Add/Remove buttons, each item rendered recursively
- **Dependencies:** `@splunk/react-ui` form components, `form-schema.ts` for definitions.

### Component 5: `TypeSelector.tsx`

- **Purpose:** Dropdown that lets the user choose a component type (e.g., which pagination strategy). When the selection changes, the form fields below it re-render to match the new type's schema.
- **Interfaces:**
  - `<TypeSelector category="paginators" value={currentType} onChange={handler} />`
- **Design notes:** Groups options into "Supported" and "Unsupported" sections in the dropdown. Unsupported types show a warning icon and are selectable (for viewing/editing existing configs) but trigger a `ValidationBanner`.

### Component 6: `TimestampPicker.tsx`

- **Purpose:** After a test fetch returns sample records, analyze all fields across records, detect datetime candidates, and let the user pick which becomes `_time`.
- **Detection logic:**
  - Field name heuristic: matches patterns like `*_at`, `*_date`, `*_time`, `timestamp`, `created`, `updated`, `modified`
  - Value heuristic: attempt `Date.parse()` on string values; check for epoch integers > 946684800 (year 2000)
  - Show each candidate with: field name, sample value, parsed preview
- **Interfaces:**
  - `<TimestampPicker records={sampleRecords} value={selectedField} onChange={handler} />`

## Data Models

### InputSummary (Dashboard table row)

```typescript
interface InputSummary {
  name: string;
  account: string;
  baseUrl: string;
  interval: number;
  index: string;
  sourcetype: string;
  tags: string;          // comma-separated
  disabled: boolean;
  manifest: string;      // raw YAML (not displayed, used for navigation to builder)
  debug: boolean;
}
```

### SavedInput (Full input for builder)

```typescript
interface SavedInput extends InputSummary {
  // Parsed from manifest for form population
  parsedManifest: Record<string, any>;
}
```

### ValidationResult

```typescript
interface ValidationResult {
  path: string;          // e.g. "streams[0].retriever.requester.authenticator"
  type: string;          // e.g. "OAuth2WithPKCE"
  severity: 'error' | 'warning' | 'info';
  message: string;       // e.g. "OAuth2 PKCE flow is not supported"
}
```

### TestResult

```typescript
interface TestResult {
  status: 'success' | 'error';
  message?: string;
  records: { stream: string; data: Record<string, any> }[];
  recordCount: number;
  validationResults: ValidationResult[];
  detectedFields: { name: string; type: string; sample: any }[];
  timestampCandidates: { field: string; sampleValue: string; parsedPreview: string }[];
}
```

### StreamConfig (form state for one stream in Data Mapping tab)

```typescript
interface StreamConfig {
  name: string;
  retriever: {
    type: string;
    requester: Record<string, any>;     // schema-driven fields
    recordSelector: Record<string, any>;
    paginator?: Record<string, any>;
  };
  incrementalSync?: Record<string, any>;
  transformations?: Record<string, any>[];
  eventTimestamp?: Record<string, any>;
}
```

## Backend Changes

### 1. Activate `validate.py` + capability detection

Extend the existing `validate_manifest()` in `ucc/urc_app/package/lib/urc/validate.py`:

```python
def validate_manifest_capabilities(manifest_yaml: str) -> list[dict]:
    """Validate manifest and check all component types against REGISTRY.
    
    Returns list of {path, type, severity, message} dicts.
    """
    results = []
    manifest = process_manifest(manifest_yaml)
    
    # Walk the manifest tree, find all 'type' fields
    _walk_types(manifest, results, path="")
    
    # Check specific unsupported patterns
    _check_auth_capabilities(manifest, results)
    
    return results

def _walk_types(obj, results, path):
    """Recursively find all type fields and check against REGISTRY."""
    if isinstance(obj, dict):
        type_name = obj.get("type")
        if type_name and type_name not in _SKIP_TYPES:
            if type_name not in REGISTRY:
                results.append({
                    "path": path,
                    "type": type_name,
                    "severity": "error",
                    "message": f"Unknown component type '{type_name}'"
                })
        for k, v in obj.items():
            _walk_types(v, results, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_types(v, results, f"{path}[{i}]")

_SKIP_TYPES = {"DeclarativeSource", "DeclarativeStream"}  # top-level types, not components

def _check_auth_capabilities(manifest, results):
    """Check for specific unsupported auth patterns."""
    # JWT with asymmetric algorithms
    # OAuth2 with PKCE fields
    # ... specific pattern checks
```

### 2. Extend `test_connection` endpoint

Add `mode=validate` support to the existing handler in `urc_test_connection.py`:

```python
def handlePOST(self, request_info):
    # ... existing payload parsing ...
    mode = payload.get("mode", "test")  # "validate" or "test"
    
    # Always run validation first
    from urc.validate import validate_manifest_capabilities
    validation_results = validate_manifest_capabilities(manifest_yaml)
    
    if mode == "validate":
        return _json_response(200, {
            "status": "success",
            "validation": validation_results,
        })
    
    # mode == "test": existing behavior + validation results
    # ... existing dry-run collection ...
    return _json_response(200, {
        "status": "success",
        "records": records,
        "record_count": len(records),
        "validation": validation_results,
    })
```

### 3. Debug logging in `emit()`

Modify `structured_logger.py`:

```python
def emit(level="info", **fields):
    parts = [f"{k}={_quote(v)}" for k, v in fields.items() if v is not None]
    if parts:
        msg = " ".join(parts)
        if level == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)
```

Modify AOP hooks in `structured_logger.py`:

```python
def after_hook(component_type, method_name, result, elapsed, context):
    # Always emit at INFO
    emit(action="call", component=component_type, method=method_name,
         elapsed=f"{elapsed:.3f}")
    # Emit verbose context at DEBUG
    if context and logger.isEnabledFor(logging.DEBUG):
        debug_fields = {k: str(v)[:200] for k, v in context.items()
                       if k in ('url', 'stream_slice', 'stream_name')}
        if debug_fields:
            emit(level="debug", action="call_detail", component=component_type,
                 method=method_name, **debug_fields)

def before_hook(component_type, method_name, context):
    """NEW: registered only when debug is enabled."""
    if logger.isEnabledFor(logging.DEBUG):
        emit(level="debug", action="enter", component=component_type,
             method=method_name)
```

Register the before hook in `engine.py`:

```python
register_hook("before", before_hook)  # new — only emits at DEBUG level
```

### 4. Tags and debug fields in `globalConfig.json`

Add two fields to the input entity:

```json
{
    "type": "text",
    "label": "Tags",
    "field": "tags",
    "help": "Comma-separated tags for organizing streams.",
    "required": false,
    "defaultValue": ""
},
{
    "type": "checkbox",
    "label": "Debug Logging",
    "field": "debug",
    "help": "Enable verbose debug logging for this stream.",
    "required": false,
    "defaultValue": false
}
```

Read in modular input handler (`urc_app_input_helper.py`):

```python
# In stream_events(), after setting log level from settings:
if input_item.get("debug") in ("1", "true", True):
    logger.setLevel(logging.DEBUG)
    logging.getLogger("urc.engine").setLevel(logging.DEBUG)
```

## Schema-to-Form Generation (Build-Time Script)

A Node.js script (`scripts/generate-form-schema.js`) runs at `prebuild`:

1. Reads `declarative_component_schema.yaml` via `js-yaml`
2. Walks `definitions`, groups by category (authenticator, paginator, etc.)
3. For each component definition:
   - Extracts `title`, `description`, `required` fields
   - For each property: extracts `title`, `description`, `type`, `enum`, `default`, `examples`
   - Resolves `$ref` pointers for nested objects
   - Checks `type` enum value against a hardcoded list of supported types (from REGISTRY)
4. Writes `form-schema.ts` with typed exports

This runs once at build time, not at runtime. The generated file is committed to the repo so it works without the build step during development.

## Splunk View Registration

### New view: `streams.xml`

```xml
<view template="urc_app:/templates/streams.html" type="html" isDashboard="False">
    <label>Streams</label>
</view>
```

### New template: `streams.html`

Same structure as existing `builder.html` — loads Splunk config, i18n, and the page JS bundle.

### Updated navigation: `default.xml`

```xml
<nav>
    <view name="streams" default="true"/>
    <view name="builder"/>
    <view name="configuration"/>
    <view name="search"/>
</nav>
```

Note: `builder` is not in the nav menu (accessed via streams row click) but needs a view registration for direct URL access.

## Error Handling

### Error Scenarios

1. **UCC REST endpoint unreachable**
   - **Handling:** `splunk-api.ts` catches fetch errors, returns typed error objects
   - **User Impact:** Dashboard shows "Unable to connect to Splunk REST API" message with retry button. Builder shows same in a Message banner.

2. **Manifest parse failure on edit**
   - **Handling:** `manifestToFormState()` catches YAML parse errors, returns partial state with unparseable sections flagged
   - **User Impact:** Builder loads with populated fields where possible. A warning banner shows "Some manifest sections could not be parsed" with option to view raw YAML.

3. **Test connection fails (network/auth error)**
   - **Handling:** `test_connection` endpoint returns structured error with HTTP status and message
   - **User Impact:** Test panel shows error message with actionable suggestion (e.g., "401 Unauthorized — verify credentials in Configuration → Accounts").

4. **Unsupported component type in existing manifest**
   - **Handling:** TypeSelector shows the type as selected with a warning banner. Fields render based on raw manifest values (schema-driven rendering falls back to text fields for unknown types).
   - **User Impact:** Warning banner: "This component type is not supported by the URC engine. It may not work at runtime." User can still edit and save.

5. **Save fails (validation error from UCC)**
   - **Handling:** UCC's AdminExternalHandler returns field-level validation errors
   - **User Impact:** Fields with errors are highlighted, error messages shown inline.

## Testing Strategy

### Unit Testing (Jest + React Testing Library)

- **manifest-serializer.ts:** Round-trip tests — form state → YAML → form state. Edge cases: empty streams, multiple streams, all component types.
- **form-schema.ts generation:** Verify the build script produces valid TypeScript with correct field counts and types.
- **SchemaField.tsx:** Render each field type (text, enum, boolean, number, template, nested object) and verify correct Splunk UI component is used.
- **TypeSelector.tsx:** Verify supported/unsupported grouping, warning display.
- **TimestampPicker.tsx:** Test detection heuristics with various field patterns.
- **splunk-api.ts:** Mock fetch, verify URL construction and response parsing.
- **BuilderContext reducer:** Test all action types, verify state transitions.

### Integration Testing (Jest + mocked Splunk API)

- **Dashboard → Builder flow:** Verify clicking a row navigates with correct input name.
- **Load → Edit → Save round-trip:** Load existing input, modify fields, save, verify manifest YAML output.
- **Test → Timestamp detection flow:** Mock test_connection response, verify timestamp candidates appear in Splunk Output tab.
- **Validation warnings:** Load manifest with unsupported types, verify warnings appear in correct form sections.

### Backend Testing (pytest)

- **validate_manifest_capabilities():** Test with manifests containing unknown types, PKCE auth, JWT RS256. Verify correct severity and messages.
- **emit() with level parameter:** Verify DEBUG-level messages only emit when logger is at DEBUG.
- **before_hook:** Verify it emits at DEBUG and is silent at INFO.
- **test_connection with mode=validate:** Verify validation-only response format.

### Manual Testing

- Full flow in Splunk: create account → create stream via builder → test connection → save → verify data collection → check debug logs.
- Theme testing: verify all components render correctly in both light and dark Splunk themes.
- Existing UCC pages: verify Configuration tab (accounts, logging, proxy) still works unchanged.
