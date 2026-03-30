# Design Document — URC Guided UI

## Overview

A guided connector builder for the URC Splunk add-on. Renders as a separate Splunk page (CIMplicity hybrid pattern) alongside the UCC-generated Configuration page. Built with `@splunk/react-ui` components, `@splunk/react-page` layout, and `styled-components` theming. Generates Airbyte-compatible manifest YAML from a visual form.

## Code Reuse Analysis

### Existing Components to Leverage
- **`@splunk/react-ui` v5.9.0**: All form controls (ControlGroup, Text, Select, TextArea, Switch, ComboBox), layout (Card, TabLayout, Accordion, ColumnLayout), feedback (Message, WaitSpinner, Tooltip), navigation (TabBar, StepBar)
- **`@splunk/react-page` v8.2.1**: Page mounting in Splunk Enterprise layout
- **`@splunk/themes` v1.6.0**: Design tokens (spacing, colors, typography) via styled-components
- **`@splunk/splunk-utils` v3.4.0**: CSRF tokens, theme loading, REST URL creation
- **CIMplicity patterns**: 3-column layout, custom stepper, `postToEndpoint` REST helper, build pipeline
- **Existing React monorepo**: Webpack 5 + Babel 7 + TypeScript build pipeline at `react/`
- **URC test manifests**: 5 pre-built manifests at `ucc/urc_app/tests/manifests/` for template gallery

### Integration Points
- **UCC REST API**: Create/update/list inputs via `/servicesNS/-/urc_app/urc_app_input/` endpoint
- **Test Connection**: POST to `/servicesNS/-/urc_app/urc_app/test_connection` with manifest + account
- **Account listing**: GET `/servicesNS/-/urc_app/urc_app_account/` for credential dropdown
- **UCC Configuration page**: Remains untouched — builder adds a new page, not a new tab

## Architecture Decision: Separate Page (not UCC Custom Tab)

### Options Evaluated

| Option | Pros | Cons |
|--------|------|------|
| **A: UCC Custom Tab** | Lives inside UCC, gets save/validate for free | Limited to single panel, can't do 3-column layout, wizard awkward in tab context |
| **B: UCC Custom Control** | Replaces manifest textarea, minimal change | Only controls one field, can't add test panel or stream sidebar |
| **C: Separate Splunk Page** (chosen) | Full creative freedom, 3-column layout, wizard flow, proven by CIMplicity | Separate build, must handle save/validate ourselves |
| **D: Full external SPA** | Maximum flexibility | Breaks Splunk integration, no theme support, auth headaches |

**Decision: Option C** — Separate Splunk page following CIMplicity pattern. This gives us full layout control for the guided wizard while staying native to Splunk. The builder page handles connector creation/editing. UCC Configuration handles accounts, logging, proxy.

## UI Layout — ASCII Mockups

### Main Builder View (3-column layout)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  URC — Universal REST Client                    [Configuration] [Builder]  │
├──────────────┬──────────────────────────────────────┬───────────────────────┤
│              │                                      │                       │
│  STREAMS     │  Stream: github_repos                │  HELP & CONTEXT       │
│              │                                      │                       │
│  ┌────────┐  │  ┌─────────────────────────────────┐  │  ┌─────────────────┐  │
│  │● repos │  │  │ ▼ Endpoint                      │  │  │ Endpoint        │  │
│  ├────────┤  │  │                                 │  │  │                 │  │
│  │  issues│  │  │ URL ___________________________│  │  │ The base URL of │  │
│  ├────────┤  │  │ https://api.github.com/orgs/... │  │  │ your REST API.  │  │
│  │  pulls │  │  │                                 │  │  │                 │  │
│  └────────┘  │  │ Method [GET ▼]                  │  │  │ Example:        │  │
│              │  │                                 │  │  │ https://api.    │  │
│  [+ Add]     │  └─────────────────────────────────┘  │  │ github.com      │  │
│              │                                      │  │                 │  │
│              │  ┌─────────────────────────────────┐  │  │ Tip: Use        │  │
│  ──────────  │  │ ▶ Authentication                │  │  │ {{ config['x'] }}│  │
│              │  └─────────────────────────────────┘  │  │ for dynamic     │  │
│  ACTIONS     │                                      │  │ values from     │  │
│              │  ┌─────────────────────────────────┐  │  │ your Account.   │  │
│  [Form|YAML] │  │ ▶ Pagination                    │  │  └─────────────────┘  │
│              │  └─────────────────────────────────┘  │                       │
│  [Templates] │                                      │  ┌─────────────────┐  │
│              │  ┌─────────────────────────────────┐  │  │ 💡 Quick Start  │  │
│              │  │ ▶ Record Extraction              │  │  │                 │  │
│              │  └─────────────────────────────────┘  │  │ 1. Set endpoint │  │
│              │                                      │  │ 2. Pick auth    │  │
│              │  ┌─────────────────────────────────┐  │  │ 3. Test it      │  │
│              │  │ ▶ Incremental Sync               │  │  │ 4. Save         │  │
│              │  └─────────────────────────────────┘  │  └─────────────────┘  │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Transformations                 │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Error Handling                  │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
├──────────────┴──────────────────────────────────────┴───────────────────────┤
│  TEST PANEL                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Account [prod-github ▼]   [🔬 Test Connection]   ⟳ 5 records, 230ms│   │
│  ├──────────┬───────────┬──────────┬───────────┐                       │   │
│  │ Records  │ Request   │ Response │ Schema    │                       │   │
│  ├──────────┴───────────┴──────────┴───────────┘                       │   │
│  │  id │ name          │ full_name           │ description      │ ... │   │
│  │  1  │ anthropic-sdk │ anthropics/anthro... │ Python SDK for...│     │   │
│  │  2  │ claude-code   │ anthropics/claude... │ CLI for Claude   │     │   │
│  │  3  │ courses       │ anthropics/courses   │ Educational...   │     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                  [Save ▼]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Authentication Section (expanded)

```
┌─────────────────────────────────────────────────┐
│ ▼ Authentication                                │
│                                                 │
│ Auth Type  [Bearer Token          ▼]            │
│            ┌────────────────────────┐           │
│            │ None                   │           │
│            │ API Key                │           │
│            │ ● Bearer Token         │  ← Each  │
│            │ Basic Auth             │    has a  │
│            │ OAuth 2.0 (Client)     │    1-line │
│            │ Digest Auth            │    desc   │
│            │ JWT                    │           │
│            │ Session Token          │           │
│            └────────────────────────┘           │
│                                                 │
│ Token      [{{ config['api_key'] }} ]           │
│            ℹ️ References the API Key from your  │
│              Account configuration              │
│                                                 │
│ ┌─ Preview ──────────────────────────────────┐  │
│ │ Authorization: Bearer ••••••••••••sk-1234  │  │
│ └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Pagination Section (expanded)

```
┌─────────────────────────────────────────────────┐
│ ▼ Pagination                                    │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ How does this API paginate?                 │ │
│ │                                             │ │
│ │ ○ No pagination (single page of results)    │ │
│ │                                             │ │
│ │ ● Offset — API uses offset + limit params   │ │
│ │   "Give me items 100-200"                   │ │
│ │                                             │ │
│ │ ○ Page number — API uses page + per_page    │ │
│ │   "Give me page 3"                          │ │
│ │                                             │ │
│ │ ○ Cursor — API returns a next_page token    │ │
│ │   "Continue from this token"                │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Page Size      [100          ]                  │
│ Offset Param   [_start       ]  ← auto-filled  │
│ Limit Param    [_limit       ]  ← from strategy │
│                                                 │
│ ┌─ How it works ─────────────────────────────┐  │
│ │ Request 1: ?_start=0&_limit=100            │  │
│ │ Request 2: ?_start=100&_limit=100          │  │
│ │ Request 3: ?_start=200&_limit=100          │  │
│ │ ...until fewer than 100 results returned   │  │
│ └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Record Extraction (with test data)

```
┌─────────────────────────────────────────────────┐
│ ▼ Record Extraction                             │
│                                                 │
│ Where are the records in the API response?      │
│                                                 │
│ Field Path  [result                   ]         │
│             ℹ️ Dot-separated path to the array  │
│                of records. Leave empty if the   │
│                response IS the array.           │
│                                                 │
│ ┌─ API Response Preview ─────────────────────┐  │
│ │ {                                          │  │
│ │   "count": 156,                            │  │
│ │   "next": "...?offset=100",                │  │
│ │   ►"result": [  ← records are here         │  │
│ │     { "sys_id": "abc", "number": "INC01" } │  │
│ │     { "sys_id": "def", "number": "INC02" } │  │
│ │   ]                                        │  │
│ │ }                                          │  │
│ └────────────────────────────────────────────┘  │
│                                                 │
│ [Apply Filter] (optional)                       │
│ Condition  [{{ record['status'] == 'active' }}] │
└─────────────────────────────────────────────────┘
```

### YAML Editor Mode

```
┌──────────────┬──────────────────────────────────────┬───────────────────────┐
│              │                                      │                       │
│  STREAMS     │  ┌─ YAML Editor ──────────────────┐  │  HELP & CONTEXT       │
│              │  │ 1  type: DeclarativeSource      │  │                       │
│  ┌────────┐  │  │ 2  version: "1.0.0"            │  │  ┌─────────────────┐  │
│  │● repos │  │  │ 3                              │  │  │ YAML Mode       │  │
│  ├────────┤  │  │ 4  check:                      │  │  │                 │  │
│  │  issues│  │  │ 5    type: CheckStream         │  │  │ Edit the raw    │  │
│  └────────┘  │  │ 6    stream_names: [repos]     │  │  │ manifest YAML   │  │
│              │  │ 7                              │  │  │ directly.       │  │
│  [+ Add]     │  │ 8  streams:                    │  │  │                 │  │
│              │  │ 9    - type: DeclarativeStream  │  │  │ Switch back to  │  │
│  ──────────  │  │10      name: repos             │  │  │ Form mode to    │  │
│              │  │11      retriever:               │  │  │ use the visual  │  │
│  [●Form|YAML]│  │12        type: SimpleRetriever  │  │  │ editor.         │  │
│              │  │13        requester:             │  │  │                 │  │
│  [Templates] │  │14  ❌ L14: expected indent     │  │  │ Errors are      │  │
│              │  │                                │  │  │ shown inline.   │  │
│              │  └────────────────────────────────┘  │  └─────────────────┘  │
│              │                                      │                       │
├──────────────┴──────────────────────────────────────┴───────────────────────┤
│  TEST PANEL (collapsed — click to expand)                           [▼]    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Template Gallery (modal)

```
┌───────────────────────────────────────────────────────────────────┐
│  Start from Template                                     [✕]     │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  🌐         │ │  🐙         │ │  ❄️         │ │  🔷        │ │
│  │  Generic    │ │  GitHub     │ │  ServiceNow │ │  Infoblox  │ │
│  │  REST API   │ │  API        │ │  Table API  │ │  WAPI      │ │
│  │             │ │             │ │             │ │            │ │
│  │  Blank      │ │  Bearer     │ │  Basic Auth │ │  Basic Auth│ │
│  │  canvas     │ │  Cursor pag │ │  Offset pag │ │  Cursor pag│ │
│  │             │ │  Substreams │ │  Incremental│ │            │ │
│  │  [Select]   │ │  [Select]   │ │  [Select]   │ │  [Select]  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐                                 │
│  │  ☁️         │ │  📋         │                                 │
│  │  Azure      │ │  JSON       │                                 │
│  │  Mgmt API   │ │  Placeholder│                                 │
│  │             │ │             │                                 │
│  │  OAuth 2.0  │ │  No auth    │                                 │
│  │  Cursor pag │ │  Offset pag │                                 │
│  │             │ │  Great for  │                                 │
│  │  [Select]   │ │  testing    │                                 │
│  └─────────────┘ │  [Select]   │                                 │
│                   └─────────────┘                                 │
│                                                                   │
│                               [Start from Scratch]                │
└───────────────────────────────────────────────────────────────────┘
```

### Save Flow (modal)

```
┌───────────────────────────────────────────────────────────┐
│  Save Connector                                    [✕]    │
│                                                           │
│  Input Name    [github_connector      ]                   │
│                ℹ️ Unique name for this data input          │
│                                                           │
│  Account       [prod-github           ▼]                  │
│                ℹ️ Credentials to use (configured in        │
│                   Configuration → Account)                │
│                                                           │
│  Interval      [300                   ] seconds           │
│                ℹ️ How often to collect data (300 = 5 min)  │
│                                                           │
│  Index         [main                  ▼]                  │
│                                                           │
│  Sourcetype    [urc:github:repos      ]                   │
│                ℹ️ Auto-generated from stream names         │
│                                                           │
│                          [Cancel]  [Save & Enable]        │
└───────────────────────────────────────────────────────────┘
```

## Component Architecture

### Package Structure (CIMplicity Pattern)

```
react/packages/
├── urc-builder/                    # Reusable component library
│   ├── package.json
│   ├── src/
│   │   ├── index.ts                # Public exports
│   │   ├── ConnectorBuilder.tsx     # Root component (3-column layout)
│   │   ├── state/
│   │   │   ├── BuilderState.ts      # TypeScript interfaces for form state
│   │   │   ├── useBuilderState.ts   # React state hook (form ↔ YAML)
│   │   │   └── manifestSerializer.ts # form state → YAML (and reverse)
│   │   ├── sections/               # Collapsible config sections
│   │   │   ├── EndpointSection.tsx
│   │   │   ├── AuthSection.tsx
│   │   │   ├── PaginationSection.tsx
│   │   │   ├── ExtractionSection.tsx
│   │   │   ├── IncrementalSection.tsx
│   │   │   ├── TransformSection.tsx
│   │   │   └── ErrorHandlerSection.tsx
│   │   ├── panels/
│   │   │   ├── StreamSidebar.tsx     # Left: stream list + actions
│   │   │   ├── HelpPanel.tsx         # Right: contextual help
│   │   │   └── TestPanel.tsx         # Bottom: test connection results
│   │   ├── dialogs/
│   │   │   ├── SaveDialog.tsx        # Save/deploy modal
│   │   │   └── TemplateGallery.tsx   # Template picker modal
│   │   ├── editor/
│   │   │   └── YamlEditor.tsx        # CodeMirror 6 YAML editor
│   │   └── utils/
│   │       ├── api.ts                # Splunk REST helpers
│   │       ├── templates.ts          # Built-in manifest templates
│   │       └── helpContent.ts        # Field-level help text catalog
│   └── tests/
│       └── ...
│
├── urc-app/                         # Splunk app wrapper
│   ├── package.json
│   ├── webpack.config.js
│   ├── src/
│   │   ├── main/
│   │   │   ├── webapp/pages/
│   │   │   │   └── builder/
│   │   │   │       └── index.tsx     # Entry: getUserTheme → layout(<ConnectorBuilder />)
│   │   │   └── resources/splunk/
│   │   │       ├── default/
│   │   │       │   └── data/ui/
│   │   │       │       ├── nav/default.xml
│   │   │       │       └── views/builder.xml
│   │   │       └── appserver/templates/
│   │   │           └── builder.html
│   │   └── stage/                    # Build output → merged into UCC output
│   └── ...
```

### Key Components

#### ConnectorBuilder (Root)
- **Purpose:** 3-column layout orchestrator. Manages active stream, form/YAML mode toggle, test panel visibility.
- **Interfaces:** `<ConnectorBuilder initialManifest?: string, inputName?: string />`
- **Dependencies:** All section components, StreamSidebar, HelpPanel, TestPanel
- **Layout:** CSS Grid: `grid-template-columns: 240px 1fr 280px; grid-template-rows: 1fr auto`

#### useBuilderState (State Hook)
- **Purpose:** Single source of truth for all builder state. Manages form fields, YAML string, active stream, validation errors.
- **Interfaces:**
  ```typescript
  interface BuilderState {
    mode: 'form' | 'yaml';
    streams: StreamConfig[];
    activeStreamIndex: number;
    globalConfig: { check: { stream_names: string[] } };
    yamlString: string;
    yamlErrors: YamlError[];
    validationErrors: Record<string, string>;
    testResults: TestResult | null;
    isDirty: boolean;
  }
  ```
- **Key methods:** `setField(path, value)`, `addStream()`, `removeStream(idx)`, `toYaml()`, `fromYaml(str)`, `toManifest()`

#### manifestSerializer
- **Purpose:** Pure functions for bidirectional form state ↔ YAML conversion. No React dependencies — fully testable.
- **Interfaces:**
  ```typescript
  function formStateToYaml(state: BuilderState): string;
  function yamlToFormState(yaml: string): BuilderState | ValidationError[];
  function formStateToManifest(state: BuilderState, config: Record<string, string>): string;
  ```
- **Reuses:** `js-yaml` for serialization, YAML field ordering to match Airbyte conventions

#### Section Components (AuthSection, PaginationSection, etc.)
- **Purpose:** Each renders a collapsible `Accordion` panel with form fields for one manifest concept
- **Pattern:** All sections follow the same interface:
  ```typescript
  interface SectionProps {
    stream: StreamConfig;
    onChange: (path: string, value: any) => void;
    onFocus: (fieldId: string) => void;  // triggers help panel update
    validationErrors: Record<string, string>;
    testResponse?: any;  // used by ExtractionSection for path preview
  }
  ```
- **Splunk UI components used:** `Accordion`, `ControlGroup`, `Text`, `Select`, `ComboBox`, `Switch`, `TextArea`, `RadioBar`, `Tooltip`, `Message`

#### TestPanel
- **Purpose:** Bottom docked panel. Calls `/test_connection`, displays results in `TabLayout` (Records, Request, Response, Schema tabs).
- **Interfaces:** `<TestPanel manifest={yaml} onAutoDetect={(fieldPath) => void} />`
- **Dependencies:** `api.ts` for REST call, `@splunk/react-ui/Table` for record display, `@splunk/react-ui/TabLayout`
- **Auto-detect:** When test succeeds, analyzes response shape and suggests `field_path` for record extraction

#### YamlEditor
- **Purpose:** CodeMirror 6 editor with YAML syntax highlighting, error markers, and line numbers.
- **Interfaces:** `<YamlEditor value={yaml} onChange={(yaml) => void} errors={YamlError[]} />`
- **Dependencies:** `@codemirror/lang-yaml`, `@codemirror/view`, `@codemirror/state`
- **Integration:** Debounced onChange (300ms) triggers `yamlToFormState` validation

#### api.ts (REST Helper)
- **Purpose:** All Splunk REST calls. Follows CIMplicity pattern with CSRF tokens.
- **Interfaces:**
  ```typescript
  function testConnection(manifest: string, accountName: string): Promise<TestResult>;
  function listAccounts(): Promise<Account[]>;
  function listIndexes(): Promise<string[]>;
  function saveInput(name: string, config: InputConfig): Promise<void>;
  function loadInput(name: string): Promise<InputConfig>;
  function listInputs(): Promise<InputSummary[]>;
  ```
- **Reuses:** `@splunk/splunk-utils` for `createRESTURL`, `getDefaultFetchInit`

## Data Models

### StreamConfig (Form State)
```typescript
interface StreamConfig {
  name: string;
  endpoint: {
    url: string;
    httpMethod: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
    headers: Record<string, string>;
    params: Record<string, string>;
    bodyType: 'none' | 'json' | 'form' | 'graphql';
    body: string;
  };
  auth: {
    type: 'none' | 'api_key' | 'bearer' | 'basic' | 'oauth2' | 'digest' | 'jwt' | 'session_token';
    // Type-specific fields populated conditionally
    apiToken?: string;
    header?: string;
    username?: string;
    password?: string;
    clientId?: string;
    clientSecret?: string;
    tokenUrl?: string;
    // ... (all use {{ config['x'] }} interpolation references)
  };
  pagination: {
    type: 'none' | 'offset' | 'page' | 'cursor';
    pageSize: number;
    offsetParam?: string;
    limitParam?: string;
    cursorValue?: string;
    stopCondition?: string;
    pageParam?: string;
  };
  extraction: {
    fieldPath: string[];      // e.g., ['data', 'items']
    filterCondition?: string;
  };
  incremental: {
    enabled: boolean;
    type: 'datetime' | 'count';
    cursorField?: string;
    startDatetime?: string;
    datetimeFormat?: string;
    startValue?: number;
  };
  transformations: TransformConfig[];
  errorHandling: {
    maxRetries: number;
    backoffType: 'constant' | 'exponential' | 'header';
    retryStatusCodes: number[];
    failStatusCodes: number[];
  };
}

interface TransformConfig {
  type: 'AddFields' | 'RemoveFields' | 'KeysToLower' | 'KeysToSnakeCase' | 'FlattenFields';
  fields?: { path: string[]; value: string }[];
  fieldPointers?: string[][];
  separator?: string;
}
```

### TestResult
```typescript
interface TestResult {
  success: boolean;
  records: Record<string, any>[];
  recordCount: number;
  error?: string;
  request: { url: string; method: string; headers: Record<string, string>; params: Record<string, string> };
  response: { status: number; headers: Record<string, string>; body: any };
  durationMs: number;
}
```

## Build Pipeline

Following the CIMplicity two-stage pattern:

```
1. ucc-gen build --source ucc/urc_app -o ucc/output/
   └── Generates: REST handlers, conf files, UCC UI (Configuration page)

2. Webpack build of react/packages/urc-app/
   └── Input: react/packages/urc-builder/ component library
   └── Output: stage/appserver/static/pages/builder.js

3. Merge: Copy ucc/output/urc_app/ as base, overlay react build output
   └── Add: appserver/static/pages/builder.js
   └── Add: appserver/templates/builder.html
   └── Add: default/data/ui/views/builder.xml
   └── Modify: default/data/ui/nav/default.xml (add builder as default view)

4. ucc-gen package (or manual tar) for distribution
```

**Taskfile integration:**
```yaml
ucc:build-ui:
  desc: Build the connector builder React UI
  cmds:
    - cd react && yarn workspace @splunk/urc-app build

ucc:build-full:
  desc: UCC build + React UI build + merge
  cmds:
    - task: ucc:build
    - task: ucc:build-ui
    - task: ucc:merge-ui
```

## Navigation Integration

```xml
<!-- default/data/ui/nav/default.xml -->
<nav>
    <view name="builder" default="true"/>
    <view name="configuration"/>
    <view name="search"/>
</nav>
```

```xml
<!-- default/data/ui/views/builder.xml -->
<view template="urc_app:/templates/builder.html" type="html">
    <label>Connector Builder</label>
</view>
```

## Error Handling

### Error Scenarios

1. **YAML Parse Error**
   - **Handling:** `js-yaml` parse error caught, line number extracted, error marker shown in editor
   - **User Impact:** Red underline on error line, message in gutter: "Invalid YAML at line 14: expected indent"

2. **Test Connection Failure (network)**
   - **Handling:** Catch fetch error, show in test panel with retry button
   - **User Impact:** "Could not reach the API. Check your URL and network settings." + [Retry]

3. **Test Connection Failure (auth)**
   - **Handling:** Parse 401/403 status, map to actionable message
   - **User Impact:** "Authentication failed (401). Check your API key in Configuration → Account."

4. **Manifest Validation Error**
   - **Handling:** Pydantic validation errors returned from test_connection, mapped to form field paths
   - **User Impact:** Red border + message on the specific field: "Required: pagination page_size must be a number"

5. **Save Failure (duplicate name)**
   - **Handling:** UCC REST API returns conflict, offer rename or update
   - **User Impact:** "An input named 'github_connector' already exists. Update it or choose a different name."

6. **Session Lost (page refresh)**
   - **Handling:** Form state persisted to `sessionStorage` on every change, restored on mount
   - **User Impact:** "Welcome back. Your unsaved changes have been restored." (dismissible banner)

## Testing Strategy

### Unit Testing
- `manifestSerializer.ts` — round-trip tests: form → YAML → form for every component type
- Section components — render tests with `@testing-library/react`, verify field changes propagate
- `useBuilderState` — hook tests for add/remove stream, mode toggle, validation

### Integration Testing
- `api.ts` — mock Splunk REST responses, verify request format and error handling
- TestPanel — mock test_connection response, verify records render in table
- Save flow — mock UCC input API, verify manifest YAML is correctly saved

### End-to-End Testing
- Open builder → select template → test connection → verify records → save → verify input created
- Open builder → YAML mode → paste manifest → switch to form → verify fields populated
- Edit existing input → modify stream → save → verify update applied

## Dependencies (New)

```json
{
  "@codemirror/lang-yaml": "^6.1.0",
  "@codemirror/view": "^6.x",
  "@codemirror/state": "^6.x",
  "@codemirror/theme-one-dark": "^6.x",
  "js-yaml": "^4.1.0"
}
```

All other dependencies (`@splunk/react-ui`, `@splunk/react-page`, `styled-components`, etc.) are already available in the monorepo.
