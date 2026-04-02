# React UI for Splunk Add-ons — A Teaching Guide

A practical guide to understanding and extending the React UI for the URC (Universal REST Client) Splunk add-on. This teaches the architecture, patterns, and why each part exists.

**Audience:** React developers with basic React knowledge (hooks, components, state) who want to add form UIs or interactive pages to Splunk add-ons.

**Last updated:** 2026-04-02

---

## Table of Contents

1. [Why React + Splunk?](#1-why-react--splunk)
2. [Monorepo Structure](#2-monorepo-structure)
3. [The Three Packages](#3-the-three-packages)
4. [Build Pipeline: From React to Splunk](#4-build-pipeline-from-react-to-splunk)
5. [UrcBuilder Architecture](#5-urcbuilder-architecture)
6. [State Management with Context + Reducer](#6-state-management-with-context--reducer)
7. [The Four Builder Steps](#7-the-four-builder-steps)
8. [TypeScript Types and Validation](#8-typescript-types-and-validation)
9. [REST API Communication](#9-rest-api-communication)
10. [Common Patterns and Best Practices](#10-common-patterns-and-best-practices)
11. [Development Workflow](#11-development-workflow)

---

## 1. Why React + Splunk?

Splunk Web is built on a custom JavaScript framework called `Splunk.js`, but modern Splunk add-ons can supply their own React components. This is powerful because:

- **Guided UX:** Instead of raw forms, you create interactive wizards with validation, previews, and step-by-step guidance
- **Component reuse:** The `@splunk/react-ui` library gives you pre-built, styled components that match Splunk's design
- **Developer experience:** Hot reload, TypeScript, modern tooling, and Jest for testing
- **Separation of concerns:** React handles the UI; Python/UCC handles the backend data model

**The catch:** Splunk runs your React app inside a Mako template in the add-on's `appserver/`, so you're not building a standalone SPA. Instead, you're building a component that gets compiled by Webpack and embedded into the add-on. The UCC framework generates the REST endpoints and conf files; React fills the UI layer.

---

## 2. Monorepo Structure

The React code lives in `/workspace/react/` as a **yarn workspace monorepo**. This means multiple independent packages share one `node_modules` and can depend on each other.

### Layout

```
react/
├── package.json                    # Root workspace definition
├── yarn.lock                       # Locked versions for reproducibility
├── babel.config.js                 # Babel config shared by all packages
├── .prettierrc                      # Formatting rules
│
└── packages/
    ├── test-page/                  # Example/testing package
    ├── urc-app/                    # Main Splunk app (the output)
    └── urc-builder/                # Shared component library
        ├── package.json            # Dependencies: react, @splunk/react-ui, types
        ├── src/
        │   ├── UrcBuilder.tsx       # Entry point component
        │   ├── types.ts             # Shared interfaces
        │   ├── builder/
        │   │   ├── StreamBuilder.tsx        # Main wrapper
        │   │   ├── tabs/                    # Tab components (5 files)
        │   │   └── form/
        │   ├── context/
        │   │   ├── BuilderContext.tsx       # Form state + reducer
        │   │   └── DashboardContext.tsx     # Dashboard state
        │   ├── services/
        │   │   ├── splunk-api.ts            # REST calls to Splunk
        │   │   └── manifest-serializer.ts   # Form ↔ YAML conversion
        │   ├── dashboard/               # Dashboard builder components
        │   └── schema/                      # Generated field schemas
        └── tests/                   # Jest unit tests
```

### Key Files in Each Package

Every package has this structure:

```
package.json              # npm metadata, scripts, dependencies
src/
  index.ts               # Export the main component
  <Component>.tsx        # React component (TSX = TypeScript + JSX)
tests/
  <Component>.test.tsx   # Jest unit tests
webpack.config.js        # Webpack bundler config
tsconfig.json            # TypeScript compiler config
jest.config.js           # Jest testing config
stylelint.config.js      # CSS linting rules
```

---

## 3. The Three Packages

### 3a. `@splunk/urc-builder`

**Purpose:** Reusable React component library for building REST API collectors.

**What it exports:**
- `StreamDashboard` — Screen 1: table of all streams with tags, filters, bulk actions
- `StreamBuilder` — Screen 2: 5-tab form for creating/editing streams
- `BuilderProvider` / `DashboardProvider` — React Context providers for state
- `types.ts` — TypeScript interfaces used everywhere
- `services/splunk-api.ts` — wrapper around Splunk's REST API
- `services/manifest-serializer.ts` — bidirectional form state <-> YAML conversion

**Why separate?** So the Splunk app wrapper (`urc-app`) just mounts components. The library is testable and reusable independently.

**How `urc-app` uses it:**
- `pages/streams/index.tsx` imports and mounts `StreamDashboard`
- `pages/builder/index.tsx` imports and mounts `StreamBuilder`

### 3b. `@splunk/urc-app`

**Purpose:** The actual Splunk add-on React UI entry point. Thin wrapper — almost no logic.

**What it does:**
1. Has page entry points under `src/main/webapp/pages/` (webpack auto-discovers them)
2. Each page imports a component from `@splunk/urc-builder`
3. Wraps it in `@splunk/react-page` layout with theme support
4. Webpack compiles each page to `stage/appserver/static/pages/<name>.js`

**Key pages:**
- `pages/streams/index.tsx` — mounts `StreamDashboard` (default landing page)
- `pages/builder/index.tsx` — mounts `StreamBuilder`, reads `?input=name` from URL for edit mode

**Splunk integration files:**
- `default/data/ui/views/streams.xml` + `builder.xml` — Splunk view definitions
- `appserver/templates/streams.html` + `builder.html` — Mako templates that load the JS bundles
- `default/data/ui/nav/default.xml` — navigation (streams is default)

**What "linking the app" means:**
```bash
yarn workspace @splunk/urc-app run link:app
```
This copies the built app into `/workspace/ucc/output/urc_app/`, making it available to Splunk.

### 3c. `@splunk/test-page`

**Purpose:** Isolated testing/demo page for UI development.

**Why?** So developers can test the UI without spinning up a full Splunk instance. It has the same Webpack + Jest setup, but targets a browser instead of Splunk.

**Typically used for:**
- Visual regression testing
- Component isolation testing
- Demo/docs of available components

---

## 4. Build Pipeline: From React to Splunk

Understanding this pipeline is crucial because the UI doesn't auto-sync to Splunk — it must be built and linked.

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: yarn build                                          │
│ ────────────────────────────────────────────────────────────│
│ Every package runs: tsc (type-check) + webpack              │
│ Output: dist/ folder in each package, stage/ in urc-app     │
│                                                             │
│ urc-builder produces:                                       │
│   dist/index.d.ts          (TypeScript declarations)        │
│   dist/index.js            (compiled JS)                    │
│                                                             │
│ urc-app produces:                                           │
│   stage/appserver/static/pages/entry_page.js               │
│   stage/appserver/templates/base.html                      │
│   stage/default/app.conf                                   │
│   (+ other auto-generated files)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: yarn link:app  (or: task ucc:link)                 │
│ ────────────────────────────────────────────────────────────│
│ Copies stage/ → /workspace/ucc/output/urc_app/             │
│ This is a COMPLETE Splunk add-on directory                 │
│                                                             │
│ /workspace/ucc/output/urc_app/                             │
│   appserver/static/pages/entry_page.js  (our compiled React)│
│   appserver/templates/base.html         (Mako wrapper)     │
│   default/app.conf                      (add-on metadata)  │
│   bin/urc_app_input_helper.py           (collection logic) │
│   lib/urc/                              (vendored deps)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: task dev:up  (docker-compose up)                   │
│ ────────────────────────────────────────────────────────────│
│ Splunk mounts /workspace/ucc/config/apps/ via symlink      │
│ When you access Splunk Web, it finds entry_page.js         │
│ The Mako template loads React, starts the component        │
└─────────────────────────────────────────────────────────────┘
```

### Why This Design?

1. **Separation:** UCC generates Python + conf files; React generates the UI bundle
2. **Rebuild-safe:** Rebuilding React doesn't touch the Python code
3. **Staging:** The `stage/` directory acts as a "draft" of the full app
4. **Webpack magic:** Webpack bundles + minifies React code into a single file

---

## 5. UrcBuilder Architecture

The UrcBuilder is the heart of the UI. It manages creating and editing REST API collection manifests through a 4-step wizard.

### High-Level Structure

```
Screen 1: StreamDashboard
  ├─ DashboardProvider (context for filters + selection)
  ├─ PageHeader (title + "New Stream" CTA)
  ├─ SummaryCards (total/active/disabled with icons and colored accents)
  ├─ TagFilter (RadioBar status filter + Chip tag toggles)
  ├─ BulkActions (enable/disable/delete selected)
  └─ StreamTable (Badge status, Chip tags, kebab Menu per row, empty-state hero)

Screen 2: StreamBuilder (4-step wizard)
  ├─ BuilderProvider (context + useReducer for form state)
  ├─ Breadcrumbs ("Streams / New Stream" or "Streams / Edit: {name}")
  ├─ StepBar (Connect → Configure → Output → Test & Save)
  ├─ Step description banner (icon + contextual help)
  │
  ├─ Step 1: ConnectionTab        (stream name, account, base URL, HTTP method)
  ├─ Step 2: DataMappingTab       (schema-driven: extractor, pagination, sync,
  │                                 partition router, decoder, transforms, errors)
  ├─ Step 3: SplunkOutputTab      (index + sourcetype | interval + tags, advanced)
  └─ Step 4: TestPreviewTab       (test connection, save)

  Footer: [Cancel] [← Back] [Next →] or [← Back] [Save] on final step
```

**Content registry:** All UI copy, icons, and help text live in `content.ts` — a single source of truth. Components import from this file; no strings are hardcoded.

**Schema-driven forms:** The Configure step renders forms from `form-schema.ts`, which is auto-generated from the Airbyte declarative YAML schema. 53 component types across 11 categories — see `docs/schema-components-guide.md`.

**Splunk UI components used:** StepBar, Card, Badge, Chip, Menu, Tooltip, Breadcrumbs, CollapsiblePanel, RadioBar, ColumnLayout, DefinitionList, plus 15+ icons from @splunk/react-icons.

### Example: How Editing Works

1. User selects an account in ConnectionTab
2. Tab calls: `dispatch({ type: 'SET_FIELD', path: 'account', value: 'my_account' })`
3. Reducer in BuilderContext updates the form state
4. All tabs re-render with new state
5. On save, manifest-serializer converts form state → YAML → REST call

---

## 6. State Management with Context + Reducer

React Context + useReducer is the pattern used for global form state. This avoids "prop drilling" (passing state through 10 layers of components).

### BuilderContext.tsx Pattern

**Why Context?**
- Tabs are deeply nested; passing state as props is tedious
- Multiple components need the same data
- Updates must be coordinated (changing one tab might affect another)

**How it works:**

```typescript
// 1. Create context (at top of file)
const BuilderContext = createContext<BuilderContextType | null>(null);

// 2. Define reducer function (handles all state changes)
function builderReducer(state: BuilderState, action: BuilderAction): BuilderState {
    switch (action.type) {
        case 'SET_FIELD': {
            // Update nested state at path like "streams[0].retriever.type"
            return {
                ...state,
                streams: updateAtPath(state.streams, action.path, action.value),
            };
        }
        case 'LOAD_INPUT': {
            return manifestToFormState(action.payload);
        }
        // ... other cases
    }
}

// 3. Provider component wraps the app
export function BuilderProvider({ children }) {
    const [state, dispatch] = useReducer(builderReducer, INITIAL_STATE);
    return (
        <BuilderContext.Provider value={{ state, dispatch }}>
            {children}
        </BuilderContext.Provider>
    );
}

// 4. Hook for consumers to use the context
export function useBuilder() {
    const ctx = useContext(BuilderContext);
    if (!ctx) throw new Error('useBuilder must be inside BuilderProvider');
    return ctx;
}
```

### Usage in a Tab Component

```typescript
export function ConnectionTab() {
    const { state, dispatch } = useBuilder();  // ← pull current state + dispatch fn

    const setField = useCallback(
        (path: string, value: any) =>
            dispatch({ type: 'SET_FIELD', path, value }),
        [dispatch]
    );

    return (
        <Select
            value={state.account}
            onChange={(e, { value }) => setField('account', value)}
        />
    );
}
```

### Deep Dive: The Dot-Path Helper

One powerful feature is **dot-path updates**, used to update nested state safely:

```typescript
// Example: state.streams[0].retriever.type needs to change
dispatch({
    type: 'SET_FIELD',
    path: 'streams[0].retriever.type',
    value: 'HttpRetriever'
});
```

This requires a helper function that:
1. Splits the path into keys: `['streams', '0', 'retriever', 'type']`
2. Recursively descends, creating new objects at each level (immutable!)
3. Sets the value at the leaf

**Why immutability?** React re-renders when state changes. If you mutate the old object, React might not detect the change.

```typescript
// ❌ WRONG: mutates old state
state.streams[0].retriever.type = 'HttpRetriever';

// ✅ RIGHT: creates new object at each level
const newStreams = [
    {
        ...state.streams[0],
        retriever: {
            ...state.streams[0].retriever,
            type: 'HttpRetriever'
        }
    },
    ...state.streams.slice(1),
];
```

---

## 7. Schema-Driven Forms (the key pattern)

The DataMappingTab doesn't hardcode fields for each API component type. Instead, it **generates forms from a YAML schema at build time**.

### How it works

```
Build time:
  declarative_component_schema.yaml  →  generate-form-schema.js  →  form-schema.ts
  (5127 lines, 125 definitions)         (Node.js script)             (765 lines, 53 components)

Runtime:
  form-schema.ts  →  SchemaSection  →  TypeSelector + SchemaField[]
```

**Step 1: Build-time generation** (`scripts/generate-form-schema.js`)

The Airbyte schema defines every component type with titles, descriptions, enums, defaults.
The generator reads this YAML and outputs TypeScript:

```typescript
// form-schema.ts (auto-generated)
export const PAGINATORS: ComponentFormDef[] = [
  { type: "OffsetIncrement", category: "paginator", label: "Offset Increment",
    supported: true,
    description: "Pagination strategy that uses offset/limit parameters.",
    fields: [
      { key: "page_size", label: "Page Size", type: "number", required: false,
        helpText: "The number of records to request per page.", defaultValue: 100 },
      // ...more fields
    ] },
  { type: "CursorPagination", category: "paginator", label: "Cursor Pagination",
    supported: true, /* ... */ },
  // ...
];
```

**Step 2: Generic rendering** (`SchemaSection` + `SchemaField`)

```tsx
// DataMappingTab.tsx — just wires categories to sections
<SchemaSection
    title="Pagination"
    components={PAGINATORS}          // ← from form-schema.ts
    basePath="streams[0].retriever.paginator"
/>
```

`SchemaSection` renders:
1. A `TypeSelector` dropdown (e.g., "Offset Increment" / "Cursor Pagination")
2. `SchemaField` for each field in the selected type's definition

`SchemaField` dispatches by field type:
- `text` → `<Text />`
- `number` → `<Number />`
- `boolean` → `<Switch />`
- `enum` → `<Select />` with options from schema
- `template` → `<Text />` with `{{ }}` indicator (supports Jinja2)
- `object` → recursive nested fields
- `array` → list with add/remove buttons

### Why this matters

When the engine adds a new component type:
1. Register it in Python (`@component("NewType")`)
2. Add it to the YAML schema (or it's already there from Airbyte)
3. Run `yarn generate:schema` → form-schema.ts updates
4. The form automatically renders fields for the new type

**Zero React code changes needed** for new component types. The form system is fully generic.

### Supported vs. unsupported types

The generator checks each type against a hardcoded SUPPORTED_TYPES list (mirroring the Python REGISTRY). Unsupported types show in the dropdown with "(unsupported)" and trigger a `ValidationBanner` warning.

---

## 8. The Four Builder Steps

The wizard uses a `StepBar` with 4 steps. Each step renders a dedicated component. The footer shows contextual navigation (Cancel/Next for steps 1-3, Back/Save for step 4). Step state is local (`useState`), not in the reducer.

### Step 1: ConnectionTab

**What it collects:**
- Stream name (unique identifier for the input)
- Account (authentication — links to UCC Configuration page)
- Base URL (the API endpoint root)
- HTTP method (GET, POST)

**Layout:** Two card sections ("Stream Identity" and "API Connection") with `SectionHeader` icons and help tooltips. Auth type shown as a `Badge` when an account is selected.

**Key code patterns:**

```typescript
// Fetch accounts from Splunk
const [accounts, setAccounts] = useState<AccountSummary[]>([]);

useEffect(() => {
    (async () => {
        try {
            const accts = await listAccounts();
            setAccounts(accts);
        } catch {
            // graceful failure if accounts not configured
        }
    })();
}, []);

// Render dropdown
<Select
    value={state.account}
    onChange={(e, { value }) => setField('account', value)}
>
    {accounts.map(acct =>
        <Select.Option key={acct.name} label={acct.name} value={acct.name} />
    )}
</Select>
```

**Why separate local state (accounts) from context state (form)?**
- UI state (loading) is local to this component
- Persistent state (form values) lives in context
- Mixing them causes unnecessary re-renders

### Tab 2: DataMappingTab

**What it collects (via schema-driven forms):**
- Record Selector: how to extract records from JSON responses (DpathExtractor, field_path)
- Pagination: strategy for multi-page APIs (offset, cursor, page increment)
- Incremental Sync: cursor-based sync for only-new-data collection
- Transformations: field additions, removals, key normalization (array — user can add multiple)
- Error Handling: retry strategies, backoff, response filters

**Why schema-driven?** The Airbyte declarative schema defines 50+ component types. Hardcoding forms for each is unmaintainable. Instead, `SchemaSection` renders forms generically from the schema. See Section 7 above.

### Tab 3: SplunkOutputTab

**What it collects:**
- **Index:** Which Splunk index to write events to (dropdown from Splunk's data/indexes)
- **Sourcetype:** How to identify this data type in Splunk (default: `urc:api:json`)
- **Timestamp field:** Which API response field becomes Splunk's `_time` — detected from test results with a radio picker showing sample values and parsed previews

### Tab 4: ScheduleTagsTab

**What it collects:**
- **Interval:** How often to collect data (10–86400 seconds)
- **Tags:** Comma-separated labels for organizing streams in the dashboard
- **Debug toggle:** Per-stream verbose logging (enriches AOP hooks with URL, stream_slice, method args)
- **Enable/Disable toggle:** Controls whether the stream runs

### Tab 5: TestPreviewTab

**What it does:**
1. Takes all form state from tabs 1–4
2. Builds a manifest YAML
3. Calls a Splunk REST endpoint to test the collection
4. Returns sample data + detected fields
5. Allows user to preview before saving

**Key code:**

```typescript
const handleTest = async () => {
    setLoading(true);
    try {
        // Serialize form → YAML
        const manifest = buildManifestYAML(state);

        // Call Splunk test endpoint
        const result = await testCollection(manifest);

        // Result includes: sample records, field types, validation errors
        setTestResult(result);
    } catch (err) {
        setError(err.message);
    } finally {
        setLoading(false);
    }
};
```

---

## 8. TypeScript Types and Validation

TypeScript is used to prevent runtime errors at compile-time. Key interfaces are in `types.ts`.

### Core Types

```typescript
// Represents one REST collection ("stream")
export interface BuilderState {
    account: string;           // "my_oauth_account"
    baseUrl: string;          // "https://api.example.com/v2"
    method: string;           // "GET", "POST"
    interval: number;         // 300 (seconds)
    tags: string;             // "app:urc,source:api"
    index: string;            // "default"
    sourcetype: string;       // "urc:api:json"
    disabled: boolean;
    manifest: string;         // YAML as string (for display)
    streams: StreamDefinition[];  // Airbyte-format definitions
}

// What's returned from the Splunk test endpoint
export interface TestResult {
    status: 'success' | 'error';
    records: { stream: string; data: Record<string, any> }[];
    recordCount: number;
    validationResults: ValidationResult[];
    detectedFields: DetectedField[];   // Extracted field types
    timestampCandidates: TimestampCandidate[];
}

// Field auto-detected from JSON response
export interface DetectedField {
    name: string;      // "user_id"
    type: string;      // "integer", "string"
    sample: any;       // Example value: 12345
}
```

### Why Types Matter

```typescript
// ❌ Without types: might pass wrong data
dispatch({ type: 'SET_FIELD', path: 'interval', value: 'every 5 min' });
// Oops! interval should be a number, not a string

// ✅ With types: caught at compile-time
dispatch({
    type: 'SET_FIELD',
    path: 'interval',
    value: 'every 5 min'  // ← TypeScript error: string is not assignable to number
});

// Even better: enums prevent typos
type BuilderActionType = 'SET_FIELD' | 'LOAD_INPUT' | 'RESET';
dispatch({ type: 'SET_FEILD' });  // ← TypeScript error: typo caught!
```

---

## 9. REST API Communication

The `services/splunk-api.ts` file wraps Splunk's REST API. Why wrap it?

- **Centralized:** All API logic in one place
- **Error handling:** UCC endpoints sometimes return unhelpful errors
- **Type-safe:** Return types match our interfaces
- **Easy to mock:** For unit tests

### Pattern: Splunk REST Wrapper

```typescript
async function splunkFetch(path: string, init?: RequestInit): Promise<any> {
    // Create full URL using @splunk/splunk-utils
    const url = createRESTURL(path);  // → /en-US/splunkd/__raw/data/inputs/urc_app_input

    // Add JSON output mode
    const separator = url.includes('?') ? '&' : '?';
    const fullUrl = `${url}${separator}output_mode=json`;

    // Fetch with credentials (important: include cookies)
    const resp = await fetch(fullUrl, {
        credentials: 'include',
        ...init,
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            ...init?.headers,
        },
    });

    // Error handling
    if (!resp.ok) {
        const text = await resp.text();
        let message = `HTTP ${resp.status}`;
        try {
            const json = JSON.parse(text);
            if (json.messages?.[0]?.text) {
                message = json.messages[0].text;  // Use UCC error message if available
            }
        } catch {
            // Fall back to default
        }
        throw new Error(message);
    }

    return resp.json();
}
```

### Parsing Splunk's Response Format

Splunk's REST API returns a specific structure:

```json
{
    "entry": [
        {
            "name": "my_input",
            "content": {
                "account": "my_account",
                "interval": "300",
                "disabled": "0"
            }
        }
    ]
}
```

Parsing helper:

```typescript
function parseEntries<T>(response: any, mapper: (name: string, content: any) => T): T[] {
    const entries = response?.entry || [];
    return entries.map(e => mapper(e.name, e.content || {}));
}

// Usage:
const inputs = parseEntries(response, (name, content) => ({
    name,
    interval: parseInt(content.interval, 10) || 300,
    disabled: content.disabled === '1',
}));
```

### Working with Accounts

```typescript
export async function listAccounts(): Promise<AccountSummary[]> {
    const data = await splunkFetch('data/accounts/urc_account');
    return parseEntries(data, (name, content) => ({
        name,
        authType: content.auth_type || 'unknown',
    }));
}

export async function getAccount(name: string): Promise<AccountSummary> {
    const data = await splunkFetch(`data/accounts/urc_account/${encodeURIComponent(name)}`);
    const entries = parseEntries<AccountSummary>(data, ...);
    return entries[0];
}
```

### Why encode URIs?

```typescript
const name = "my:account";  // Contains special characters
// ❌ WRONG: /data/accounts/my:account  (might break URL parsing)
// ✅ RIGHT: /data/accounts/my%3Aaccount  (safely encoded)
splunkFetch(`data/accounts/${encodeURIComponent(name)}`);
```

---

## 10. Common Patterns and Best Practices

### Pattern 1: useCallback for Dispatch Wrappers

When passing dispatch functions to child components, wrap them in `useCallback` to avoid unnecessary re-renders:

```typescript
// ❌ Creates new function every render
<ChildComponent setField={(path, value) => dispatch({ type: 'SET_FIELD', path, value })} />

// ✅ Memoized function; only recreated if dispatch changes
const setField = useCallback(
    (path: string, value: any) => dispatch({ type: 'SET_FIELD', path, value }),
    [dispatch]
);
<ChildComponent setField={setField} />
```

### Pattern 2: Abort Fetch Requests on Unmount

When fetching data in useEffect, always cleanup to prevent memory leaks:

```typescript
useEffect(() => {
    let cancelled = false;
    (async () => {
        try {
            const accounts = await listAccounts();
            if (!cancelled) setAccounts(accounts);  // ← Don't update if unmounted
        } catch {
            if (!cancelled) setError('Failed to load accounts');
        }
    })();
    return () => { cancelled = true; };  // ← Cleanup: set flag
}, []);
```

### Pattern 3: Styled-Components for Theming

All UI styling uses `styled-components` to integrate with Splunk's theme system:

```typescript
import styled from 'styled-components';
import { variables } from '@splunk/themes';

const TabContent = styled.div`
    padding: ${variables.spacingLarge} 0;
    max-width: 640px;
    color: ${variables.contentColorDefault};
`;

// Renders <div> with colors/spacing from Splunk theme
<TabContent>Content here</TabContent>
```

### Pattern 4: Loading States + Error Handling

Always show loading indicators and graceful error handling:

```typescript
const [accounts, setAccounts] = useState<AccountSummary[]>([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState<string | null>(null);

useEffect(() => {
    (async () => {
        try {
            const accts = await listAccounts();
            setAccounts(accts);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    })();
}, []);

return (
    <>
        {loading && <WaitSpinner />}
        {error && <Message type="error">{error}</Message>}
        {!loading && !error && <Select>...</Select>}
    </>
);
```

---

## 11. Development Workflow

### Quick Development Cycle

1. **Open React code** in VS Code
   ```bash
   code /workspace/react/packages/urc-builder/src/
   ```

2. **Start dev server** with hot reload
   ```bash
   cd /workspace/react
   yarn workspace @splunk/urc-app run start:app
   ```
   This watches for file changes and rebuilds.

3. **Link to Splunk**
   ```bash
   yarn workspace @splunk/urc-app run link:app
   ```
   Copies build output to Splunk app directory.

4. **Refresh browser** at `http://localhost:8000/app/urc_app/...`
   Your changes appear instantly.

5. **Commit when ready**
   ```bash
   cd /workspace/react
   yarn run format  # Auto-format code
   yarn run lint    # Check for errors
   ```

### Testing

Run unit tests:

```bash
# All packages
yarn test

# One package
yarn workspace @splunk/urc-builder test

# Watch mode (re-run on file change)
yarn workspace @splunk/urc-builder test:watch
```

### Example: Adding a New Field to ConnectionTab

1. **Update the type** (`types.ts`):
   ```typescript
   export interface BuilderState {
       account: string;
       timeout: number;  // ← New field
   }
   ```

2. **Update INITIAL_STATE** (`BuilderContext.tsx`):
   ```typescript
   const INITIAL_STATE: BuilderState = {
       account: '',
       timeout: 30,  // ← New field
   };
   ```

3. **Add UI in ConnectionTab** (`tabs/ConnectionTab.tsx`):
   ```typescript
   <ControlGroup label="Timeout (seconds)">
       <Text
           value={String(state.timeout)}
           onChange={(e, { value }) => setField('timeout', parseInt(value, 10))}
       />
   </ControlGroup>
   ```

4. **Test it**:
   ```bash
   yarn workspace @splunk/urc-builder test:watch
   # See the new field in the test snapshots
   ```

5. **Build and link**:
   ```bash
   yarn build
   yarn workspace @splunk/urc-app run link:app
   ```

6. **Verify in Splunk**: The new field should appear in the form.

---

## Summary: The Big Picture

```
┌─ React Code (TypeScript)
│  ├─ Components render UI
│  ├─ Context + Reducer manage state
│  ├─ Services fetch from Splunk
│  └─ Types keep everything type-safe
│
├─ Build: yarn build
│  ├─ TypeScript compiles to JS
│  ├─ Webpack bundles deps
│  └─ Output: dist/ (library) or stage/ (app)
│
├─ Link: yarn link:app
│  └─ Copies stage/ → /workspace/ucc/output/urc_app/
│
├─ Splunk loads it
│  ├─ Mako template wraps React component
│  ├─ User interacts with form
│  └─ REST calls go to Python UCC handlers
│
└─ User clicks Save
   └─ Manifest YAML is created and stored
```

The React UI is the user-facing layer. The Python code (UCC, modular input handler) is the backend. Together they form a complete Splunk add-on.

---

## Glossary

- **Monorepo:** Single git repo with multiple npm packages (managed by yarn workspaces)
- **Webpack:** Bundles React code + dependencies into a single file
- **TSX:** TypeScript + JSX (React XML syntax)
- **Context + Reducer:** React pattern for global state management
- **Mako:** Splunk's template language (Python-like)
- **UCC:** Universal Configuration Console — framework for generating Splunk add-on boilerplate
- **KV Store:** Splunk's key-value database for storing configuration
- **Manifest:** YAML file describing an API collection (format for Airbyte CDK)
- **Sourcetype:** Splunk term for event type (e.g., "urc:api:json")

---

## Next Steps

1. **Run a local build:** `task dev:up` to start Splunk, then navigate to the app
2. **Explore the components:** Open `src/builder/tabs/*.tsx` to read real code
3. **Add a test:** `yarn workspace @splunk/urc-builder test` to run existing tests
4. **Modify ConnectionTab:** Add a new field and watch hot reload work
5. **Read the UCC guide:** See [ucc-golden-path.md](ucc-golden-path.md) for backend context

