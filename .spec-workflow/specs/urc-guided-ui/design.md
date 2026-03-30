# Design Document — URC Guided UI (Schema-Driven)

## Overview

A guided connector builder for the URC Splunk add-on. Renders as a separate Splunk page (CIMplicity hybrid pattern) alongside the UCC-generated Configuration page. Built with `@splunk/react-ui` components, `@splunk/react-page` layout, and `styled-components` theming. Generates Airbyte-compatible manifest YAML from a visual form.

**Key design choice:** Forms are **schema-driven**, following Airbyte's Connector Builder pattern. The JSON Schema from `declarative_component_schema.yaml` defines field types, enums, oneOf polymorphism, defaults, required fields, and descriptions. Hand-coded card components control **layout and grouping** only — field rendering is automatic from the schema. A `SchemaFormRemainingFields` catch-all ensures nothing is hidden.

## Architecture Decision: Schema-Driven Forms

### Options Evaluated

| Option | Pros | Cons |
|--------|------|------|
| **A: Hand-coded sections** | Full UX control per field | Covers ~10 components, ignores 100+. Adding AsyncRetriever means writing a sub-application. |
| **B: Schema-driven forms** (chosen) | All 136 model types get working forms automatically. Cards add UX polish to common fields. Nothing hidden. | Needs a generic `SchemaFormControl` component. Polymorphic (oneOf) handling adds complexity. |
| **C: Full JSON Schema renderer** | Zero hand-coding | Poor UX for common paths — no grouping, no visual pickers, no help text |

**Decision: Option B** — Schema-driven with curated card layout for common components, auto-rendered fields for everything else. This is how Airbyte's Connector Builder works.

### How the schema drives the UI

```
declarative_component_schema.yaml (JSON Schema)
    │
    ▼ (shipped as static JSON to frontend at build time)
    │
SchemaFormControl({ path: "streams[0].retriever.requester.url", schema })
    │ ← reads: type, enum, description, required, default, oneOf
    │ ← applies widget overrides for known patterns (Jinja templates, secrets)
    ▼
Auto-rendered Splunk UI control (Text, Select, Switch, key-value editor, etc.)
```

**Cards define layout, schema defines fields.** Each card says "render these paths here." The actual field type, validation, description, and placeholder come from the schema. `SchemaFormRemainingFields` at the bottom auto-renders any properties not explicitly placed in a card — nothing is hidden.

## Code Reuse Analysis

### Existing Components to Leverage
- **`@splunk/react-ui` v5.9.0**: All form controls, layout (Card, TabLayout, Accordion), feedback (Message, WaitSpinner)
- **`@splunk/react-page` v8.2.1**: Page mounting in Splunk Enterprise layout
- **`@splunk/themes` v1.6.0**: Design tokens via styled-components
- **`@splunk/splunk-utils` v3.4.0**: CSRF tokens, theme loading, REST URL creation
- **Existing React monorepo**: Webpack 5 + Babel 7 + TypeScript build pipeline at `react/`
- **URC test manifests**: 5 pre-built manifests at `ucc/urc_app/tests/manifests/` for template gallery
- **`declarative_component_schema.yaml`**: 125 definitions, 123 with descriptions, 297/516 field descriptions — drives all form rendering

### Integration Points
- **UCC REST API**: Create/update/list inputs via `/servicesNS/-/urc_app/urc_app_input/`
- **Test Connection**: POST to `/servicesNS/-/urc_app/urc_app/test_connection`
- **Account listing**: GET `/servicesNS/-/urc_app/urc_app_account/`
- **UCC Configuration page**: Untouched — builder adds a new page, not a new tab

## UI Layout — ASCII Mockups

### Main Builder View (3-column layout)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  URC — Universal REST Client                    [Configuration] [Builder]  │
├──────────────┬──────────────────────────────────────┬───────────────────────┤
│              │                                      │                       │
│  STREAMS     │  Stream: github_repos                │  HELP & CONTEXT       │
│              │                                      │                       │
│  ┌────────┐  │  ┌─────────────────────────────────┐  │  (field description   │
│  │● repos │  │  │ ▼ Endpoint         [Card 1]     │  │   from schema, shown  │
│  ├────────┤  │  │   URL, Method, Headers, Params  │  │   for focused field)  │
│  │  issues│  │  │   ← SchemaFormControl per field  │  │                       │
│  ├────────┤  │  └─────────────────────────────────┘  │                       │
│  │  pulls │  │                                      │                       │
│  └────────┘  │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Authentication    [Card 2]     │  │                       │
│  [+ Add]     │  │   oneOf dropdown → type-specific │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│  ──────────  │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│  ACTIONS     │  │ ▶ Pagination        [Card 3]     │  │                       │
│  [Form|YAML] │  └─────────────────────────────────┘  │                       │
│  [Templates] │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Record Extraction  [Card 4]    │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Incremental Sync   [Card 5]    │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Transformations     [Card 6]    │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Error Handling      [Card 7]    │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
│              │  ┌─────────────────────────────────┐  │                       │
│              │  │ ▶ Advanced (auto-rendered)       │  │                       │
│              │  │   SchemaFormRemainingFields      │  │                       │
│              │  │   (anything not placed above)    │  │                       │
│              │  └─────────────────────────────────┘  │                       │
│              │                                      │                       │
├──────────────┴──────────────────────────────────────┴───────────────────────┤
│  TEST PANEL  [Account ▼]  [Test]  ⟳ 20 records, 230ms                      │
│  ┌──────────┬───────────┬──────────┬───────────┐                            │
│  │ Records  │ Request   │ Response │ Schema    │                            │
│  └──────────┴───────────┴──────────┴───────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AsyncRetriever Stream (extra tabs appear automatically)

When a stream uses `AsyncRetriever`, the schema's `retriever` oneOf resolves to the async type, and the card layout adds tabs for the sub-requesters:

```
┌─────────────────────────────────────────────────┐
│ ▼ Retriever: Async Job                          │
│                                                 │
│ Retriever Type  [Async (polling) ▼]             │
│                                                 │
│ ┌─ Creation Request ──────────────────────────┐ │
│ │  URL, Method, Headers, Body                 │ │
│ │  (SchemaFormControl for each field)          │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Polling Request ───────────────────────────┐ │
│ │  URL, Method, Headers                       │ │
│ │  Status mapping: running/completed/failed   │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Download Request ──────────────────────────┐ │
│ │  URL, Method, Decoder, Paginator            │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Advanced ──────────────────────────────────┐ │
│ │  abort_requester, delete_requester, timeout │ │
│ │  (SchemaFormRemainingFields)                 │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Component Architecture

### Package Structure

```
react/packages/
├── urc-builder/                    # Reusable component library
│   ├── src/
│   │   ├── index.ts                # Public exports
│   │   ├── ConnectorBuilder.tsx    # Root component (3-column layout)
│   │   │
│   │   ├── schema/                 # Schema-driven form engine
│   │   │   ├── SchemaFormControl.tsx    # Renders any field from JSON Schema
│   │   │   ├── SchemaFormRemainingFields.tsx  # Auto-renders unplaced fields
│   │   │   ├── OneOfSelector.tsx       # Dropdown for oneOf/anyOf polymorphism
│   │   │   ├── KeyValueEditor.tsx      # For object-typed fields (headers, params)
│   │   │   ├── ArrayEditor.tsx         # For array-typed fields (FormRows)
│   │   │   ├── JinjaInput.tsx          # Template-aware text input ({{ config['x'] }})
│   │   │   ├── widgetRegistry.ts       # Maps schema patterns → custom widgets
│   │   │   └── schemaUtils.ts          # Schema traversal, ref resolution, defaults
│   │   │
│   │   ├── cards/                  # Layout grouping — cards say WHICH fields, not HOW
│   │   │   ├── EndpointCard.tsx        # url, http_method, request_headers, request_parameters
│   │   │   ├── AuthCard.tsx            # authenticator (oneOf dropdown)
│   │   │   ├── PaginationCard.tsx      # paginator (oneOf dropdown)
│   │   │   ├── ExtractionCard.tsx      # record_selector, with response preview
│   │   │   ├── IncrementalCard.tsx     # incremental_sync (oneOf: datetime vs count)
│   │   │   ├── TransformCard.tsx       # transformations (array of oneOf)
│   │   │   ├── ErrorHandlingCard.tsx   # error_handler
│   │   │   └── AdvancedCard.tsx        # SchemaFormRemainingFields catch-all
│   │   │
│   │   ├── panels/
│   │   │   ├── StreamSidebar.tsx       # Left: stream list + actions + mode toggle
│   │   │   ├── HelpPanel.tsx           # Right: schema description for focused field
│   │   │   └── TestPanel.tsx           # Bottom: test connection results
│   │   │
│   │   ├── dialogs/
│   │   │   ├── SaveDialog.tsx          # Save/deploy modal
│   │   │   └── TemplateGallery.tsx     # Template picker modal
│   │   │
│   │   ├── editor/
│   │   │   └── YamlEditor.tsx          # CodeMirror 6 YAML editor
│   │   │
│   │   ├── state/
│   │   │   ├── useBuilderState.ts      # React state hook (manifest as source of truth)
│   │   │   └── manifestSerializer.ts   # manifest ↔ YAML conversion
│   │   │
│   │   └── utils/
│   │       ├── api.ts                  # Splunk REST helpers
│   │       └── templates.ts            # Built-in manifest templates
│   └── tests/
│
├── urc-app/                         # Splunk app wrapper (already scaffolded)
```

### Key Components

#### SchemaFormControl (the core)

Renders any single field from a JSON Schema property definition. This is the workhorse.

```typescript
interface SchemaFormControlProps {
  path: string;              // e.g., "streams[0].retriever.requester.url"
  schema: JSONSchemaProperty; // the schema definition for this field
  value: any;
  onChange: (path: string, value: any) => void;
  onFocus: (path: string) => void;
  errors?: Record<string, string>;
}
```

**Auto-rendering rules by schema type:**

| Schema type | Splunk component | Notes |
|-------------|-----------------|-------|
| `string` | `Text` | With placeholder from `examples` or `default` |
| `string` + `enum` | `Select` | Options from enum values |
| `string` + `interpolation_context` | `JinjaInput` | Shows `{{ config['...'] }}` hints |
| `integer` / `number` | `Number` | With `min`/`max` from `minimum`/`maximum` |
| `boolean` | `Switch` | |
| `object` (key-value) | `KeyValueEditor` | For headers, params |
| `array` of primitives | `FormRows` | Add/remove items |
| `array` of objects | `ArrayEditor` | Each item rendered recursively |
| `oneOf` / `anyOf` | `OneOfSelector` | Dropdown → conditional sub-form |
| `$ref` | Follow ref → render recursively | |

**Every field gets:** label (from property name, prettified), help text (from `description`), required indicator (from `required` array), validation errors.

#### OneOfSelector

Handles polymorphic fields — the most complex part. Auth, pagination, retriever type, decoder, etc.

```typescript
interface OneOfSelectorProps {
  path: string;
  schema: { oneOf: JSONSchema[] } | { anyOf: JSONSchema[] };
  value: any;
  onChange: (path: string, value: any) => void;
}
```

**How it works:**
1. Reads the `oneOf`/`anyOf` array from the schema
2. Each option has a `type` const that identifies it (e.g., `"BearerAuthenticator"`)
3. Renders a `Select` dropdown with type names + descriptions
4. On selection, shows the sub-schema's fields via recursive `SchemaFormControl`
5. Switching types preserves common fields, clears type-specific ones

#### SchemaFormRemainingFields

Catches anything not explicitly placed in a card. Prevents fields from being accidentally hidden.

```typescript
interface RemainingFieldsProps {
  schema: JSONSchema;          // full schema for current component
  placedPaths: Set<string>;    // paths already rendered by cards
  value: any;
  onChange: (path: string, value: any) => void;
}
```

Iterates schema properties, skips anything in `placedPaths`, renders the rest in a collapsible "Advanced" card.

#### Cards (layout only)

Cards don't render fields themselves — they declare which schema paths to place, then delegate to `SchemaFormControl`:

```typescript
// Example: EndpointCard
const EndpointCard: React.FC<CardProps> = ({ schema, value, onChange, onFocus }) => {
  const paths = [
    'retriever.requester.url',
    'retriever.requester.http_method',
    'retriever.requester.request_headers',
    'retriever.requester.request_parameters',
    'retriever.requester.request_body',
  ];

  return (
    <Accordion title="Endpoint">
      {paths.map(path => (
        <SchemaFormControl
          key={path}
          path={path}
          schema={resolveSchemaPath(schema, path)}
          value={getNestedValue(value, path)}
          onChange={onChange}
          onFocus={onFocus}
        />
      ))}
    </Accordion>
  );
};
```

**Cards register their paths** so `SchemaFormRemainingFields` knows what's already placed.

#### HelpPanel (schema-driven)

No more hand-written `helpContent.ts`. The help panel reads the `description` from the focused field's schema definition:

```typescript
const HelpPanel: React.FC<{ focusedPath: string; schema: JSONSchema }> = ({ focusedPath, schema }) => {
  const fieldSchema = resolveSchemaPath(schema, focusedPath);
  return (
    <Card>
      <Heading>{prettifyFieldName(focusedPath)}</Heading>
      <Typography>{fieldSchema?.description || 'No description available.'}</Typography>
      {fieldSchema?.examples && <Code>{JSON.stringify(fieldSchema.examples[0], null, 2)}</Code>}
      {fieldSchema?.default !== undefined && <Typography>Default: {String(fieldSchema.default)}</Typography>}
    </Card>
  );
};
```

### State Management

**The manifest IS the form state.** Following Airbyte's pattern, the form binds directly to the manifest structure — no separate `StreamConfig` intermediate type.

```typescript
interface BuilderState {
  mode: 'form' | 'yaml';
  manifest: ManifestDict;       // THE source of truth — same shape as the YAML
  activeStreamIndex: number;
  yamlString: string;            // kept in sync when in YAML mode
  testResults: TestResult | null;
  isDirty: boolean;
  focusedPath: string | null;    // drives help panel
}
```

This eliminates the `manifestSerializer` complexity — the form state IS the manifest dict. `js-yaml.dump(state.manifest)` produces the YAML. `js-yaml.load(yamlString)` produces the form state. No mapping layer.

### Widget Overrides

For the ~5% of fields that need special treatment beyond auto-rendering:

```typescript
// widgetRegistry.ts
const WIDGET_OVERRIDES: Record<string, WidgetOverride> = {
  // Jinja template fields get a special input with {{ }} hints
  '*.url': { widget: 'jinja', placeholder: 'https://api.example.com/v1/{{ config["resource"] }}' },
  '*.url_base': { widget: 'jinja' },
  '*.api_token': { widget: 'jinja', placeholder: "{{ config['api_key'] }}" },

  // Field path fields get a dot-notation hint
  '*.field_path': { widget: 'dpath', placeholder: 'data.items' },

  // Request body gets a multi-line JSON editor
  '*.request_body_json': { widget: 'json-textarea' },
};
```

These are ~15 overrides total, not 800 lines of hand-coded sections.

## Data Flow

```
User interacts with form
    │
    ▼
SchemaFormControl updates manifest dict at path
    │ onChange("streams[0].retriever.requester.url", "https://...")
    ▼
useBuilderState updates manifest
    │
    ├── mode === 'form': re-renders form controls
    ├── mode === 'yaml': js-yaml.dump(manifest) → updates editor
    │
    ▼ (on Test)
api.testConnection(js-yaml.dump(manifest), account)
    │ POST to /services/urc_app/test_connection
    ▼
TestPanel shows results
    │
    ▼ (on Save)
api.saveInput(name, { manifest: js-yaml.dump(manifest), interval, index, ... })
    │ POST to /servicesNS/-/urc_app/urc_app_input/
    ▼
Stored in Splunk inputs.conf as single manifest text field
```

## Schema Delivery

The JSON Schema is shipped to the frontend as a static asset at build time:

```
Build time:
  1. Read ucc/urc_app/schema/declarative_component_schema.yaml
  2. Convert to JSON (smaller, faster to parse)
  3. Bundle as import in urc-builder: import schema from './schema.json'

Runtime:
  SchemaFormControl reads schema properties on render
  No fetch, no server round-trip
```

## Build Pipeline

Already implemented via `task ucc:add-react` (scaffolding) and existing Taskfile targets:

```
task ucc:build-full
  ├── ucc:build       → UCC output (REST handlers, conf, UCC UI)
  ├── ucc:build-ui    → webpack builds react/packages/urc-app → stage/
  └── ucc:merge-ui    → copies stage/ into UCC output
```

## Navigation Integration

Already implemented in scaffolded files:

```xml
<!-- default/data/ui/nav/default.xml -->
<nav>
    <view name="builder" default="true"/>
    <view name="configuration"/>
    <view name="search"/>
</nav>
```

## Error Handling

1. **YAML Parse Error**: `js-yaml` error → inline marker in CodeMirror, line number
2. **Test Connection Failure**: Catch fetch error or HTTP status, show actionable message in TestPanel
3. **Manifest Validation Error**: Pydantic errors from test_connection mapped to form field paths
4. **Schema field missing**: SchemaFormControl renders gracefully if a path doesn't exist in schema
5. **Save Failure**: UCC REST API error → dialog shows message, offer rename or retry
6. **Session Lost**: `sessionStorage` persistence on every state change

## Testing Strategy

- **SchemaFormControl**: Render tests — given schema property of each type, verify correct Splunk component rendered
- **OneOfSelector**: Test type switching, field preservation, dropdown options from schema
- **SchemaFormRemainingFields**: Given schema + placed paths, verify only unplaced fields render
- **manifestSerializer**: Round-trip — manifest dict → YAML → manifest dict for all test fixtures
- **api.ts**: Mock Splunk REST responses, verify request format
- **E2E**: Template → test → save → verify input created

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

No react-hook-form or zod — the form state is a plain dict managed by `useBuilderState`. Splunk's `@splunk/react-ui` form controls handle their own local state.
