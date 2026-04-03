# Design Document: GUI Polish

## Overview

This design covers 18 features that elevate the URC React UI from MVP to production-ready. All new code follows the established patterns: `DashboardContext` for dashboard state, `BuilderContext` with `useReducer` for the wizard, Splunk React UI primitives, styled-components with `@splunk/themes` variables, and `splunk-api.ts` as the sole API layer. Each feature is designed as an isolated, testable unit (hook, utility, or component) so that test-a-little/develop-a-little works naturally.

Fine-grained collection metrics (event throughput, checkpoint cursors) are handled by a native Splunk Dashboard v2 JSON definition rather than custom React, since Splunk's search commands are the right tool for time-series data.

## Code Reuse Analysis

### Existing Components to Leverage
- **DashboardContext** (`context/DashboardContext.tsx`): Extend with sort, search, pagination, and grouping state; keep the existing tag/status filter logic intact
- **BuilderContext** (`context/BuilderContext.tsx`): Add `isDirty` tracking, draft persistence, and validation state; leverage existing `SET_FIELD` reducer action and dot-path updater
- **StreamTable** (`dashboard/StreamTable.tsx`): Add sortable headers, clone action, last-run column, pagination footer; keep existing column/row/action patterns
- **TagFilter** (`dashboard/TagFilter.tsx`): Add search input and grouping dropdown alongside existing RadioBar + Chip pattern
- **BulkActions** (`dashboard/BulkActions.tsx`): Add "Export Selected" button to existing action bar
- **SplunkOutputTab** (`builder/tabs/SplunkOutputTab.tsx`): Replace comma-separated text input with tag autocomplete chip input
- **splunk-api.ts** (`services/splunk-api.ts`): Add `getStreamHealth()`, `cloneInput()` helper; keep existing `splunkFetch` wrapper and error parsing
- **manifest-serializer.ts** (`services/manifest-serializer.ts`): Reuse for export YAML generation and import YAML parsing
- **content.ts**: Add new UI copy entries for all new features (tooltips, labels, empty states, error messages)

### Integration Points
- **Splunk REST API**: Existing `data/inputs/urc_app_input` endpoints for all CRUD; new `urc_app/stream_health` endpoint for error status (or KV store lookup)
- **Browser localStorage**: New integration for draft auto-save (keyed by `urc-draft-{mode}-{inputName}`)
- **Splunk Dashboard v2**: New `default/data/ui/views/collection_metrics.xml` file referencing a JSON dashboard definition

---

## Architecture

### Module Map

```
src/
  context/
    DashboardContext.tsx    ← EXTEND: add sort, search, pagination, grouping state
    BuilderContext.tsx      ← EXTEND: add isDirty tracking, draft save/restore actions
  dashboard/
    StreamDashboard.tsx     ← EXTEND: compose new filter bar layout
    StreamTable.tsx         ← EXTEND: sortable headers, clone action, last-run column, pagination
    TagFilter.tsx           ← EXTEND: add search input, group-by dropdown
    BulkActions.tsx         ← EXTEND: add Export Selected button
    SummaryCards.tsx        ← EXTEND: add error count card
  builder/
    StreamBuilder.tsx       ← EXTEND: unsaved-changes guard, draft restore banner
    tabs/
      ConnectionTab.tsx     ← EXTEND: inline validation, name uniqueness check
      DataMappingTab.tsx    ← EXTEND: cross-field warnings
      SplunkOutputTab.tsx   ← EXTEND: tag autocomplete chip input
      TestPreviewTab.tsx    ← EXTEND: transformation before/after preview
  hooks/                   ← NEW directory
    useSort.ts             ← Sorting logic (column, direction, comparator)
    usePagination.ts       ← Pagination logic (page, pageSize, slice)
    useSearch.ts           ← Debounced search filter
    useGroupBy.ts          ← Grouping logic (key, grouped data)
    useDraftPersistence.ts ← localStorage save/restore with debounce
    useFieldValidation.ts  ← Per-field validation rules + onBlur handler
    useUnsavedGuard.ts     ← beforeunload + navigation intercept
  utils/                   ← NEW directory
    validators.ts          ← Pure validation functions (URL, name, interval, cross-field)
    export-import.ts       ← YAML export/import with conflict detection
    tag-utils.ts           ← Tag parsing, dedup, autocomplete helpers
  services/
    splunk-api.ts          ← EXTEND: getStreamHealth(), exportStream()
    manifest-serializer.ts ← Reuse as-is for import/export
  types.ts                 ← EXTEND: new interfaces
  content.ts               ← EXTEND: new UI copy
```

---

## Components and Interfaces

### A. Dashboard Hooks (new `hooks/` directory)

#### `useSort(items: T[], defaultKey?: string)`
- **Purpose:** Reusable sorting for any array of objects
- **Interface:**
  ```typescript
  interface SortState { key: string | null; direction: 'asc' | 'desc' }
  function useSort<T>(items: T[], defaultKey?: string): {
    sortedItems: T[];
    sortKey: string | null;
    sortDir: 'asc' | 'desc';
    onSort: (key: string) => void;  // toggles direction if same key
  }
  ```
- **Dependencies:** None (pure logic)
- **Reuses:** Existing `filteredInputs` as input array

#### `usePagination(items: T[], defaultPageSize?: number)`
- **Purpose:** Client-side pagination with page-size control
- **Interface:**
  ```typescript
  function usePagination<T>(items: T[], defaultPageSize?: number): {
    pagedItems: T[];
    page: number;
    pageSize: number;
    totalPages: number;
    setPage: (n: number) => void;
    setPageSize: (n: number) => void;
  }
  ```
- **Dependencies:** None (pure logic)
- **Resets page to 0** when items array identity changes (filter/sort change)

#### `useSearch(items: T[], keys: string[], debounceMs?: number)`
- **Purpose:** Debounced text search across specified fields
- **Interface:**
  ```typescript
  function useSearch<T>(items: T[], keys: (keyof T)[], debounceMs?: number): {
    query: string;
    setQuery: (q: string) => void;
    searchedItems: T[];
  }
  ```
- **Dependencies:** None (uses `setTimeout` for debounce)
- **Search logic:** Case-insensitive substring match on any of the specified keys

#### `useGroupBy(items: T[], groupKey: string | null)`
- **Purpose:** Groups items by a field value, returns ordered groups
- **Interface:**
  ```typescript
  interface Group<T> { label: string; items: T[] }
  function useGroupBy<T>(items: T[], groupKey: string | null): {
    groups: Group<T>[] | null;  // null when groupKey is null (flat mode)
    groupKey: string | null;
    setGroupKey: (key: string | null) => void;
  }
  ```
- **Dependencies:** None (pure logic)
- **Tag grouping:** When groupKey is `'tags'`, a stream with multiple tags appears in each matching group

---

### B. Builder Hooks (new in `hooks/`)

#### `useDraftPersistence(state: BuilderState, mode: string, inputName: string | null)`
- **Purpose:** Auto-save form state to localStorage, offer restore on load
- **Interface:**
  ```typescript
  function useDraftPersistence(state: BuilderState, mode: string, inputName: string | null): {
    hasDraft: boolean;
    restoreDraft: () => BuilderState;
    discardDraft: () => void;
    clearDraft: () => void;  // called after successful save
  }
  ```
- **Storage key:** `urc-draft-${mode}-${inputName || 'new'}`
- **Debounce:** 2000ms after last state change
- **Excludes:** `testResult`, `testLoading`, `validationResults` from persisted state

#### `useFieldValidation(existingNames: string[])`
- **Purpose:** Per-field validation rules with onBlur integration
- **Interface:**
  ```typescript
  interface FieldError { field: string; message: string }
  function useFieldValidation(existingNames: string[]): {
    errors: Record<string, string>;  // field path -> error message
    validateField: (path: string, value: any) => string | null;
    clearError: (path: string) => void;
    validateCrossField: (state: BuilderState) => FieldError[];
  }
  ```
- **Dependencies:** `validators.ts` for pure validation functions
- **Reuses:** Existing `ControlGroup` error styling via `error` prop

#### `useUnsavedGuard(isDirty: boolean)`
- **Purpose:** Warn on navigation when form has unsaved changes
- **Interface:**
  ```typescript
  function useUnsavedGuard(isDirty: boolean): void  // side-effect only
  ```
- **Implementation:** Registers `beforeunload` event + overrides `window.location.href` setter via a ref-based approach; shows Splunk UI `Modal` confirmation on in-app navigation (back button, cancel)
- **Dependencies:** None

---

### C. Utility Modules (new `utils/` directory)

#### `validators.ts`
- **Purpose:** Pure functions for all validation rules — no React, no side effects
- **Functions:**
  ```typescript
  isValidUrl(value: string): boolean          // http(s):// prefix, basic format
  isValidStreamName(value: string): boolean   // ^[a-zA-Z0-9_-]+$
  isValidInterval(value: number): boolean     // 10 <= n <= 86400
  isNameUnique(name: string, existing: string[]): boolean
  getCrossFieldWarnings(state: BuilderState): FieldError[]
  ```

#### `export-import.ts`
- **Purpose:** Export streams to YAML, import from YAML/JSON with conflict detection
- **Functions:**
  ```typescript
  exportStreams(inputs: InputSummary[]): string           // Multi-document YAML
  parseImportFile(content: string): ImportedStream[]      // YAML or JSON
  detectConflicts(imported: ImportedStream[], existing: InputSummary[]): ConflictReport
  ```
- **Reuses:** `js-yaml` (already a dependency), `manifest-serializer.ts` patterns

#### `tag-utils.ts`
- **Purpose:** Tag parsing, dedup, and autocomplete matching
- **Functions:**
  ```typescript
  parseTags(tagString: string): string[]              // split, trim, dedup
  formatTags(tags: string[]): string                  // join with ', '
  getAutocompleteSuggestions(query: string, allTags: string[]): string[]
  ```

---

### D. Dashboard Component Changes

#### StreamTable Enhancements
- **Sortable headers:** Replace `Table.HeadCell` content with a clickable wrapper that calls `onSort(key)`. Show `ChevronUp`/`ChevronDown` icon from `@splunk/react-icons` based on sort direction.
- **Clone action:** Add "Clone" `Menu.Item` with `Copy` icon in the row action dropdown. On click: navigate to `builder?clone={name}` — the builder reads `clone` param, loads the input, sets mode to `'create'`, and appends `-copy` to the name.
- **Last-run column:** New column between "Interval" and "Actions" showing relative timestamp + error icon. Data comes from `getStreamHealth()` API call (batched once on dashboard mount alongside `listInputs()`).
- **Pagination footer:** Render `Paginator` from `@splunk/react-ui/Paginator` below the table body. Show page size selector (10/25/50) as a small `Select`.

#### TagFilter Enhancements
- **Search input:** Add `Text` input with search icon at the left of the filter bar. Wired to `useSearch.setQuery`.
- **Group-by dropdown:** Add `Select` with options: None, Account, Tag. Wired to `useGroupBy.setGroupKey`. Placed at the right end of the filter bar.

#### BulkActions Enhancement
- **Export Selected:** Add "Export" `Button` with `Download` icon. Calls `exportStreams()` and triggers browser download.

#### SummaryCards Enhancement
- **Error count card:** Add 4th card (red) showing count of streams with recent errors. Data from `getStreamHealth()`.

---

### E. Builder Component Changes

#### StreamBuilder Changes
- **Draft restore banner:** On mount in create mode, check `useDraftPersistence.hasDraft`. If true, show a `Message` type="info" banner: "You have an unsaved draft. [Restore] [Discard]"
- **Unsaved guard:** Call `useUnsavedGuard(state.isDirty)` at the top level. isDirty is tracked by the reducer (set true on any `SET_FIELD`, reset on save/load).
- **Clone mode:** Read `clone` URL param. If present, call `getInput(clone)`, dispatch `LOAD_INPUT`, then override `mode: 'create'` and set name to `{original}-copy`.

#### ConnectionTab Changes
- **Name validation:** On blur, call `validateField('inputName', value)` which checks format + uniqueness. Show inline error via `ControlGroup` error prop.
- **URL validation:** On blur, validate URL format.
- **Read-only name in edit mode:** Set `Text` disabled prop when `mode === 'edit'`.

#### DataMappingTab Changes
- **Cross-field warnings:** At the top of the tab, render warnings from `validateCrossField(state)` as `Message` type="warning" components.

#### SplunkOutputTab Changes
- **Tag autocomplete:** Replace the simple `Text` input with a custom `TagInput` component:
  - Renders existing tags as `Chip` components (with X to remove)
  - Text input for new tag entry
  - Dropdown with autocomplete suggestions from `getAutocompleteSuggestions()`
  - On Enter or comma: add tag to array
  - State stored as `string[]` internally, serialized to comma-separated string on save

#### TestPreviewTab Changes
- **Transformation preview:** When test records exist and transformations are configured, show a two-panel layout:
  - Left: "Raw" — first record as received from API
  - Right: "Transformed" — record after applying configured transformations
  - Uses existing `RecordPreview` component for both panels
  - If no transformations configured, show single panel (current behavior)

---

### F. Template String Preview

- **Location:** New `TemplatePreview` sub-component rendered below template-type `SchemaField` inputs
- **Implementation:** Parse `{{ variable }}` expressions, resolve against a sample context (stream name, base URL, current date), render resolved string in a `Code` block
- **Unresolved variables:** Highlighted with `accentColorNegative` theme variable
- **No context:** Show raw template with grey placeholder markers

---

### G. Account Health Indicator

- **Data source:** Extend `listAccounts()` response parsing to include a `lastTestStatus` field, or maintain a client-side map keyed by account name updated whenever `testConnection()` runs
- **Display:** Small `Badge` (green/yellow/grey) next to account name in the builder's account `Select` dropdown and in the dashboard's Account column
- **Tooltip:** On hover, show "Last verified: (relative time)" or "Not tested yet"
- **Pragmatic approach:** Store last-test results in `sessionStorage` (per browser session) since there's no dedicated backend endpoint for account health — test results from any stream using that account update the health map

---

### H. Import/Export Flow

#### Export
1. Single stream: Row action menu -> "Export" -> browser downloads `{name}.yaml`
2. Bulk: Select rows -> "Export Selected" in BulkActions -> downloads `urc-streams-export.yaml`
3. YAML format: Multi-document (`---` separator) with metadata (name, account, index, sourcetype, interval, tags) + manifest

#### Import
1. Dashboard toolbar: "Import" button -> native file picker (accept `.yaml,.yml,.json`)
2. Parse file with `parseImportFile()`
3. Run `detectConflicts()` against existing inputs
4. Show import preview `Modal`:
   - Table of streams to import (name, account, status: new/conflict)
   - Conflict resolution per-stream: Skip / Rename (auto-suffix) / Overwrite
5. On confirm: Create/update streams via existing API, show summary toast

---

### I. Error Status & Last-Run (REQ-6, REQ-16)

#### Data Source Strategy
The collection engine already writes structured logs. We add a lightweight health summary:

**Option chosen: KV store health record**
- The Python collection engine writes a record per stream after each run to a KV store collection `urc_stream_health`:
  ```json
  {
    "_key": "{stream_name}",
    "last_run": "2026-04-03T10:30:00Z",
    "last_status": "success|error",
    "last_error": "Connection timeout after 30s",
    "last_record_count": 42,
    "error_count_24h": 3
  }
  ```
- New API function: `getStreamHealth(): Promise<StreamHealth[]>` — GET from `storage/collections/data/urc_stream_health`
- Dashboard fetches health data alongside inputs on mount, merges by stream name

#### UI Integration
- **Status badge:** StreamTable checks health data; if `last_status === 'error'`, badge shows "Error" (red) instead of "Active"
- **Last Run column:** Shows `last_run` as relative time ("2 min ago"); if error, prepends error icon
- **Error popover:** Click error badge -> `Popover` with last error message, timestamp, and 24h error count
- **Summary card:** 4th card shows error count
- **Error filter:** DashboardContext `selectedStatus: 'error'` now filters to streams where health `last_status === 'error'`

---

### J. Splunk Dashboard v2 — Collection Metrics (REQ-17)

#### File Location
`output/urc_app/default/data/ui/views/collection_metrics.xml`

The XML wrapper points to an inline JSON dashboard definition (standard Dashboard v2 pattern).

#### Dashboard Panels

| Panel | Type | SPL Search |
|-------|------|-----------|
| Stream Health Summary | Single Value x3 | `\| inputlookup urc_stream_health \| stats count(eval(last_status="success")) as healthy, count(eval(last_status="error")) as errored, count(eval(disabled="true")) as disabled` |
| Event Ingestion Rate | Timechart | `index=_internal sourcetype=urc_app_collection \| timechart span=5m count by stream_name` |
| Error Rate Over Time | Timechart | `index=_internal sourcetype=urc_app_collection level=ERROR \| timechart span=1h count by stream_name` |
| Top Errors by Stream | Table | `index=_internal sourcetype=urc_app_collection level=ERROR \| stats count, latest(message) as last_error by stream_name \| sort -count` |
| Last Collection per Stream | Table | `\| inputlookup urc_stream_health \| eval age=relative_time(now(), last_run) \| table stream_name, last_run, last_status, last_record_count, error_count_24h \| sort last_run` |

#### Time Picker
- Global time range input wired to all search-based panels
- KV store panels use "All Time" (they show current state)

---

## Data Models

### New Types (added to `types.ts`)

```typescript
// Stream health from KV store
interface StreamHealth {
  name: string;              // _key in KV store
  lastRun: string | null;    // ISO timestamp
  lastStatus: 'success' | 'error' | null;
  lastError: string | null;
  lastRecordCount: number;
  errorCount24h: number;
}

// Sort state for dashboard table
interface SortState {
  key: keyof InputSummary | 'lastRun' | null;
  direction: 'asc' | 'desc';
}

// Pagination state
interface PaginationState {
  page: number;
  pageSize: number;
}

// Import/export types
interface ExportedStream {
  name: string;
  account: string;
  base_url: string;
  index: string;
  sourcetype: string;
  interval: number;
  tags: string;
  manifest: string;          // YAML string
}

interface ImportedStream extends ExportedStream {
  conflict: 'none' | 'name';
  resolution: 'create' | 'skip' | 'rename' | 'overwrite';
}

interface ConflictReport {
  clean: ImportedStream[];
  conflicts: ImportedStream[];
}

// Field validation
interface FieldError {
  field: string;
  message: string;
}

// Account health (client-side session cache)
interface AccountHealth {
  name: string;
  lastTestStatus: 'success' | 'error' | null;
  lastTestTime: string | null;
}

// Tag input component
interface TagInputProps {
  tags: string[];
  allTags: string[];          // for autocomplete
  onChange: (tags: string[]) => void;
}
```

### Extended Types

```typescript
// Extend DashboardState
interface DashboardState {
  // ... existing fields ...
  healthMap: Record<string, StreamHealth>;  // keyed by stream name
  searchQuery: string;
  sortState: SortState;
  paginationState: PaginationState;
  groupBy: string | null;                   // 'account' | 'tags' | null
}

// Extend BuilderState
interface BuilderState {
  // ... existing fields ...
  isDirty: boolean;           // true after any SET_FIELD
}
```

---

## Error Handling

### Error Scenarios

1. **Stream health fetch fails**
   - **Handling:** Catch error, set `healthMap` to empty object, log warning
   - **User Impact:** Dashboard shows streams without health data; status column shows "Active"/"Disabled" only (graceful degradation)

2. **Import file parse error**
   - **Handling:** Catch YAML/JSON parse error, show `Message` type="error" in the import modal
   - **User Impact:** "Could not parse file. Please check the format is valid YAML or JSON."

3. **Import partial failure (some streams fail to create)**
   - **Handling:** Collect per-stream results, show summary with successes and failures
   - **User Impact:** "Imported 3 of 5 streams. 2 failed: {name}: {error}"

4. **localStorage quota exceeded (draft save)**
   - **Handling:** Catch `QuotaExceededError`, silently skip save
   - **User Impact:** None visible; draft just won't be available on crash recovery

5. **Clone target not found**
   - **Handling:** If `getInput(cloneName)` returns 404, redirect to builder in create mode with error toast
   - **User Impact:** "Stream '{name}' not found. Starting with a blank configuration."

---

## Testing Strategy

### Test-a-Little, Develop-a-Little Approach

Each feature is implemented with its tests in the same task. Tests are written for the hook/utility first (or alongside), then the component integration.

### Unit Testing (hooks + utilities)

| Module | Test File | Key Cases |
|--------|-----------|-----------|
| `useSort` | `hooks/useSort.unit.ts` | Sort asc/desc, toggle direction, null key, stable sort |
| `usePagination` | `hooks/usePagination.unit.ts` | Page bounds, page size change resets to 0, empty list |
| `useSearch` | `hooks/useSearch.unit.ts` | Case-insensitive match, multi-key search, debounce, empty query |
| `useGroupBy` | `hooks/useGroupBy.unit.ts` | Group by string field, group by tags (multi-group), null key |
| `validators` | `utils/validators.unit.ts` | URL formats, stream name chars, interval bounds, cross-field rules |
| `export-import` | `utils/export-import.unit.ts` | Round-trip YAML, conflict detection, malformed input |
| `tag-utils` | `utils/tag-utils.unit.ts` | Parse/format round-trip, autocomplete filtering, dedup |
| `useDraftPersistence` | `hooks/useDraftPersistence.unit.ts` | Save/restore/discard, debounce, excludes test state |
| `useFieldValidation` | `hooks/useFieldValidation.unit.ts` | Each validation rule, clear on correction, cross-field |

### Component Testing

| Component | Test File | Key Cases |
|-----------|-----------|-----------|
| StreamTable (sorting) | `dashboard/StreamTable.unit.tsx` | Click header sorts, click again reverses, arrow indicator |
| StreamTable (pagination) | `dashboard/StreamTable.unit.tsx` | Page navigation, page size change, filter resets page |
| TagFilter (search) | `dashboard/TagFilter.unit.tsx` | Type in search, results filter, clear restores |
| TagInput | `builder/TagInput.unit.tsx` | Add tag via Enter, add via comma, remove chip, autocomplete |
| Import modal | `dashboard/ImportModal.unit.tsx` | File parse, conflict display, resolution selection |
| Draft banner | `builder/DraftBanner.unit.tsx` | Show when draft exists, restore action, discard action |

### Test Infrastructure
- Use existing Jest + React Testing Library + `@testing-library/user-event` stack
- Mock `splunk-api.ts` functions with `jest.mock()` for component tests
- Mock `localStorage` with a simple in-memory implementation
- Mock `window.location` for navigation tests
- Test files follow existing convention: `*.unit.tsx` / `*.unit.ts`
