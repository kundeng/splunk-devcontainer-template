# Tasks: GUI Polish

## Phase 1: Dashboard Hooks & Utilities (Foundation)

- [x] 1. Create validation utilities and tests
  - Files: `react/packages/urc-builder/src/utils/validators.ts`, `react/packages/urc-builder/src/utils/validators.unit.ts`
  - Implement pure validation functions: `isValidUrl`, `isValidStreamName`, `isValidInterval`, `isNameUnique`, `getCrossFieldWarnings`
  - Write unit tests first, then implement functions to pass them
  - Purpose: Foundation for all form validation features (REQ-10, REQ-11, REQ-12)
  - _Leverage: `react/packages/urc-builder/src/types.ts` for BuilderState interface_
  - _Requirements: REQ-10, REQ-11, REQ-12_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript Developer specializing in pure utility functions and TDD. Task: Create `validators.ts` with pure validation functions and `validators.unit.ts` with comprehensive tests. Read the existing `types.ts` for the BuilderState interface. Write tests FIRST covering: valid/invalid URLs (http/https prefix, empty, malformed), stream name chars (alphanumeric, underscore, hyphen only), interval bounds (10-86400), name uniqueness check, and cross-field warnings (POST without body, incremental sync without cursor, pagination without record selector). Then implement the functions to pass all tests. Follow existing test convention: files end in `.unit.ts`, use Jest. Restrictions: No React imports, no side effects, pure functions only. Success: All tests pass, 100% branch coverage on validators._

- [x] 2. Create tag utilities and tests
  - Files: `react/packages/urc-builder/src/utils/tag-utils.ts`, `react/packages/urc-builder/src/utils/tag-utils.unit.ts`
  - Implement `parseTags`, `formatTags`, `getAutocompleteSuggestions`
  - Write unit tests alongside implementation
  - Purpose: Foundation for tag autocomplete and improved tag filtering (REQ-3)
  - _Leverage: Existing comma-separated tag pattern in `SplunkOutputTab.tsx` and `DashboardContext.tsx`_
  - _Requirements: REQ-3_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript Developer. Task: Create `tag-utils.ts` and `tag-utils.unit.ts`. Read the existing `DashboardContext.tsx` and `SplunkOutputTab.tsx` to see how tags are currently handled (comma-separated strings). Implement: `parseTags(tagString)` splits on comma, trims whitespace, deduplicates, removes empty strings; `formatTags(tags[])` joins with ", "; `getAutocompleteSuggestions(query, allTags)` returns tags matching the query prefix (case-insensitive), excluding already-selected tags. Write tests covering: empty string, single tag, multiple tags, whitespace handling, duplicate removal, autocomplete matching, empty query returns all. Restrictions: Pure functions, no React. Success: All tests pass, round-trip `formatTags(parseTags(s))` preserves semantics._

- [x] 3. Create useSort hook and tests
  - Files: `react/packages/urc-builder/src/hooks/useSort.ts`, `react/packages/urc-builder/src/hooks/useSort.unit.ts`
  - Implement generic sorting hook with toggle behavior
  - Purpose: Sortable column headers for stream table (REQ-1)
  - _Leverage: None (new standalone hook)_
  - _Requirements: REQ-1_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer specializing in custom hooks. Task: Create `useSort<T>` hook and tests. The hook takes `(items: T[], defaultKey?: string)` and returns `{ sortedItems, sortKey, sortDir, onSort }`. `onSort(key)` toggles direction if same key, else sets ascending. Sorting uses `String(a[key]).localeCompare(String(b[key]))` for strings, numeric comparison for numbers. Test: sort ascending/descending, toggle direction on same key click, new key resets to ascending, null key returns original order, stable sort for equal values, empty array. Use `renderHook` from `@testing-library/react`. Restrictions: Generic hook, no Splunk UI imports. Success: All tests pass, hook is reusable for any typed array._

- [x] 4. Create usePagination hook and tests
  - Files: `react/packages/urc-builder/src/hooks/usePagination.ts`, `react/packages/urc-builder/src/hooks/usePagination.unit.ts`
  - Implement client-side pagination with page-size control
  - Purpose: Table pagination for large stream lists (REQ-4)
  - _Leverage: None (new standalone hook)_
  - _Requirements: REQ-4_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `usePagination<T>` hook and tests. Takes `(items: T[], defaultPageSize = 25)`. Returns `{ pagedItems, page, pageSize, totalPages, setPage, setPageSize }`. Page resets to 0 when items array reference changes (use useEffect on items). setPageSize also resets to page 0. Clamp page to valid range. Test: correct slice for page 0/1/2, page size change resets to 0, items change resets to 0, empty array returns empty, last page with partial items, out-of-bounds page clamped. Restrictions: No UI imports. Success: All tests pass._

- [x] 5. Create useSearch hook and tests
  - Files: `react/packages/urc-builder/src/hooks/useSearch.ts`, `react/packages/urc-builder/src/hooks/useSearch.unit.ts`
  - Implement debounced text search across specified fields
  - Purpose: Stream search/filter by name or URL (REQ-2)
  - _Leverage: None (new standalone hook)_
  - _Requirements: REQ-2_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `useSearch<T>` hook and tests. Takes `(items: T[], keys: (keyof T)[], debounceMs = 300)`. Returns `{ query, setQuery, searchedItems }`. Search is case-insensitive substring match on any of the specified keys. Debounce the filtering using setTimeout. Test: matches on name field, matches on baseUrl field, case-insensitive, no match returns empty, empty query returns all items, multi-key search. For debounce testing use `jest.useFakeTimers()`. Restrictions: No UI imports. Success: All tests pass, debounce works correctly._

- [x] 6. Create useGroupBy hook and tests
  - Files: `react/packages/urc-builder/src/hooks/useGroupBy.ts`, `react/packages/urc-builder/src/hooks/useGroupBy.unit.ts`
  - Implement grouping logic for table display
  - Purpose: Group streams by account or tag (REQ-7)
  - _Leverage: `tag-utils.ts` for tag parsing when grouping by tags_
  - _Requirements: REQ-7_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `useGroupBy<T>` hook and tests. Takes `(items: T[], groupKey: string | null)`. Returns `{ groups, groupKey, setGroupKey }`. When groupKey is null, groups is null (flat mode). When groupKey is a string field like 'account', group by that field value. Special case: when groupKey is 'tags', use `parseTags` from tag-utils to split comma-separated tags and place each item in all matching groups. Each group: `{ label: string, items: T[] }`. Sort groups alphabetically by label. Test: null key returns null, group by string field, group by tags (multi-group), empty items, items with no value for group key go into "Ungrouped". Restrictions: Only import from tag-utils, no UI imports. Success: All tests pass._

---

## Phase 2: Dashboard UI Integration

- [x] 7. Integrate sorting into StreamTable
  - Files: `react/packages/urc-builder/src/dashboard/StreamTable.tsx`, `react/packages/urc-builder/src/dashboard/StreamTable.unit.tsx`
  - Add sortable column headers with direction indicators
  - Write component tests for sort interaction
  - Purpose: REQ-1 column sorting in the stream table
  - _Leverage: `hooks/useSort.ts`, existing `Table.HeadCell` pattern, `ChevronUp`/`ChevronDown` from `@splunk/react-icons`_
  - _Requirements: REQ-1_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Full-stack React Developer familiar with Splunk UI. Task: Read the existing `StreamTable.tsx` thoroughly. Import and wire `useSort` hook. Make column headers (Name, Status, Account, Interval) clickable — wrap header text in a styled clickable span. Show ChevronUp/ChevronDown icon next to the active sort column. Pass `filteredInputs` through `useSort` before rendering rows. Write `StreamTable.unit.tsx` tests: render table with mock data, click Name header sorts ascending, click again sorts descending, sort indicator visible on active column. Mock the DashboardContext provider for tests. Restrictions: Do not change the existing column layout or action menu. Keep Tags column non-sortable (multi-value). Success: Headers are clickable, sort works, arrow indicator shows, tests pass._

- [x] 8. Integrate search and group-by into TagFilter
  - Files: `react/packages/urc-builder/src/dashboard/TagFilter.tsx`, `react/packages/urc-builder/src/dashboard/StreamDashboard.tsx`, `react/packages/urc-builder/src/context/DashboardContext.tsx`, `react/packages/urc-builder/src/dashboard/TagFilter.unit.tsx`
  - Add search input and group-by dropdown to filter bar
  - Extend DashboardContext with searchQuery and groupBy state
  - Purpose: REQ-2 search filter and REQ-7 grouping
  - _Leverage: `hooks/useSearch.ts`, `hooks/useGroupBy.ts`, existing `TagFilter.tsx` layout, `Text` and `Select` from `@splunk/react-ui`_
  - _Requirements: REQ-2, REQ-7_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Full-stack React Developer. Task: Read existing `TagFilter.tsx`, `DashboardContext.tsx`, and `StreamDashboard.tsx`. Extend `DashboardContext` with `searchQuery`, `setSearchQuery`, `groupBy`, `setGroupBy` state. In `TagFilter.tsx` add a `Text` input with Search icon at the left of the filter bar for search, and a `Select` dropdown at the right with options None/Account/Tag for grouping. In `StreamDashboard.tsx`, wire `useSearch` (keys: name, baseUrl) and `useGroupBy` into the data pipeline: inputs -> filter -> search -> group. When grouped, render group headers in the table area. Write tests for `TagFilter.unit.tsx`: search input renders, typing filters results, group-by dropdown renders. Restrictions: Keep existing tag chip and status filter behavior unchanged. Success: Search filters streams by name/URL, group-by shows grouped sections, tests pass._

- [x] 9. Integrate pagination into StreamTable
  - Files: `react/packages/urc-builder/src/dashboard/StreamTable.tsx`, `react/packages/urc-builder/src/dashboard/StreamDashboard.tsx`
  - Add pagination footer below table with page-size selector
  - Purpose: REQ-4 table pagination
  - _Leverage: `hooks/usePagination.ts`, `Paginator` from `@splunk/react-ui/Paginator` (or custom), `Select` for page size_
  - _Requirements: REQ-4_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer familiar with Splunk UI. Task: Read existing `StreamTable.tsx` and `StreamDashboard.tsx`. Wire `usePagination` into the data pipeline after sort (inputs -> filter -> search -> sort -> paginate). Add a pagination footer below the table with prev/next buttons and page indicator. Add a small Select for page size (10/25/50). Hide pagination when total items are under the page size. Pagination resets to page 0 when filters/sort change (handled by the hook). Add to existing `StreamTable.unit.tsx`: test pagination appears with 30+ items, clicking next shows next page, page size change works. Restrictions: Use Splunk UI Button for prev/next if Paginator component is not available. Success: Table paginates at 25 rows default, controls work, hidden when unnecessary._

- [x] 10. Add stream clone action
  - Files: `react/packages/urc-builder/src/dashboard/StreamTable.tsx`, `react/packages/urc-builder/src/builder/StreamBuilder.tsx`
  - Add "Clone" to row action menu, handle clone param in builder
  - Purpose: REQ-5 stream cloning
  - _Leverage: Existing action menu pattern in `StreamTable.tsx`, existing `getInput` + `LOAD_INPUT` in `StreamBuilder.tsx`_
  - _Requirements: REQ-5_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Full-stack React Developer. Task: Read existing `StreamTable.tsx` action menu and `StreamBuilder.tsx` edit-mode loading. In StreamTable, add a "Clone" Menu.Item with Copy icon between Edit and Enable/Disable in the dropdown. On click, navigate to `builder?clone={name}`. In StreamBuilder, read the `clone` URL param. If present, load the input via `getInput(clone)`, dispatch `LOAD_INPUT`, then override mode to 'create' and set the name to `{original}-copy`. The name field must be editable. Add test: clone menu item renders, clicking it navigates with clone param. Restrictions: Do not modify the original stream. Keep edit mode loading separate from clone logic. Success: Clone opens builder pre-filled, mode is create, name is editable with -copy suffix._

---

## Phase 3: Builder UX Improvements

- [x] 11. Add unsaved changes guard
  - Files: `react/packages/urc-builder/src/hooks/useUnsavedGuard.ts`, `react/packages/urc-builder/src/hooks/useUnsavedGuard.unit.ts`, `react/packages/urc-builder/src/context/BuilderContext.tsx`, `react/packages/urc-builder/src/builder/StreamBuilder.tsx`
  - Track isDirty in reducer, warn on navigation
  - Purpose: REQ-8 unsaved changes warning
  - _Leverage: Existing `BuilderContext.tsx` reducer, `Modal` from `@splunk/react-ui`_
  - _Requirements: REQ-8_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Read existing `BuilderContext.tsx` reducer. Add `isDirty: boolean` to BuilderState (initial false). Set to true on any `SET_FIELD` action, reset to false on `LOAD_INPUT` and `RESET`. Create `useUnsavedGuard(isDirty)` hook that registers a `beforeunload` event listener when dirty. In `StreamBuilder.tsx`, use the hook and add a confirmation Modal that appears when the user clicks Cancel or Back while dirty. Test the hook: beforeunload registered when dirty, not registered when clean, cleanup on unmount. Restrictions: Do not block navigation when not dirty. Use Splunk UI Modal for the confirmation dialog. Success: Browser warns on page close when dirty, in-app Cancel shows modal, tests pass._

- [x] 12. Add draft auto-save
  - Files: `react/packages/urc-builder/src/hooks/useDraftPersistence.ts`, `react/packages/urc-builder/src/hooks/useDraftPersistence.unit.ts`, `react/packages/urc-builder/src/builder/StreamBuilder.tsx`
  - Auto-save form state to localStorage, restore banner on load
  - Purpose: REQ-9 draft persistence
  - _Leverage: Existing `BuilderContext.tsx` state, `Message` from `@splunk/react-ui`_
  - _Requirements: REQ-9_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `useDraftPersistence` hook. Takes `(state: BuilderState, mode: string, inputName: string | null)`. Returns `{ hasDraft, restoreDraft, discardDraft, clearDraft }`. Storage key: `urc-draft-${mode}-${inputName || 'new'}`. Debounce save at 2000ms using useEffect + setTimeout. Exclude `testResult`, `testLoading`, `validationResults` from saved state. On restore, return the saved BuilderState. In `StreamBuilder.tsx`, show a `Message` type="info" banner when hasDraft is true in create mode: "You have an unsaved draft. [Restore] [Discard]". Call clearDraft after successful save. Test: save triggers after debounce, restore returns saved state, discard clears storage, excludes test fields. Mock localStorage. Restrictions: Catch QuotaExceededError silently. Success: Drafts survive page reload, restore works, tests pass._

- [x] 13. Add real-time field validation
  - Files: `react/packages/urc-builder/src/hooks/useFieldValidation.ts`, `react/packages/urc-builder/src/hooks/useFieldValidation.unit.ts`, `react/packages/urc-builder/src/builder/tabs/ConnectionTab.tsx`
  - Validate fields on blur with inline error messages
  - Purpose: REQ-10 real-time validation, REQ-11 name uniqueness
  - _Leverage: `utils/validators.ts`, existing `ControlGroup` error prop, `splunk-api.ts` listInputs for existing names_
  - _Requirements: REQ-10, REQ-11_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `useFieldValidation(existingNames: string[])` hook. Returns `{ errors: Record<string, string>, validateField(path, value), clearError(path) }`. `validateField` uses validators from `validators.ts`: 'inputName' checks format + uniqueness, 'baseUrl' checks URL format, 'interval' checks range. Read existing `ConnectionTab.tsx`. Add onBlur handlers to Name (create mode only), Base URL, and make Name field disabled in edit mode. Pass error messages to `ControlGroup` error prop for inline display. Clear error on next valid change. Fetch existing names from `listInputs()` on mount. Test: each validation rule triggers on blur, errors clear on correction, edit mode name is read-only. Restrictions: Do not modify the reducer or other tabs in this task. Success: Invalid fields show inline errors on blur, name uniqueness checked, tests pass._

- [x] 14. Add cross-field validation warnings
  - Files: `react/packages/urc-builder/src/builder/tabs/DataMappingTab.tsx`, `react/packages/urc-builder/src/builder/tabs/ConnectionTab.tsx`
  - Show warnings for inconsistent configurations
  - Purpose: REQ-12 cross-field validation
  - _Leverage: `utils/validators.ts` getCrossFieldWarnings, existing `Message` component, `useBuilder` context_
  - _Requirements: REQ-12_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Read existing `DataMappingTab.tsx` and `ConnectionTab.tsx`. Import `getCrossFieldWarnings` from validators.ts. In each tab, call it with the current builder state and filter warnings relevant to that tab. Render matching warnings as `Message` type="warning" components at the top of the tab content. Warnings to implement: POST/PUT/PATCH without request body (ConnectionTab), incremental sync without cursor field (DataMappingTab), pagination without record selector path (DataMappingTab). Warnings should be dismissible but reappear if the condition persists on next render. Restrictions: Do not block navigation or saving based on warnings. These are advisory only. Success: Relevant warnings appear on the correct tabs when conditions are met, disappear when resolved._

- [x] 15. Add tag autocomplete chip input
  - Files: `react/packages/urc-builder/src/builder/components/TagInput.tsx`, `react/packages/urc-builder/src/builder/components/TagInput.unit.tsx`, `react/packages/urc-builder/src/builder/tabs/SplunkOutputTab.tsx`
  - Replace comma-separated text input with chip-based tag entry
  - Purpose: REQ-3 tag autocomplete and improved tag UX
  - _Leverage: `utils/tag-utils.ts`, `Chip` and `Text` from `@splunk/react-ui`, existing tag pattern in `SplunkOutputTab.tsx`_
  - _Requirements: REQ-3_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create a `TagInput` component. Props: `{ tags: string[], allTags: string[], onChange: (tags: string[]) => void }`. Renders existing tags as `Chip` components with X button to remove. Below chips, a `Text` input for typing new tags. On Enter or comma keypress, add the typed value as a new tag (trimmed, deduplicated). Show a dropdown of autocomplete suggestions from `getAutocompleteSuggestions()` below the input when typing. Read existing `SplunkOutputTab.tsx` to see the current tags field. Replace the comma-separated `Text` input with `TagInput`, converting between string[] (internal) and comma-separated string (state) using `parseTags`/`formatTags`. Wire `allTags` from DashboardContext or a prop. Test: renders existing tags as chips, remove chip updates tags, typing + Enter adds tag, comma adds tag, autocomplete shows matching suggestions, duplicate prevention. Restrictions: Keep the stored format as comma-separated string in BuilderState for backward compatibility. Success: Tag input is chip-based with autocomplete, tests pass._

---

## Phase 4: Operational Visibility & Advanced Features

- [x] 16. Add stream health API and error status
  - Files: `react/packages/urc-builder/src/services/splunk-api.ts`, `react/packages/urc-builder/src/types.ts`, `react/packages/urc-builder/src/context/DashboardContext.tsx`, `react/packages/urc-builder/src/dashboard/StreamTable.tsx`, `react/packages/urc-builder/src/dashboard/SummaryCards.tsx`
  - Add getStreamHealth API, integrate error status into dashboard
  - Purpose: REQ-6 error status, REQ-16 last-run indicator
  - _Leverage: Existing `splunkFetch` pattern, `Badge` and `Tooltip` from `@splunk/react-ui`, KV store REST endpoint pattern_
  - _Requirements: REQ-6, REQ-16_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Full-stack Developer. Task: Add `StreamHealth` interface to types.ts (name, lastRun, lastStatus, lastError, lastRecordCount, errorCount24h). Add `getStreamHealth()` to splunk-api.ts that GETs from `storage/collections/data/urc_stream_health` and parses the response. Extend DashboardContext with `healthMap: Record<string, StreamHealth>` state, fetch health data alongside inputs on mount. In StreamTable: add "Last Run" column showing relative timestamp (use `Date` math for "2 min ago" etc.), show error icon if last_status is error, show error badge instead of Active when errored. Add click handler on error badge to show Tooltip/Popover with last error message. In SummaryCards: add 4th card (red) showing error count. Extend the status filter so 'error' actually filters streams with lastStatus=error. Restrictions: Graceful degradation if health endpoint returns 404 (set empty healthMap). Success: Error status shows in dashboard, last-run column works, error filter works, summary card shows error count._

- [x] 17. Add account health indicator
  - Files: `react/packages/urc-builder/src/services/splunk-api.ts`, `react/packages/urc-builder/src/builder/tabs/ConnectionTab.tsx`, `react/packages/urc-builder/src/dashboard/StreamTable.tsx`
  - Show health icon next to account names based on last test results
  - Purpose: REQ-15 account health indicator
  - _Leverage: Existing `testConnection` result, `Badge` and `Tooltip` from `@splunk/react-ui`, `sessionStorage`_
  - _Requirements: REQ-15_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create a simple account health tracking system using sessionStorage. Key: `urc-account-health`. After each `testConnection()` call, update the health map with the account name, status (success/error from test result), and timestamp. In `ConnectionTab.tsx` account Select dropdown, show a small colored dot (green/yellow/grey) next to each account option based on health data. In `StreamTable.tsx` Account column, show a similar dot. Add a Tooltip on hover: "Last verified: X min ago" or "Not tested yet". Restrictions: This is session-only data, no backend storage needed. Do not block any UI on missing health data. Success: Account health dots appear, update after test connections, tooltips show last-verified time._

- [x] 18. Add transformation and template previews
  - Files: `react/packages/urc-builder/src/builder/test/TransformPreview.tsx`, `react/packages/urc-builder/src/builder/form/TemplatePreview.tsx`, `react/packages/urc-builder/src/builder/tabs/TestPreviewTab.tsx`, `react/packages/urc-builder/src/builder/form/SchemaField.tsx`
  - Show before/after for transformations, live preview for templates
  - Purpose: REQ-13 transformation preview, REQ-14 template preview
  - _Leverage: Existing `RecordPreview` component, existing `Code` from `@splunk/react-ui`, `SchemaField.tsx` template type handling_
  - _Requirements: REQ-13, REQ-14_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: React Developer. Task: Create `TransformPreview` component for TestPreviewTab. When test records exist and transformations are configured, show a two-column layout with "Raw" (original record) and "Transformed" (after transformations) using the existing RecordPreview component. If no transformations, show single panel (current behavior). Create `TemplatePreview` component for SchemaField. When a field has type 'template' and contains double-curly expressions, render a preview line below the input showing the resolved value using sample context (stream name, base URL, current date). Unresolved variables shown in warning color. Read existing `TestPreviewTab.tsx`, `RecordPreview.tsx`, and `SchemaField.tsx` first. Wire TransformPreview into TestPreviewTab. Wire TemplatePreview into SchemaField for template-type fields. Restrictions: Transformation preview is display-only (no client-side transform execution, just raw vs test-result comparison). Template preview is best-effort with available context. Success: Transformation before/after shows when applicable, template preview renders below template fields._

---

## Phase 5: Import/Export & Dashboard v2

- [x] 19. Add import/export utilities and tests
  - Files: `react/packages/urc-builder/src/utils/export-import.ts`, `react/packages/urc-builder/src/utils/export-import.unit.ts`
  - Implement YAML export/import with conflict detection
  - Purpose: Foundation for REQ-18 import/export feature
  - _Leverage: `js-yaml` (existing dependency), `manifest-serializer.ts` patterns, `types.ts` for InputSummary_
  - _Requirements: REQ-18_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: TypeScript Developer. Task: Create `export-import.ts` and `export-import.unit.ts`. Implement: `exportStreams(inputs: InputSummary[]): string` produces multi-document YAML (--- separator) with metadata (name, account, index, sourcetype, interval, tags) + manifest per stream. `parseImportFile(content: string): ImportedStream[]` parses YAML or JSON, validates required fields, returns array. `detectConflicts(imported, existing): ConflictReport` checks name collisions, returns `{ clean: [], conflicts: [] }`. Read existing `manifest-serializer.ts` for YAML patterns and `types.ts` for interfaces. Write tests: export round-trip (export then import recovers data), multi-stream export, import malformed YAML throws descriptive error, conflict detection finds name collisions, clean imports have no conflicts, JSON import works. Restrictions: Sanitize stream names (strip HTML/script tags) for XSS prevention. Do not include account credentials in exports. Success: All tests pass, round-trip works, conflicts detected correctly._

- [x] 20. Add import/export UI
  - Files: `react/packages/urc-builder/src/dashboard/ImportModal.tsx`, `react/packages/urc-builder/src/dashboard/ImportModal.unit.tsx`, `react/packages/urc-builder/src/dashboard/StreamTable.tsx`, `react/packages/urc-builder/src/dashboard/BulkActions.tsx`
  - Add Export action to row menu and bulk bar, Import button with modal
  - Purpose: REQ-18 import/export UI
  - _Leverage: `utils/export-import.ts`, existing action menu and BulkActions patterns, `Modal` and `Table` from `@splunk/react-ui`_
  - _Requirements: REQ-18_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Full-stack React Developer. Task: In StreamTable row action menu, add "Export" item that calls `exportStreams([input])` and triggers a browser download of `{name}.yaml`. In BulkActions, add "Export Selected" button that exports all selected streams. Add "Import" button to the dashboard toolbar (next to "New Stream"). Create `ImportModal` component: file picker accepting .yaml/.yml/.json, parse with `parseImportFile`, detect conflicts with `detectConflicts`, show preview table (name, account, status: new/conflict), per-conflict resolution dropdown (Skip/Rename/Overwrite), confirm button that creates/updates streams via API, summary message on completion. Test ImportModal: renders file picker, shows preview after file select, conflict resolution works, confirm creates streams. Restrictions: Use native file input for file picker. Show progress during bulk import. Success: Single and bulk export work, import modal with conflict resolution works, tests pass._

- [x] 21. Create Splunk Dashboard v2 collection metrics view
  - Files: `output/urc_app/default/data/ui/views/collection_metrics.xml`
  - Create native Splunk dashboard for operational metrics
  - Purpose: REQ-17 collection metrics dashboard
  - _Leverage: Splunk Dashboard v2 JSON schema, SPL searches against _internal and urc_stream_health KV store_
  - _Requirements: REQ-17_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: Splunk Developer specializing in Dashboard v2. Task: Create `collection_metrics.xml` that wraps a Dashboard v2 JSON definition. Include these panels: (1) Stream Health Summary - three single-value viz showing healthy/errored/disabled counts from inputlookup urc_stream_health, (2) Event Ingestion Rate - timechart of events by stream_name from index=_internal sourcetype=urc_app_collection, (3) Error Rate Over Time - timechart of ERROR level events by stream, (4) Top Errors by Stream - table with count + last error message, (5) Last Collection per Stream - table from KV store with relative timestamps. Add a global time range input wired to search-based panels. Use standard Dashboard v2 layout (absolute or grid). Restrictions: Use standard Splunk viz types only (single value, line chart, table). Keep SPL simple and efficient. Success: Dashboard renders in Splunk with all 5 panels, time picker works, data populates when collection engine runs._

---

## Phase 6: Polish & Integration Testing

- [x] 22. Integration testing and final polish
  - Files: Various test files across the codebase
  - Run full test suite, fix any integration issues, ensure all features work together
  - Purpose: Ensure all 18 requirements are met end-to-end
  - _Leverage: All previously created tests, Jest test runner_
  - _Requirements: All_
  - _Prompt: Implement the task for spec gui-polish, first run spec-workflow-guide to get the workflow guide then implement the task: Role: QA Engineer. Task: Run the full test suite with `npm test` or equivalent in `react/packages/urc-builder`. Fix any failing tests. Check for integration issues between features (e.g., search + sort + pagination pipeline, tag autocomplete + tag filter, draft save + unsaved guard). Verify the data pipeline order in StreamDashboard: inputs -> status/tag filter -> search -> sort -> group -> paginate. Add any missing edge-case tests discovered during integration. Run a coverage report and identify any modules below 80% coverage on new code. Restrictions: Do not modify feature implementations unless there is a bug. Focus on test quality and integration correctness. Success: All tests pass, no integration issues, coverage target met on new code._
