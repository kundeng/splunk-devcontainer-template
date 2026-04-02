---
name: URC engine gap analysis — all priorities complete, UI redesigned with wizard
description: Component-by-component delta after completing specs 01-07 + UI redesign
type: project
---

## Engine Gap Analysis (updated 2026-04-01)

Current engine: ~7400 runtime lines (excl. generated models), 50 components.
Test suite: 239 tests (unit + integration), 0.30s execution.
React UI: ~4500 lines across 28 components in @splunk/urc-builder.
Builder: 4-step wizard (StepBar), schema-driven forms, content.ts registry.
Dashboard: Badge/Chip/Menu components, status filter, empty-state hero.
Playwright e2e: 5 tests covering login, dashboard, wizard navigation, stream creation.

### Completed

**Per-partition state, AsyncRetriever, Rate limiting, Event Timestamp, Decoders, Transformations, Partition Router — DONE** (spec 05)

**AOP Instrumentation — DONE** (spec 06)
- inspect + functools wrapping, structured k=v logging, centralized recovery

**Builder UI — DONE** (spec 07)
- Stream Dashboard: table, tag filter, summary cards, bulk actions
- Stream Builder: 5-tab editor (Connection, Data Mapping, Splunk Output, Schedule/Tags, Test/Preview)
- Schema-driven forms: generated from declarative_component_schema.yaml
- Manifest validation: capability detection for unsupported types
- Debug logging: per-stream toggle, enriched AOP hooks at DEBUG level
- Tags: stored in inputs.conf via UCC, filterable in dashboard
- test_connection extended with mode=validate

### Remaining gaps (LOW priority)

**Interpolation (+498 lines) — LOW**
- More Jinja2 context vars, custom macros/filters, error context

**Auth (+412 lines) — LOW**
- OAuth2 PKCE, JWT key loading/RS256, session token edge cases

**Pagination (+380 lines) — LOW**
- Stop condition refinements, keyset pagination

**HttpRequester (+210 lines) — LOW**
- Request body type refinements, proxy support

**Error handling (+157 lines) — LOW**
- Response filter regex, jitter in backoff, WaitUntilTimeFromHeader

**Stream slicers (+146 lines) — LOW**
- Additional slicer patterns

### Known hard gaps (no pure Python solution)
- JWT RS256/ES256 signing — needs cryptography lib
- OAuth2 PKCE — needs cryptographic code verifier

**How to apply:** All CRITICAL, HIGH, and MEDIUM priorities are done. Builder UI is complete. Remaining items are LOW priority — implement as needed when specific manifests require them.
