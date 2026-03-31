---
name: URC engine gap analysis — ~3000 lines remaining for 99.9% CDK parity
description: Component-by-component delta after completing CRITICAL+HIGH+MEDIUM priorities
type: project
---

## Engine Gap Analysis (updated 2026-03-31)

Current engine: 3510 runtime lines (excl. generated models), 49 components.
Test suite: 176 tests, 3151 lines, 0.19s execution.

### Completed

**Per-partition state — DONE**
- PerPartitionStateManager, lookback windows, time window slicing

**AsyncRetriever — DONE**
- Create → poll → download lifecycle, status mapping, URL extraction

**Rate limiting — DONE**
- TokenBucket, MovingWindowRateLimiter, APIBudget

**Event Timestamp — DONE (URC extension)**
- CursorBasedTimestamp, FieldBasedTimestamp, FetchTimestamp
- Separate extension schema (urc_extensions_schema.yaml)
- Auto-derives _time from DatetimeBasedCursor

**Decoders — DONE**
- CsvDecoder (BOM, malformed rows), JsonlDecoder, ZipfileDecoder

**Transformations — DONE**
- DpathFlattenFields, KeysReplace, SchemaNormalization

**Partition Router — DONE**
- CartesianProductStreamSlicer (itertools.product, 100k limit)

**Test Suite — DONE**
- 157 unit tests + 19 integration tests = 176 total
- Covers: auth, pagination, decoders, transforms, error handling, rate limiting,
  event timestamp, manifest parsing, registry, full pipeline, incremental sync,
  per-partition state, async retriever, modular input with mocked Splunk SDK

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

**How to apply:** All CRITICAL, HIGH, and MEDIUM priorities are done. Remaining items are LOW priority — implement as needed when specific manifests require them.
