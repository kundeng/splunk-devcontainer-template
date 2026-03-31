---
name: URC engine gap analysis — 5300 lines to 99.9% CDK parity
description: Component-by-component delta between current pure engine (2148 lines) and full CDK parity, with line estimates
type: project
---

## Engine Gap Analysis (2026-03-31)

Current engine: 2148 lines, 40 runtime classes. Need ~5300 more for 99.9% CDK parity.

### Priority breakdown

**Incremental/cursors (+1316 lines) — CRITICAL**
- Per-partition state management (nested dict in KV Store)
- Lookback windows (subtract from cursor before querying)
- Resumable full refresh
- MinMaxDatetime clamping

**AsyncRetriever (+796 lines) — HIGH**
- Bulk job lifecycle: create → poll → download
- Status mapping, URL extraction
- Splunk REST API and many enterprise APIs use this pattern

**Rate limiting (+575 lines) — HIGH**
- Token bucket, fixed/moving window policies
- API budget with per-endpoint limits
- stdlib time.monotonic + deque, no native deps

**Interpolation (+498 lines) — MEDIUM**
- More Jinja2 context vars: response, headers, next_page_token, stream_interval
- Custom macros and filters
- Error context for better debugging

**Transformations (+468 lines) — MEDIUM**
- DpathFlattenFields, KeysReplace, SchemaNormalization
- Config transforms (ConfigAddFields, ConfigRemoveFields, ConfigRemapField)
- ComponentMappingDefinition, GroupByKeyMergeStrategy

**Extraction (+434 lines) — MEDIUM**
- CSV extraction via stdlib csv (not pandas)
- PropertiesFromEndpoint (dynamic property fetching)
- JsonSchemaPropertySelector

**Auth (+412 lines) — MEDIUM**
- OAuth2: token refresh edge cases, PKCE (needs crypto — real gap)
- JWT: key loading, multiple algorithms, payload customization
- Session token refinements

**Pagination (+380 lines) — LOW**
- Stop condition evaluation refinements
- Interpolated cursor edge cases
- Keyset pagination

**Decoders (+297 lines) — LOW**
- ZipfileDecoder (stdlib zipfile)
- CSV refinement, JSONL edge cases

**HttpRequester (+210 lines) — LOW**
- Request body types: PlainText, UrlEncodedForm, JsonObject, GraphQL
- Proxy support refinement

**Error handling (+157 lines) — LOW**
- Response filter regex matching
- Jitter in backoff
- WaitUntilTimeFromHeader

**Stream slicers (+146 lines) — LOW**
- Cartesian product slicers

### Known hard gaps (no pure Python solution)
- JWT RS256/ES256 signing — needs cryptography or pure-Python RSA lib
- OAuth2 PKCE — needs cryptographic code verifier

**How to apply:** Start with CRITICAL + HIGH priorities. Each component is independent and testable. Use the test manifests (jsonplaceholder, github, servicenow, infoblox, azure) to validate.
