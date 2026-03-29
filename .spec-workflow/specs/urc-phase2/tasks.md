# URC Phase 2 — Tasks

## Group A: New Components (independent, parallelizable)

- [x] 1. Add transformations module (R3)
  - Create `urc/components/transformations.py`
  - Implement: AddFields, RemoveFields, KeysToLower, KeysToSnakeCase, FlattenFields
  - Wire into engine.py: apply stream's `transformations` list after record extraction
  - _Requirements: R3_

- [x] 2. Add decoders module (R5)
  - Create `urc/components/decoders.py`
  - Implement: JsonDecoder, JsonlDecoder, CsvDecoder, XmlDecoder, GzipDecoder
  - Wire into retriever.py: use decoder instead of hardcoded response.json()
  - _Requirements: R5_

- [x] 3. Add new auth types (R2)
  - Add to `urc/components/auth.py`: DigestAuth, JwtAuth, SessionTokenAuth
  - DigestAuth: use requests.auth.HTTPDigestAuth
  - JwtAuth: sign JWT with configurable algorithm/headers/payload (use PyJWT if available, fallback to manual)
  - SessionTokenAuth: login request, extract token, inject into subsequent requests
  - _Requirements: R2_

## Group B: Enhance Existing Components

- [x] 4. Enhance requester with full HTTP support (R1)
  - All HTTP methods: GET, POST, PUT, PATCH, DELETE, HEAD
  - Proxy support: read from UCC proxy settings tab via ConfManager
  - SSL: verify toggle + custom CA bundle path
  - Cookie persistence: use requests.Session cookie jar
  - Configurable timeout from manifest or input config
  - _Requirements: R1_

- [x] 5. Enhance error handling (R6)
  - Add CompositeErrorHandler: chain multiple handlers
  - Add HttpResponseFilter: match by status code, error message, predicate
  - Add WaitTimeFromHeader: extract retry timing from Retry-After or custom headers
  - Add WaitUntilTimeFromHeader: wait until a specific time
  - Response actions: RETRY, FAIL, IGNORE, SUCCESS
  - _Requirements: R6_

## Group C: Stream Orchestration

- [-] 6. Add partition routers (R4)
  - Create `urc/components/partition_router.py`
  - SubstreamPartitionRouter: collect parent stream, iterate partitions for child
  - ListPartitionRouter: iterate over static or config-driven list
  - Wire into engine.py: before collecting a stream, resolve partitions
  - _Requirements: R4_

- [ ] 7. Wire decoders and transformations into retriever and engine
  - retriever.py: accept decoder component, use it instead of response.json()
  - engine.py: apply transformations after extraction, before yielding records
  - Test with CSV and XML responses
  - _Requirements: R3, R5_

## Group D: UI Enhancements

- [-] 8. Add Test Connection REST endpoint (R8)
  - Create custom REST handler: PersistentServerConnectionApplication
  - Endpoint accepts manifest + account config, runs dry-run collection (1 page)
  - Returns sample records or error with details
  - Register in restmap.conf
  - _Requirements: R8_

- [ ] 9. Add CodeMirror YAML editor custom control (R7)
  - Bundle CodeMirror 6 with YAML mode into a single JS file
  - Create UCC custom control at appserver/static/js/build/custom/yaml_editor.js
  - Update globalConfig.json: change manifest field type from textarea to custom
  - _Requirements: R7_

## Group E: Future (not in this sprint)

- [ ] 10. Guided builder UI (R9) — separate spec needed
- [ ] 11. HTML/CSS selector extraction (R10) — separate spec needed
