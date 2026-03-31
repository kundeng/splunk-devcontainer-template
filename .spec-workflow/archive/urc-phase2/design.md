# URC Phase 2 — Design

## Overview

Add missing components, network features, and UI enhancements to make URC the definitive REST API collection solution for Splunk. All work builds on the existing `urc/` runtime with the `@component` registry pattern.

## Architecture

No structural changes needed. Each new feature is either:
- A new `@component("TypeName")` class in `urc/components/`
- A new file in `urc/components/` for new categories (transformations, decoders, partition routers)
- UCC globalConfig.json changes for proxy/SSL settings
- Custom JS for UI enhancements

```
urc/components/
├── auth.py            # EXTEND: add Digest, JWT, SessionToken
├── pagination.py      # No changes needed
├── extraction.py      # No changes needed
├── requester.py       # EXTEND: all HTTP methods, proxy, SSL, cookies, timeout config
├── retriever.py       # EXTEND: pass decoder to response parsing
├── error_handler.py   # EXTEND: CompositeErrorHandler, HttpResponseFilter, WaitTimeFromHeader
├── transformations.py # NEW: AddFields, RemoveFields, KeysToLower, KeysToSnakeCase, Flatten
├── decoders.py        # NEW: JSON, JSONL, CSV, XML, GZip
├── partition_router.py # NEW: SubstreamPartitionRouter, ListPartitionRouter
└── html_extractor.py  # NEW: CSS selector extraction (web scraping)
```

## Implementation Strategy

Components are independent — they can be built in parallel. Group by dependency:

**Group A (no dependencies, pure additions):**
- R3: Transformations (5 types)
- R5: Decoders (5 types)
- R2: New auth types (3 types)

**Group B (extends existing code):**
- R1: HTTP features in requester.py (methods, proxy, SSL, cookies)
- R6: Error handling enhancements

**Group C (requires Group A retriever changes):**
- R4: Partition routers (needs retriever to support stream slicing from parent)
- R5: Wire decoders into retriever

**Group D (frontend, independent of backend):**
- R7: CodeMirror YAML editor
- R8: Test Connection REST endpoint + UI button

**Group E (future):**
- R9: Guided builder UI (Phase 2b — large effort, separate spec)
- R10: HTML/CSS selector extraction

## Component Designs

### Transformations (R3)

Applied after record extraction, before yielding to the engine:

```python
@component("AddFields")
class AddFields:
    def transform(self, record, config, **context): ...

@component("RemoveFields")
class RemoveFields:
    def transform(self, record, config, **context): ...
```

Wired in engine.py: for each record, apply stream's `transformations` list in order.

### Decoders (R5)

Pluggable response parsing — replaces hardcoded `response.json()`:

```python
@component("JsonDecoder")
class JsonDecoder:
    def decode(self, response): return response.json()

@component("CsvDecoder")
class CsvDecoder:
    def decode(self, response): # csv.DictReader on response.text
```

Wired in retriever.py: use decoder instead of `response.json()`.

### Partition Routers (R4)

Enable parent-child stream collection:

```python
@component("SubstreamPartitionRouter")
class SubstreamPartitionRouter:
    def get_partitions(self, config, parent_records):
        for record in parent_records:
            yield {"parent_id": record[self.parent_key]}
```

Wired in engine.py: before collecting a stream, check for `partition_router`, collect parent stream first, iterate partitions.

### Requester Enhancements (R1)

Add to HttpRequester.__init__:
- `self._method` supports all HTTP methods
- Proxy from UCC settings tab (read via ConfManager)
- SSL verify toggle + CA bundle
- Cookie jar persistence
- Configurable timeout from manifest

### Test Connection REST Endpoint (R8)

Custom PersistentServerConnectionApplication:

```python
class TestConnectionHandler(PersistentServerConnectionApplication):
    def handlePOST(self, confInfo):
        # Read manifest + account from POST body
        # Run collect() with limit=1 page
        # Return sample records or error
```

Registered in restmap.conf, called from UCC custom control button.

## Testing Strategy

- Unit tests for each new component (mock HTTP responses)
- E2E test: extend test-urc-collect.sh with pagination, auth type, and transformation tests
- Use httpbin.org or similar for testing auth methods and response types
