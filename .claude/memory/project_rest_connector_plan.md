---
name: REST connector app architecture plan
description: Architecture for generic REST API connector — pure Python engine, hybrid UCC+React, pydantic v1 models
type: project
---

## Architecture Decision: Generic REST API Connector App

Date: 2026-03-27 | Updated: 2026-03-31

### Core approach: Hybrid UCC backend + Custom React frontend

Following the CIMplicity pattern:

```
globalConfig.json → ucc-gen build → REST handlers + conf + libs (KEEP)
                                  → auto-generated JS/HTML   (REPLACE with guided UI)
Custom React UI → @splunk/react-ui → calls UCC REST endpoints
```

### Collection engine: Pure Python from Airbyte schema (ADR-001)

CDK 6.x was abandoned due to non-portable native deps (glibc-locked .so files).
See `docs/adr-001-pure-python-engine.md` for full decision rationale.

**Engine:** ~7,400 lines pure Python implementing Airbyte declarative components.
**Models:** ~3,800 lines auto-generated from schema via `datamodel-code-generator==0.25.9`.
**Validation:** pydantic v1 (pure Python, `pip install 'pydantic<2' --no-binary=pydantic`).

### Checkpointing

Per-partition cursor state → Splunk KV store:
- `stream_name` → KV store `_key` prefix
- `partition_key` → nested state dict
- `cursor_value` → datetime or integer high-water mark

### Cross-platform compatibility
- Python 3.9+ (Splunk 9.x and 10.x)
- Zero .so files, zero glibc dependency
- ~2 MB package, passes AppInspect trivially
