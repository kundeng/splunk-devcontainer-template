---
name: URC architecture — pure Python engine from Airbyte schema
description: Pure Python declarative engine, pydantic v1, Python 3.9+, no native deps. See ADR-001.
type: project
---

## URC Runtime Architecture (as of 2026-03-31)

### Decision: Pure Python engine (ADR-001)

CDK 6.x has non-portable native deps (pydantic_core, serpyco_rs, numpy, rpds — all require matching glibc). After evaluating 4 options (full CDK, stripped CDK, shipped Python, pure engine), chose pure Python implementation from Airbyte declarative schema.

### Stack
- **Python 3.9+** (works on Splunk 9.x and 10.x)
- **pydantic<2** (v1, pure Python, `--no-binary=pydantic`) for model validation
- **requests, Jinja2, PyYAML, dpath, isodate** — all pure Python
- **Zero .so files**, zero glibc dependency, ~2 MB package

### Runtime
- `models_generated.py` — auto-generated from Airbyte schema via `datamodel-code-generator==0.25.9`
- `engine.py` — stream orchestrator
- `components/` — auth, pagination, extraction, transforms, decoders, error handling
- `manifest.py` — YAML parse, $ref resolution, type propagation
- `interpolation.py` — Jinja2 with custom context
- `registry.py` — @component factory pattern

### Key files
- ADR: `docs/adr-001-pure-python-engine.md`
- Schema: `ucc/urc_app/schema/declarative_component_schema.yaml` (pinned)
- Models: `ucc/urc_app/package/lib/urc/models_generated.py`

**How to apply:** All manifest execution goes through `urc.engine.collect()`. No CDK dependency. Schema updates via `task urc:update-schema` (pins codegen 0.25.9).
