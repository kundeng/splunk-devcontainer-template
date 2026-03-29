---
name: URC architecture decisions
description: Key technical decisions for URC runtime - no CDK vendoring, Pydantic v1 validation + @component registry, no DI framework
type: project
---

## URC Runtime Architecture Decisions (2026-03-29)

### No vendored Airbyte CDK runtime
The CDK is tightly coupled (59% of files needed), uses Python 3.10+ syntax (match/case, X|None unions), and has circular import chains that resist partial vendoring. Abandoned after extensive patching effort.

### Clean runtime with Pydantic v1 + @component registry
- **Pydantic v1** (pure Python, works on Splunk 3.9) validates manifests against the Airbyte schema
- **@component("TypeName") decorator** maps manifest type names to runtime classes
- **Recursive construction** walks the validated model tree, looks up each type in the registry, instantiates with children as arguments
- This is ~50 lines of DI glue, not a framework

### Why no DI framework
Evaluated: antidote (archived), dishka (3.10+), injector, dependency-injector (Cython), wireup, lagom, punq, svcs (3.10+). None solve our specific problem: dispatch-by-type-field from a declarative YAML config. The pattern is domain-specific (Airbyte declarative config), not general DI.

### Schema → Models → Components pipeline
1. `manifest.py`: YAML parse → resolve $ref → propagate types/$parameters
2. Pydantic models (generated from Airbyte schema via datamodel-codegen): validate the processed dict
3. `registry.py`: @component decorator + create() maps validated models to runtime classes
4. `engine.py`: orchestrate streams, cursors, yield records

**Why:** Pydantic catches config errors (missing fields, wrong types) with clear field-path messages BEFORE any HTTP requests. The registry pattern keeps component implementations in separate files, self-registered, no god-object factory.

**How to apply:** When adding new component types, create a class with @component("TypeName") decorator. When updating the schema, re-run datamodel-codegen to regenerate models.
