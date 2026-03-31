---
name: URC architecture decisions
description: Key technical decisions for URC runtime - vendored airbyte-cdk>=6, cdk_bridge.py interface, Python 3.10+ (Splunk 10.2)
type: project
---

## URC Runtime Architecture Decisions

### Updated 2026-03-30: Vendor airbyte-cdk>=6

Previous approach (hand-written @component registry, custom engine.py, Pydantic v1) was abandoned. CDK 6.x is now vendored directly.

### Why CDK 6.x works now
- CDK 6.x requires Python 3.10+ — Splunk 10.2 ships Python 3.13, confirmed working
- Full CDK vendoring eliminates the partial-vendoring coupling problems encountered earlier
- 100% Airbyte manifest.yaml compatibility out of the box, no reimplementation needed

### Architecture
- **Vendored `airbyte-cdk>=6`** in `package/lib/` with all required deps
- **`cdk_bridge.py`** is the interface between CDK and Splunk (replaces custom engine.py, registry.py)
- No @component registry, no custom Pydantic models, no datamodel-codegen — CDK handles all of this internally
- Manifest validation, component instantiation, stream orchestration all delegated to CDK

### What cdk_bridge.py does
- Loads Airbyte declarative manifest YAML
- Instantiates CDK's `ManifestDeclarativeSource`
- Bridges CDK's record output to Splunk event format
- Maps Airbyte state/cursor to KVStoreCheckpointer
- Handles credential injection from Splunk conf via CDK's config object

**How to apply:** When adding new connector configs, write a standard Airbyte manifest.yaml. The CDK handles component resolution. Only modify cdk_bridge.py for Splunk-specific integration concerns.
