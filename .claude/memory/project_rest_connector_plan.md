---
name: REST connector app architecture plan
description: Architecture for generic REST API connector — hybrid UCC+React, pip-installed Airbyte CDK 6.x, CIMplicity-style build
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

### Collection engine: pip-installed airbyte-cdk>=6

**Updated 2026-03-31:** airbyte-cdk>=6,<7 is declared in `requirements.txt` and installed by `ucc-gen build` into `package/lib/`. No vendoring, no custom engine.

**History:** Hand-written @component engine was attempted for Python 3.9 compat, then CDK 6.x + Splunk 10.2 Python 3.13 made it unnecessary. All custom engine code removed.

**Interface:** `cdk_bridge.py` bridges CDK's `ConcurrentDeclarativeSource` to Splunk events + KVStoreCheckpointer.

**Post-build cleanup:** `ucc:build` task strips ~130 MB of unused transitive deps (pandas, numpy, grpcio, nltk, google-cloud, etc.) after install.

### Checkpointing

Airbyte cursor state → Splunk KV store:
- `stream_descriptor.name` → KV store `_key`
- `stream_state` dict → `state` dict in `ckpt.update()`

### Cross-platform permissions

ACLs on bind-mounted dirs (`setfacl -R -m u:1000:rwX`) applied during dev lifecycle.
