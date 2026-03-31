---
name: REST connector app architecture plan
description: Decided architecture for generic REST API connector - hybrid UCC+React, vendored Airbyte CDK, CIMplicity-style build
type: project
---

## Architecture Decision: Generic REST API Connector App

Date: 2026-03-27

### Core approach: Hybrid UCC backend + Custom React frontend

Following the CIMplicity pattern (confirmed from source analysis of github.com/livehybrid/cimplicity-ai-app):

```
globalConfig.json → ucc-gen build → REST handlers + conf + libs (KEEP)
                                  → auto-generated JS/HTML   (REPLACE)
Custom React UI → @splunk/react-ui → calls UCC REST endpoints
```

### Collection engine: Vendored airbyte-cdk>=6 (full package)

**Updated 2026-03-30:** Now vendoring the full `airbyte-cdk>=6` package (not a subtree). Splunk 10.2's Python 3.13 removes the Python 3.10+ blocker.

**Previous approach (rejected then re-adopted):** Option B (CDK vendoring) was originally planned, then rejected due to Python 3.9 constraints and coupling issues, then a hand-written @component registry was attempted. CDK 6.x + Splunk 10.2 Python 3.13 made full vendoring viable again.

**Interface:** `cdk_bridge.py` bridges CDK's ManifestDeclarativeSource to Splunk events + KVStoreCheckpointer.

**Why:** 100% Airbyte manifest.yaml compatibility. Users can drop in any Airbyte declarative connector manifest and it works. MIT license, no concerns.

**Deps to include**: requests, pydantic (v2), Jinja2, PyYAML, dpath, jsonschema, isodate, backoff, cachetools, pyrate-limiter, and CDK's own transitive deps
**Deps to SKIP** (~130 MB): pandas, numpy, grpcio, nltk, google-cloud-secret-manager

### UI design

Custom React wizard (not UCC auto-generated forms) using:
- @splunk/react-ui for components
- @splunk/splunk-utils for REST calls (createRESTURL + fetch)
- @splunk/search-job for Splunk search integration
- styled-components for styling

Features:
- Connection wizard (auth type selection, endpoint config, pagination, record extraction)
- Airbyte manifest import (paste YAML or upload file)
- Test connection / preview data
- UCC Configuration page for global settings (API keys, proxy, logging)

### Backend

- UCC globalConfig.json for settings tabs (account with auth types, proxy, logging)
- Custom PersistentServerConnectionApplication handlers for:
  - Connection testing
  - Data preview
  - Manifest validation
- Modular input via BaseModInput for actual data collection
- KVStoreCheckpointer for incremental state (maps directly to Airbyte cursor state)
- dpath for record extraction (Airbyte DpathExtractor compatible)

### Build pipeline (CIMplicity pattern)

```
1. ucc-gen build --source package/ -o build/
2. cp -R build/<addon>/* packages/<app>/src/main/resources/splunk/
3. yarn run build  (webpack via lerna)
4. ucc-gen package --path <app>/ -o dist/
```

### Key patterns from CIMplicity to reuse

- `passSystemAuth=true` + `conf_manager.ConfManager` for credential access
- Separate component library from Splunk app wrapper in Lerna monorepo
- `postToEndpoint()` helper in api.js for authenticated REST calls

### Checkpointing

Airbyte cursor state → Splunk KV store:
- `stream_descriptor.name` → KV store `_key`
- `stream_state` dict → `state` dict in `ckpt.update()`
- Manual `lookback_window` (subtract from cursor before querying)
- High-water mark with `max()` (records may arrive out of order)

### Cross-platform permissions

ACLs on bind-mounted dirs (`setfacl -R -m u:1000:rwX`) applied during `app:sync-links`.
Works on Linux (essential) and OrbStack (harmless no-op). Falls back silently on Docker Desktop.
