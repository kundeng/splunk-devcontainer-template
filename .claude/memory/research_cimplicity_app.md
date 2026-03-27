---
name: CIMplicity AI app architecture analysis
description: Hybrid UCC+custom React app - build pipeline, REST patterns, credential access, monorepo structure
type: reference
---

## Architecture: Hybrid UCC Backend + Custom React Wizard

Source: https://github.com/livehybrid/cimplicity-ai-app
Splunkbase: app 7945, Apache 2.0 license

### Build Pipeline (two-stage)
```
1. ucc-gen build --source ucc-app -o build/
2. cp -R build/cim-plicity/* packages/cim-plicity/src/main/resources/splunk/
3. yarn run build  (webpack via lerna)
4. ucc-gen package --path cim-plicity/ -o ../dist/
```

### globalConfig.json
Minimal — only settings tab (LLM endpoint, API key encrypted, model, PII detectors multipleSelect) + loggingTab + dashboard. No inputs page.

### Custom REST handlers via PersistentServerConnectionApplication
- `ai_detection.py`, `pii_detection.py`, `cim_mapping.py`
- Registered in `restmap.conf` with `passPayload=true`, `passSystemAuth=true`
- Credential access pattern:
```python
cfm = conf_manager.ConfManager(
    self.system_session_key, ADDON_NAME,
    realm=f"__REST_CREDENTIAL__#{ADDON_NAME}#configs/conf-{ADDON_NAME}_settings",
)
api_key = cfm.get_conf("cim-plicity_settings").get("ai_configuration").get("api_key")
```

### React frontend
- 5-step wizard: Data Input → Field Extraction → CIM Mapping → PII Detection → Config Generation
- @splunk/react-ui@^5, @splunk/splunk-utils@^3, @splunk/search-job@^3, styled-components
- API: `createRESTURL` + `fetch()` to call both UCC and custom endpoints

### Monorepo structure
- `packages/ci-mplicity-home/` — reusable React component library (the wizard)
- `packages/cim-plicity/` — Splunk app wrapper (webpack entry point)
- `ucc-app/` — UCC source (globalConfig.json, bin/, lib/, default/)

### Key patterns to reuse
1. `passSystemAuth=true` + `conf_manager.ConfManager` for credential access without `list_storage_passwords`
2. Two-stage build: ucc-gen → copy into React → webpack → ucc-gen package
3. Separate component library from Splunk app wrapper in monorepo
4. `postToEndpoint()` helper in api.js for authenticated REST calls
