---
name: UCC backend with custom React UI
description: UCC provides high-value backend (REST handlers, credential encryption, modular inputs) fully decoupled from its auto-generated UI
type: reference
---

## Key Finding: UCC backend is fully independent of its frontend

### What UCC backend provides (usable with custom React UI):
- **REST handlers** (rh_*.py) — full CRUD on .conf files via `/servicesNS/-/-/{restRoot}_{entity}`
- **Credential encryption** — `encrypted=True` fields auto-stored in passwords.conf
- **Modular input framework** — BaseModInput with checkpointing, credential retrieval
- **.conf file generation** — restmap.conf, web.conf, inputs.conf, server.conf, spec files
- **OpenAPI spec** — documents REST endpoints for custom frontend
- **Python libs** — solnlib, splunktaucclib vendored to lib/

### What you replace:
- `appserver/static/js/build/` (auto-generated React forms)
- `appserver/templates/` and XML views
- Build custom UI with `@splunk/react-ui` calling the same REST endpoints

### Architecture:
```
globalConfig.json → ucc-gen build → REST handlers + conf + libs (KEEP)
                                  → auto-generated JS/HTML (REPLACE)
Custom React UI → @splunk/react-ui → calls UCC REST endpoints
```

### CIMplicity AI reference:
- Proprietary (Splunkbase only, no public repo)
- Uses custom React UI with @splunk/react-ui, NOT UCC forms
- Multi-step wizard, AI-driven — can't be expressed in UCC declarative JSON

### kveditor reference (in .ref/):
- Custom React with @splunk/react-ui + @splunk/splunk-utils
- Calls Splunk REST directly via createRESTURL + fetch()
- No UCC, no globalConfig.json
