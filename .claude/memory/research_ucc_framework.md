---
name: UCC framework research findings
description: How UCC generator works, project structure conventions, globalConfig.json schema, and gaps vs current template
type: reference
---

## UCC Generator (`splunk/addonfactory-ucc-generator`)

Generates complete Splunk add-ons from a single `globalConfig.json`:
- React UI (Vite-built), REST handlers, modular input scripts, all .conf files, OpenAPI docs, monitoring dashboards

### Source structure (developer writes):
```
my_addon/
  globalConfig.json
  package/
    bin/my_input.py          # Custom collection logic (or inputHelperModule)
    lib/                     # Vendored Python (solnlib, splunktaucclib)
    default/                 # Manual .conf overrides
    appserver/static/js/build/custom/  # UI hooks
    app.manifest
```

### Build: `ucc-gen build --source package/ --ta-version x.y.z`

### globalConfig.json key sections:
- `meta`: name, restRoot, version, displayName, schemaVersion, defaultView
- `pages.configuration.tabs[]`: account (with table=referenceable), loggingTab, proxyTab
- `pages.inputs.services[]`: each service = a modular input type with entity[] fields
- Entity types: text, textarea, checkbox, singleSelect, multipleSelect, radio, oauth, interval, index, file, date, etc.
- `encrypted: true` → auto-stored in passwords.conf
- `options.referenceName` → links input field to a config tab (e.g., account)

### Runtime libraries:
- `splunktaucclib`: BaseModInput, AdminExternalHandler, GlobalConfig, RestCredentials
- `solnlib`: CredentialManager, KVStoreCheckpointer, modular_input base classes

### Template gaps for UCC alignment:
1. No globalConfig.json support or ucc-gen in build pipeline
2. No auto-generated REST handlers or input management UI
3. No app.manifest or import_declare_test.py pattern
4. No Python lib vendoring convention

### Template strengths to preserve:
- Docker dev environment (UCC provides none)
- Multi-app dev with symlinks, dev+staging side-by-side
- Taskfile automation, debugging, Splunkbase deps
