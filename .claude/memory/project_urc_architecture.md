---
name: URC architecture — pip-installed CDK 6.x on Splunk 10.2 / Python 3.13
description: Current URC runtime architecture — airbyte-cdk pip-installed via requirements.txt, cdk_bridge.py interface, Python 3.13 opted-in on both devcontainer and Splunk container
type: project
---

## URC Runtime Architecture (as of 2026-03-31)

### Runtime stack
- **Splunk Enterprise 10.2** with Python 3.13 opted-in
- **Devcontainer** also runs Python 3.13 (matched to Splunk runtime)
- **airbyte-cdk>=6,<7** declared in `requirements.txt`, installed by `ucc-gen build` into `package/lib/`

### Active code (3 files)
1. `package/bin/urc_app_input_helper.py` — Splunk modular input handler
2. `package/lib/urc/cdk_bridge.py` — wraps `ConcurrentDeclarativeSource`, yields `(stream_name, record, state)`
3. `package/bin/urc_test_connection.py` — REST endpoint for connection testing

### Build pipeline
- `task ucc:build` → `ucc-gen build` → installs deps from `requirements.txt` → strips unused transitive deps (~130 MB)
- `task ucc:appinspect` → validates package for Splunkbase certification
- `task ucc:package` → creates .tar.gz for deployment

### What was removed (2026-03-31 redirection)
Custom engine code deleted: engine.py, components/ (10 files), interpolation.py, registry.py, manifest.py, validate.py, models.py, models_generated.py, extensions.py. CDK handles all of this natively.

**Why:** CDK 6.x on Python 3.13 provides 100% Airbyte manifest compatibility with no custom code needed. The hand-written engine was a workaround for Python 3.9 limitations that no longer apply.

**How to apply:** All manifest execution goes through `cdk_bridge.py`. Tests use `test_manifests.py` which imports from `urc.cdk_bridge`.
