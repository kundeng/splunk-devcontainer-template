# URC Scaffold — Design

## Overview

Practical spike to scaffold the URC add-on, vendor dependencies, and verify the build pipeline works end-to-end on Splunk 10.2. No connector logic — just the foundation.

## Code Reuse Analysis

### Existing Components to Leverage

- **`task ucc:init`** — scaffolds globalConfig.json + package/ structure
- **`task ucc:build`** — runs `ucc-gen build`, installs requirements.txt into lib/
- **`task ucc:link`** / **`task ucc:dev`** — symlinks build output into running Splunk
- **`task deps:install`** — downloads Splunkbase dependencies

### Integration Points

- **ucc-gen build** handles `pip install --target=output/<ta>/lib/ -r requirements.txt`
- **import_declare_test.py** (auto-generated) adds `lib/` to `sys.path` at runtime
- **Splunk 10.2** runs Python 3.9 (default) — all vendored deps must be compatible

## Architecture

This spike is linear — no runtime architecture to design. The deliverable is a verified build pipeline:

```
1. ucc-gen init → ucc/urc_app/ (globalConfig.json, package/)
2. Download schema → ucc/urc_app/schema/declarative_component_schema.yaml
3. Define requirements.txt → airbyte-cdk pure-Python deps
4. ucc-gen build → output/urc_app/ (with vendored libs in lib/)
5. Load in Splunk 10.2 → verify imports work
6. Audit report → what works, what doesn't, what to patch
```

## Dependency Strategy

### Step 1: Full CDK install audit

Install `airbyte-cdk` via `pip install --target` into a temporary directory and audit:
- Total size
- Which packages have C extensions (`.so`/`.pyd` files)
- Which packages are pure Python

### Step 2: Identify minimal pure-Python subset

For the declarative runtime path, we need (based on research):

**Likely pure Python:**
- `pydantic` v1.10.x (166 KB — the v1 compat layer the CDK models actually use)
- `typing-extensions`
- `requests` + `urllib3` + `certifi` + `charset-normalizer` + `idna`
- `PyYAML`
- `Jinja2` + `MarkupSafe` (MarkupSafe has optional C speedup but works without)
- `dpath`
- `backoff`
- `cachetools`
- `isodate`
- `python-dateutil` + `pytz`
- `pyrate-limiter`
- `jsonschema` + `referencing` + `attrs`
- `jsonref`

**Known C/Rust extensions (need to verify if avoidable):**
- `orjson` — used in JsonDecoder, has `json` stdlib fallback
- `pydantic-core` — NOT needed (models use v1 compat)
- `serpyco-rs` — NOT needed (Airbyte protocol serialization)
- `cryptography` — only for JWT auth
- `rapidfuzz` — fuzzy matching, unlikely in declarative path

### Step 3: Build requirements.txt + exclude.txt

Create `package/lib/requirements.txt` with the minimal set and `package/lib/exclude.txt` to strip unwanted transitive deps that `ucc-gen build` might pull in.

### Step 4: Verify in Splunk

After `ucc-gen build`, symlink into Splunk 10.2 and test:
```python
# From Splunk's Python (via search command or REST endpoint):
import import_declare_test
from pydantic.v1 import BaseModel  # or from pydantic import BaseModel
import dpath
import yaml
import jinja2
```

## File Structure (After Spike)

```
ucc/urc_app/
├── globalConfig.json              # Minimal UCC config (scaffold)
├── package/
│   ├── app.manifest
│   ├── bin/
│   │   └── urc_input.py           # Stub modular input (just imports)
│   ├── lib/
│   │   ├── requirements.txt       # Verified pure-Python deps
│   │   └── exclude.txt            # Packages to strip from build
│   └── static/
│       ├── appIcon.png
│       └── appIcon_2x.png
├── schema/
│   └── declarative_component_schema.yaml  # Downloaded from Airbyte CDK
└── AUDIT.md                        # Dependency audit results
```

## Testing Strategy

### Build Verification
- `task ucc:build APP_NAME=urc_app` completes without errors
- Output directory contains expected structure

### Import Verification
- Load app in Splunk 10.2
- Verify `import_declare_test` works
- Test key imports: pydantic, dpath, yaml, jinja2, requests
- Document any failures with exact error messages

### Size Verification
- Total `lib/` size after build
- Compare against target (under 5 MB for pure-Python subset)
