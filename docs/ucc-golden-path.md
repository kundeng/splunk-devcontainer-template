# UCC Golden Path — Building a Splunk Add-on from Scratch

A practical handbook for building Splunk add-ons using the UCC (Universal Configuration Console) framework, with a guided React UI and modular input backend. Distilled from building the URC (Universal REST Client) add-on.

**Audience:** Developers building Splunk add-ons who want to avoid the pitfalls we hit.
**Last updated:** 2026-03-31

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [UCC Basics — globalConfig.json](#3-ucc-basics--globalconfigjson)
4. [Modular Input Handler](#4-modular-input-handler)
5. [Custom REST Endpoints](#5-custom-rest-endpoints)
6. [Python Dependencies — The Vendoring Challenge](#6-python-dependencies--the-vendoring-challenge)
7. [Adding a Custom React UI](#7-adding-a-custom-react-ui)
8. [Build Pipeline](#8-build-pipeline)
9. [Testing Strategy](#9-testing-strategy)
10. [Lessons Learned — What Failed and Why](#10-lessons-learned--what-failed-and-why)

---

## 1. Architecture Overview

A UCC add-on has three layers:

```
┌─────────────────────────────────────┐
│  Splunk Web UI                      │
│  ├─ UCC auto-generated pages        │  ← Configuration, Inputs, Dashboard
│  └─ Custom React page (optional)    │  ← Guided builder, wizard, etc.
├─────────────────────────────────────┤
│  REST Handlers                      │
│  ├─ UCC-generated (accounts, inputs)│  ← Auto from globalConfig.json
│  └─ Custom (PersistentServerConn)   │  ← Your endpoints (test_connection, etc.)
├─────────────────────────────────────┤
│  Modular Input                      │
│  └─ Airbyte CDK 6.x runtime        │  ← Declarative manifest → data collection
│     └─ Vendored libraries           │  ← Binary wheels OK on Splunk 10.2+/3.13
└─────────────────────────────────────┘
```

**Key insight:** UCC generates the glue (conf files, REST handlers, UI pages) from a single `globalConfig.json`. The modular input handler delegates to the Airbyte CDK 6.x declarative runtime, which interprets YAML manifests to collect data from REST APIs. The UI comes free — unless you want something beyond forms, in which case you build a custom React page alongside UCC.

## 2. Project Structure

```
ucc/
├── your_app/                    # UCC source directory
│   ├── globalConfig.json        # THE source of truth for UCC
│   ├── package/
│   │   ├── app.manifest         # Splunk app metadata
│   │   ├── bin/
│   │   │   ├── your_app_input_helper.py    # Modular input handler
│   │   │   └── your_custom_endpoint.py     # Custom REST handlers
│   │   ├── lib/
│   │   │   ├── requirements.txt            # Python deps to vendor
│   │   │   ├── exclude.txt                 # Packages to skip
│   │   │   └── your_module/                # Your Python package
│   │   ├── default/
│   │   │   ├── collections.conf            # KV Store collections
│   │   │   ├── transforms.conf             # Lookup definitions
│   │   │   └── restmap.conf                # Custom REST endpoints
│   │   └── static/
│   │       └── appIcon*.png
│   └── schema/                  # Dev-time reference files
│
├── output/                      # UCC build output (git-ignored)
│   └── your_app/                # Complete Splunk app
│
react/                           # Optional: custom React UI
├── packages/
│   ├── your-component/          # Reusable React component library
│   └── your-splunk-app/         # Splunk app wrapper (webpack entry)
```

**What UCC generates in `output/`:**
- `default/app.conf`, `inputs.conf.spec`, `restmap.conf`, `web.conf`
- `bin/import_declare_test.py` — the critical path-setup module
- `bin/your_app_rh_*.py` — REST handlers for each entity
- `appserver/static/js/build/` — bundled React UI (entry_page.js)
- `appserver/templates/base.html` — Mako template
- `README/` — generated .conf.spec files
- Vendored Python libs in `lib/`

## 3. UCC Basics — globalConfig.json

This is the single file that drives everything. It defines:
- **Accounts** (credentials, auth types)
- **Inputs** (data collection configuration)
- **Settings** (logging, proxy)
- **Dashboard** (optional monitoring panels)

### Account Entity Example

```json
{
  "name": "account",
  "title": "Account",
  "table": {
    "header": [
      { "field": "name", "label": "Name" },
      { "field": "auth_type", "label": "Auth Type" }
    ],
    "actions": ["edit", "delete", "clone"]
  },
  "entity": [
    {
      "type": "text",
      "label": "Name",
      "field": "name",
      "required": true,
      "validators": [
        { "type": "regex", "pattern": "^[a-zA-Z]\\w*$", "errorMsg": "Start with a letter, alphanumeric only" }
      ]
    },
    {
      "type": "singleSelect",
      "label": "Auth Type",
      "field": "auth_type",
      "required": true,
      "defaultValue": "bearer",
      "options": {
        "disableSearch": true,
        "autoCompleteFields": [
          { "label": "API Key", "value": "api_key" },
          { "label": "Bearer Token", "value": "bearer" },
          { "label": "Basic Auth", "value": "basic" },
          { "label": "OAuth 2.0", "value": "oauth2" },
          { "label": "None", "value": "none" }
        ]
      }
    },
    {
      "type": "text",
      "label": "API Key",
      "field": "api_key",
      "encrypted": true,
      "required": false,
      "options": { "display": false },
      "modifyFieldsOnValue": [
        {
          "fieldValue": "api_key",
          "mode": "edit",
          "fieldsToModify": [
            { "fieldId": "api_key", "display": true, "required": true }
          ]
        }
      ]
    }
  ]
}
```

**Key patterns:**
- `"encrypted": true` — UCC handles encryption/decryption via Splunk's credential store
- `"modifyFieldsOnValue"` — conditional field visibility based on another field's value
- Validators: `regex`, `string` (minLength/maxLength), `number` (range), `url`

### Input Entity Example

```json
{
  "name": "your_app_input",
  "title": "Data Input",
  "entity": [
    { "type": "text", "label": "Name", "field": "name", "required": true },
    { "type": "text", "label": "Account", "field": "account", "required": true },
    { "type": "text", "label": "Base URL", "field": "base_url", "required": true },
    { "type": "textarea", "label": "Manifest", "field": "manifest", "required": true },
    { "type": "interval", "label": "Interval", "field": "interval", "required": true, "defaultValue": "300" },
    { "type": "index", "label": "Index", "field": "index", "required": true, "defaultValue": "default" }
  ]
}
```

## 4. Modular Input Handler

UCC generates a helper base class. Your handler implements `stream_events()`:

```python
import import_declare_test  # MUST be first import — sets up sys.path

from splunktaucclib.modinput_wrapper.base_modinput import BaseModInput

class YourInput(BaseModInput):
    def __init__(self):
        super().__init__("your_app", "your_app_input")

    def stream_events(self, inputs, ew):
        for input_name, input_item in inputs.inputs.items():
            # input_item is a dict with all entity fields from globalConfig
            account_name = input_item.get("account")
            base_url = input_item.get("base_url")
            manifest = input_item.get("manifest")
            index = input_item.get("index", "default")
            sourcetype = input_item.get("sourcetype", "your_app")

            # Access encrypted credentials via ConfManager
            from solnlib import conf_manager
            cfm = conf_manager.ConfManager(
                self.session_key,
                "your_app",
                realm=f"__REST_CREDENTIAL__#your_app#configs/conf-your_app_account",
            )
            account = cfm.get_conf("your_app_account").get(account_name)
            api_key = account.get("api_key")

            # Your collection logic here
            records = your_collect_function(base_url, api_key, manifest)

            for record in records:
                event = ew.Event()
                event.stanza = input_name
                event.index = index
                event.sourcetype = sourcetype
                event.data = json.dumps(record)
                ew.write_event(event)
```

**Critical: `import_declare_test` must be the first import.** It adds `lib/` to `sys.path` so vendored packages resolve. If you import anything before it, vendored libraries won't be found.

### KV Store Checkpointing

For incremental collection, use KV Store instead of file-based checkpoints:

```ini
# collections.conf
[your_checkpoints]
field.state = string
```

```python
from solnlib import KVStoreCheckpointer

ckpt = KVStoreCheckpointer(
    "your_checkpoints", self.session_key, "your_app"
)
state = ckpt.get(input_name) or {}
# ... collect data using state ...
ckpt.update(input_name, new_state)
```

## 5. Custom REST Endpoints

For operations beyond CRUD (test connection, custom validation, etc.):

```python
import json
import splunk.admin as admin
from splunk.persistconn.application import PersistentServerConnectionApplication


class YourEndpoint(PersistentServerConnectionApplication):
    def __init__(self, command_line, command_arg):
        super().__init__()

    def handleList(self, in_string):
        """Handle GET requests."""
        return {"payload": json.dumps({"status": "ok"}),
                "status": 200}

    def handleCreate(self, in_string):
        """Handle POST requests."""
        payload = json.loads(in_string)
        body = dict(payload.get("form", []))
        # ... your logic ...
        return {"payload": json.dumps(result),
                "status": 200}
```

Register in `restmap.conf`:

```ini
[script:your_endpoint]
match = /your_app/your_endpoint
script = your_endpoint.py
scripttype = persist
handler = your_endpoint.YourEndpoint
requireAuthentication = true
output_mode = json
passPayload = true
passSystemAuth = true
```

**`passSystemAuth=true`** gives you `self.system_session_key` which can access encrypted credentials without the user needing `list_storage_passwords` capability.

## 6. Python Dependencies — The Vendoring Challenge

This is where most UCC projects hit trouble. Splunk ships its own Python runtime, and the constraints depend on which version you target.

### The Rules

1. **Binary wheels work** — `.so` / `.pyd` C extensions load fine on Splunk's Python. Ship `manylinux2014_x86_64` wheels for Splunk Cloud.
2. **Python 3.13** — Target Splunk 10.2+ with `python.version = python3` in `app.conf`. No syntax restrictions, full binary wheel support.
3. **No pip at runtime** — Everything must be pre-vendored into `lib/`.
4. **Size budget** — Keep under ~65 MB total. The CDK pulls ~130 MB of transitive deps; post-build cleanup is essential (see below).

### How UCC Vendors Dependencies

```
lib/
├── requirements.txt    # What to install (e.g. airbyte-cdk>=6,<7)
├── exclude.txt         # What to skip (setuptools, pip, etc.)
└── your_module/        # Your Python code (e.g. urc/)
```

`ucc-gen build` runs `pip install -r requirements.txt -t lib/` during build, then removes packages listed in `exclude.txt`. For large dependencies like `airbyte-cdk`, this is the correct approach — pip resolves the full dependency tree and installs binary wheels automatically.

### Post-Build Cleanup (Critical for CDK)

The CDK pulls in ~130 MB of transitive dependencies, many of which are unused at runtime (pandas, numpy, grpc, nltk, dateparser_cli, manifest_server, test dirs). A post-build cleanup step in the Taskfile strips these down:

```yaml
# Taskfile.yml — runs after ucc-gen build
ucc:clean-deps:
  cmds:
    - rm -rf output/your_app/lib/{pandas,numpy,grpc*,nltk*,...}
    - find output/your_app/lib -type d -name __pycache__ -exec rm -rf {} +
    - find output/your_app/lib -type d -name tests -exec rm -rf {} +
```

Result: **62 MB total / 41 MB lib** — passes AppInspect (0 failures, 0 errors, 113 passes).

### Common Pitfalls

| Problem | Symptom | Fix |
|---------|---------|-----|
| CDK bloat (pandas, numpy, grpc) | Package too large, AppInspect warnings | Post-build cleanup in Taskfile |
| Conflicting deps | `ImportError: cannot import name X` | Use `exclude.txt` to remove conflicting versions |
| Wrong platform wheel | `ImportError: cannot open shared object file` | Build with `--platform manylinux2014_x86_64` |
| `.pyc` files in package | AppInspect failure | Clean `__pycache__` during build |

### Our Dependency Stack (Splunk 10.2+ / Python 3.13)

```
airbyte-cdk>=6,<7       # Declarative manifest runtime — pip-installed via requirements.txt
pydantic>=2,<3          # v2 — fast Rust-backed validation via pydantic-core
requests + urllib3      # HTTP client
Jinja2 + MarkupSafe    # Template engine
PyYAML                  # YAML parsing
dpath                   # 15 KB, zero deps, MIT
backoff                 # Retry/backoff
python-dateutil + pytz  # Datetime handling
isodate                 # ISO 8601 parsing
jsonref                 # JSON reference resolution
jsonschema              # No longer need to pin — binary deps are fine
xmltodict               # XML → dict
```

## 7. Adding a Custom React UI

When UCC's auto-generated forms aren't enough, add a custom React page alongside UCC. This is the **CIMplicity pattern** — proven in production on Splunkbase.

### Architecture Choice Matrix

| Need | Use |
|------|-----|
| Simple config forms | UCC auto-generated (globalConfig.json) |
| Custom field widget | UCC Custom Control (`"type": "custom"`) |
| Extra config tab | UCC Custom Tab |
| Wizard, dashboard, complex UI | Separate Splunk page (CIMplicity pattern) |

### Adding React to a UCC App — One Command

```bash
task ucc:add-react APP_NAME=your_app
```

This single command does everything:

1. **Scaffolds a component library** (`your-builder/`) via `@splunk/create` — where your actual UI lives
2. **Scaffolds a Splunk app wrapper** (`your-app/`) via `@splunk/create` — the entry point that mounts into Splunk
3. **Adapts the wrapper for UCC overlay** — the 6 mechanical transformations (see below)
4. **Installs dependencies and builds** — produces `stage/` ready for merge

All names are derived from `APP_NAME`: `your_app` → `your-app` (React wrapper), `your-builder` (component library), `builder` (default page name).

### What the UCC Adaptation Does

`@splunk/create` scaffolds a standalone React Splunk app. For UCC overlay, 6 things change:

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `webpack.config.js` | `component.config` → `base.config` + `CopyWebpackPlugin` + page discovery → `stage/` | Scaffold builds a library; we need pages output |
| 2 | `package.json` | Add `@splunk/react-page`, `copy-webpack-plugin`, component dep; drop `types:build` | UCC overlay deps |
| 3 | `pages/index.tsx` | Import from `@splunk/your-builder` not `@splunk/your-app` | Wire component library |
| 4 | `nav/default.xml` | Add `configuration` + `search` views | UCC already has those pages |
| 5 | `views/*.xml` | Template path → `your_app:/templates/...` | Runtime app name is the UCC app |
| 6 | `templates/*.html` | `page_path` → `/static/app/your_app/...` | Same — files merge into UCC output |

The `react/bin/adapt-for-ucc.mjs` script applies these transformations. You never need to do them manually.

### The CIMplicity Pattern

```
Your UCC app generates:     Your React app adds:
├── configuration page      ├── builder page (default view)
├── inputs page             ├── custom nav.xml
├── REST handlers           └── appserver/static/pages/builder.js
└── conf files
```

**Navigation XML** — builder page first, UCC config second:

```xml
<!-- default/data/ui/nav/default.xml -->
<nav>
    <view name="builder" default="true"/>
    <view name="configuration"/>
    <view name="search"/>
</nav>
```

### React Package Structure

Two packages in a monorepo:

1. **Component library** (`your-builder/`) — Reusable React components, state management, API helpers. No Splunk app coupling.
2. **Splunk app wrapper** (`your-app/`) — Entry point that mounts the component into Splunk's layout.

```typescript
// Entry point: src/main/webapp/pages/builder/index.tsx
// (generated by ucc:add-react — you don't write this)
import React from 'react';
import layout from '@splunk/react-page';
import { getUserTheme } from '@splunk/splunk-utils/themes';
import YourBuilder from '@splunk/your-builder';

getUserTheme().then((theme) => {
    layout(<YourBuilder />, { theme });
});
```

### Splunk REST Calls from React

```typescript
import { createRESTURL, getDefaultFetchInit } from '@splunk/splunk-utils';

async function postToEndpoint(endpoint: string, body: object) {
    const url = createRESTURL(
        `/servicesNS/-/your_app/${endpoint}`,
        { app: 'your_app', owner: 'nobody' }
    );
    const init = getDefaultFetchInit();  // Adds CSRF token + auth
    const response = await fetch(url, {
        ...init,
        method: 'POST',
        headers: { ...init.headers, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    return response.json();
}
```

### Key Splunk React UI Components

| Component | Use For |
|-----------|---------|
| `ControlGroup` | Every form field (wraps label + input + help text) |
| `Text` / `TextArea` / `Number` | Text inputs |
| `Select` / `ComboBox` | Dropdowns |
| `Switch` / `RadioBar` | Toggles and radio groups |
| `Accordion` | Collapsible sections |
| `Card` / `CardLayout` | Content containers |
| `TabLayout` | Tabbed panels |
| `Table` | Data display |
| `Modal` | Dialogs |
| `Button` / `WaitSpinner` | Actions and loading |
| `Message` / `MessageBar` | Alerts and notifications |
| `FormRows` | Dynamic add/remove field rows |

## 8. Build Pipeline

### The Key Rule: Never Override UCC-Generated Files

UCC's `ucc-gen build` generates `restmap.conf`, `web.conf`, and other conf files from `globalConfig.json`. If you put a custom `restmap.conf` in `package/default/`, UCC's file-copy step will silently **overwrite** the auto-generated version, breaking the admin REST handlers and SplunkWeb proxy.

**The pattern:** Put custom conf entries in `package/default.d/*.conf`. The build task appends these fragments to the UCC-generated files after `ucc-gen build` completes.

```
ucc/your_app/
├── package/
│   ├── default/
│   │   ├── collections.conf     ← OK: UCC doesn't generate this
│   │   └── transforms.conf      ← OK: UCC doesn't generate this
│   │   ❌ restmap.conf           ← NEVER: would overwrite UCC's admin handlers
│   │   ❌ web.conf               ← NEVER: would overwrite UCC's expose entries
│   ├── default.d/
│   │   ├── restmap.conf         ← Custom REST endpoints (appended after build)
│   │   └── web.conf             ← Custom expose entries (appended after build)
│   ├── bin/                     ← Custom scripts (modular inputs, REST handlers)
│   └── lib/                     ← Python libraries
└── globalConfig.json            ← UCC schema (generates restmap, web.conf, handlers)
```

### Build Stages

```bash
# Stage 1: UCC generates the add-on from globalConfig.json
task ucc:build
# Runs ucc-gen build, then appends default.d/*.conf fragments,
# strips local/ and local.meta (AppInspect failures)

# Stage 2: Build React UI (webpack)
task react:build

# Stage 3: Merge React assets into UCC output
task ucc:merge-react
# Copies: page JS bundles, Mako templates, view XMLs
# Merges: nav entries (adds custom views after <nav>)

# Stage 4: Link into Splunk + refresh
task ucc:link   # also fixes local/ and metadata/ permissions for Splunk
task dev:refresh

# Or all at once:
task ucc:dev   # = ucc:build + ucc:merge-react + ucc:link + dev:refresh
```

### What Each Stage Produces

| Stage | What's Generated | Source of Truth |
|---|---|---|
| `ucc:build` | REST handlers, conf files, UCC UI, admin endpoints | `globalConfig.json` |
| `ucc:build` post-step | Appended custom REST + expose entries | `package/default.d/*.conf` |
| `react:build` | Page bundles (JS) | `react/packages/your-app/src/` |
| `ucc:merge-react` | Combined app with UCC + React assets | React stage/ output |

### What's Safe to Put Where

| Location | Safe Contents | Unsafe Contents |
|---|---|---|
| `package/default/` | Files UCC doesn't generate (collections, transforms) | restmap.conf, web.conf, app.conf, inputs.conf |
| `package/default.d/` | Custom conf fragments to append | Anything that needs to replace (not append) |
| `package/bin/` | Custom Python scripts | Anything with same name as UCC-generated script |
| `package/lib/` | Python libraries | Conflicts with UCC's vendored libs |

## 9. Testing Strategy

### Manifest Validation (Offline)

```python
# Validate manifests parse without hitting any API
python test_manifests.py --validate
# Imports urc.cdk_bridge → ConcurrentDeclarativeSource → validates manifest schema
```

### Live API Testing

```python
# Test against real APIs (JSONPlaceholder needs no auth)
python test_manifests.py --live jsonplaceholder

# Test with credentials from environment
GITHUB_TOKEN=ghp_xxx python test_manifests.py --live github
```

### Splunk E2E Testing

```bash
# 1. Build the app
task ucc:build-full

# 2. Start dev Splunk
task dev:up

# 3. Link the app
task ucc:link && task dev:refresh

# 4. Configure via REST API
curl -sk -X POST https://localhost:8089/servicesNS/admin/your_app/your_app_account \
  -u admin:$SPLUNK_PASSWORD \
  -d name=test -d auth_type=none

curl -sk -X POST https://localhost:8089/servicesNS/admin/your_app/your_app_input \
  -u admin:$SPLUNK_PASSWORD \
  -d name=test_input -d account=test -d interval=60 -d index=main \
  -d 'manifest=...'

# 5. Verify events
curl -sk https://localhost:8089/services/search/jobs \
  -u admin:$SPLUNK_PASSWORD \
  -d search='search index=main sourcetype=your_app | head 10'
```

## 10. Lessons Learned — What Failed and Why

### Attempt 1: Vendor the Full Airbyte CDK — FAILED initially, then SUCCEEDED

**Round 1 (failed on Splunk 9.x / Python 3.9):** `git subtree add` the entire Airbyte CDK declarative runtime (~4000 files, 213 MB) into our Splunk app.

**Why it failed initially:**
- **Python 3.10+ syntax** throughout: `match/case`, `type` aliases, `ExceptionGroup`
- **C extensions** (`pydantic-core`, `orjson`) assumed incompatible with Splunk's Python 3.9
- **3 WIP commits** of patching before we gave up (`35f4686b`, `10a3477a`, `4e7f6b04`)

**Interim fix:** Wrote a lightweight reimplementation (17 files, 584 KB) using the same manifest format, Pydantic v1, `@component` decorator.

**Round 2 (succeeded on Splunk 10.2+ / Python 3.13):** With Splunk 10.2 shipping Python 3.13 (opt-in via `python.version = python3` in `app.conf`), the Python syntax and binary extension blockers vanished. We now pip-install `airbyte-cdk>=6` via `requirements.txt` — UCC's build handles vendoring automatically.

**Round 3 (CDK 6.x pivot complete):** Removed the entire hand-written custom engine (38 components, `engine.py`, `components/` dir with 10 files, `interpolation.py`, `registry.py`, `manifest.py`, `validate.py`, `models.py`, `models_generated.py`, `extensions.py`). The active runtime is now just 2 files: `package/lib/urc/__init__.py` + `package/lib/urc/cdk_bridge.py`. `cdk_bridge.py` wraps `ConcurrentDeclarativeSource` from the CDK. Post-build cleanup strips ~130 MB of unused transitive deps. Result: 62 MB total, AppInspect clean (0 failures, 0 errors, 113 passes). `test_manifests.py` imports from `urc.cdk_bridge` instead of the old `urc.engine`.

**Lesson:** Platform constraints change. What fails today may work when you target the next platform version. The lightweight reimplementation was the right call for Splunk 9.x, but once Splunk 10.2+ / Python 3.13 removed those blockers, the real CDK replaced everything. Resist the urge to reimplement upstream libraries — wait for the platform to catch up.

### Attempt 2: jsonschema v4.18+ (FAILED)

**What happened:** `jsonschema>=4.18` added a dependency on `rpds-py`, which is a Rust extension.

**The fix:** Pin `jsonschema==4.17.3`, which uses `pyrsistent` (pure Python) instead. *(Note: with the binary constraint revised — see below — this pin may no longer be necessary.)*

### ~~Attempt 3: pydantic v2 (NEVER ATTEMPTED)~~ → RESOLVED

**Original assumption:** Pydantic v2 requires `pydantic-core`, a Rust extension, which wouldn't work on Splunk.

**What actually happened:** Binary extensions work fine on Splunk. The "pure Python only" assumption was never tested — just carried forward. Splunk's own MLTK add-on ships numpy/scipy/sklearn (all heavy C extensions) on Splunkbase. With Splunk 10.2+ / Python 3.13, this is even more straightforward.

**The fix:** Upgraded to `pydantic>=2,<3`. This is now required by `airbyte-cdk>=6` anyway. Benefits: `--enum-field-as-literal all` in codegen (264 → 138 classes), proper discriminated unions, `ConfigDict`, and better validation errors.

### Attempt 4: pyrate-limiter v4 (CAUGHT IN AUDIT)

**What happened:** `pyrate-limiter>=4` uses `match/case` syntax (Python 3.10+).

**The fix:** Pin `pyrate-limiter<4`.

### Key Takeaway: The Dependency Constraint (Revised)

The original "pure Python only" rule was **wrong**. Splunk loads C extensions fine. On Splunk 10.2+ with `python.version = python3`, you get Python 3.13 — no syntax restrictions, full binary wheel support. The real constraints are:

1. **Target Splunk 10.2+** — opt-in to Python 3.13 via `python.version = python3` in `app.conf`. This removes all Python 3.9-era syntax restrictions.
2. **Correct platform wheels** — use `manylinux2014_x86_64` wheels for Splunk Cloud (x86_64)
3. **No `.pyc` files** — AppInspect rejects compiled Python bytecode; clean `__pycache__` during build
4. **AppInspect validation** — run `task ucc:appinspect` before shipping; use `--included-tags cloud` for Splunk Cloud compatibility

UCC handles vendoring automatically from `requirements.txt` during `ucc-gen build`. To verify in the Splunk container:
```bash
docker exec splunk-dev /opt/splunk/bin/splunk cmd python3 -c \
  "import sys; sys.path.insert(0, '/opt/splunk/etc/apps/your_app/lib'); import airbyte_cdk"
```

---

## Quick Reference

### UCC CLI Commands

```bash
ucc-gen build --source ucc/your_app -o ucc/output/    # Build add-on
ucc-gen package --path ucc/output/your_app -o dist/    # Package .tar.gz
ucc-gen init --addon-name your_app                     # Scaffold new app
```

### Git History (URC Development Timeline)

| Commit | What | Lesson |
|--------|------|--------|
| `364ba660` | UCC support added to template | Start with UCC scaffolding |
| `f7488f04` | Scaffold UCC app, vendor CDK, audit | Always audit deps FIRST |
| `4e7f6b04` → `35f4686b` | 3 WIP CDK patching attempts | Vendoring on Py 3.9 fails |
| `58de9bb4` | Clean URC runtime, registry pattern | Interim solution for Splunk 9.x |
| `b0656e0d` | Pydantic validation layer | Generated models from schema |
| `0419a769` | FULL E2E WORKING | 100 events, JSONPlaceholder → Splunk |
| `b4d4b278` | Phase 2: 36 components | Transformations, decoders, auth |
| `910dc66a` | 38 components, test connection | Partition routers, REST endpoint |
| `6a21f686` | Remove hand-written engine (38 components) | Replaced by pip-installed airbyte-cdk>=6 via cdk_bridge.py |
| `78d4c6c7` | Guided UI spec + test manifests | 5 API manifests, spec workflow |

### File Sizes (Production URC App)

```
Active runtime:     2 files — urc/__init__.py + urc/cdk_bridge.py
Vendored libs:      ~41 MB (CDK + deps, after post-build cleanup)
Total package:      ~62 MB (includes UCC UI bundle + vendored deps)
AppInspect:         0 failures, 0 errors, 113 passes
Target platform:    Splunk 10.2+ / Python 3.13
```
