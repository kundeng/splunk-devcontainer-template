---
name: UCC generator CLI and build details
description: Exact CLI commands, flags, build pipeline, templates, and integration details for ucc-gen
type: reference
---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `ucc-gen init` | Scaffold new add-on |
| `ucc-gen build` | Build from source (default if no subcommand) |
| `ucc-gen package` | Archive to tar.gz |
| `ucc-gen validate` | Run appinspect |
| `ucc-gen publish` | Publish to Splunkbase |

## ucc-gen init flags
- `--addon-name` (required), `--addon-display-name` (required), `--addon-input-name` (required)
- `--addon-version`, `--addon-rest-root`, `--overwrite`, `--need-proxy`, `--add-license`, `--include-author`
- Scaffolds: globalConfig.json, package/{app.manifest, static/, LICENSES/, README.txt}

## ucc-gen build flags
- `--source` (default: `package`) — folder with app.manifest + source
- `--config` (auto-detected) — globalConfig.json or .yaml in parent of source
- `--ta-version` (from git tags if not specified)
- `-o/--output` (default: `output/`)
- `--python-binary-name`, `--pip-version`, `--pip-legacy-resolver`, `--pip-custom-flag`
- `--build-custom-ui`, `--overwrite`

## Build pipeline steps
1. Validate Python binary
2. Determine version (git or --ta-version)
3. Read app.manifest → get addon name
4. Load + validate globalConfig.json against JSON schema
5. Install Python libs via pip to lib/
6. Copy package/ base structure
7. Generate REST handlers (programmatic, not templated)
8. Generate modular inputs (Jinja2 templates)
9. Generate .conf files (app.conf, inputs.conf, restmap.conf, etc.)
10. Generate import_declare_test.py
11. Call additional_packaging.py hooks (if present)
12. Generate OpenAPI spec

## import_declare_test.py
- Manipulates sys.path to add lib/ and OS-specific lib dirs
- MUST be imported first in all bin/ scripts
- Handles OS-dependent libraries (windows_lib, darwin_lib, linux_lib)

## Requirements
- Python >=3.9,<3.14
- ucc-gen v6.2.0 current
- Auto-installs splunktaucclib>=6.6.0, solnlib>=5.5.0 (if OAuth)

## Cimplicity-ai-app patterns (reference UCC app)
- Monorepo: Lerna + Yarn workspaces
- REST-first: PersistentServerConnectionApplication handlers, no modular inputs
- Decoupled Python logic (no Splunk deps in core business logic)
- CI: ucc-gen build → copy into packages → webpack → ucc-gen package
- OS-dependent libs in globalConfig.json meta section
- Encrypted fields for API keys

## SplunkUI devcontainer (Bre77)
- Minimal: Splunk baked into devcontainer image, 1 port, 4 extensions
- Our template is far more comprehensive — only worth borrowing:
  - `mutantdino.resourcemonitor` extension
  - `degit` pattern for lightweight app fetching
