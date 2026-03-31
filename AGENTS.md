# Splunk Dev Container Template

Docker-out-of-Docker development environment for building Splunk applications.

## Repository Layout

- **`Taskfile.yml`** — all automation (`task dev:up`, `task ucc:build`, `task ucc:appinspect`, etc.)
- **`.devcontainer/`** — devcontainer config, compose files (Python 3.13)
- **`splunk/`** — Dockerfile, entrypoint wrapper, Splunk 10.2 container (Python 3.13)
- **`ucc/urc_app/`** — URC (Universal REST Client) add-on source
- **`docs/ucc-golden-path.md`** — practical UCC handbook with lessons learned

## URC App

The URC app is a Splunk add-on that collects data from REST APIs using declarative YAML manifests.

- **Runtime:** pip-installed `airbyte-cdk>=6,<7` (not a custom engine)
- **Active code:** 2 files — `package/lib/urc/__init__.py` + `package/lib/urc/cdk_bridge.py`
- **`cdk_bridge.py`** wraps `ConcurrentDeclarativeSource` from the Airbyte CDK 6.x
- **Build:** `ucc-gen build` installs CDK from `requirements.txt`; post-build cleanup strips ~130 MB of unused transitive deps
- **Platform:** Splunk 10.2+ / Python 3.13 (`python.version = python3` in `app.conf`)
- **AppInspect:** 0 failures, 0 errors, 113 passes (62 MB total / 41 MB lib)

## Windloop

This project uses the `spec-driven-dev` skill for autonomous development. Run `/spec-help` to get started.
