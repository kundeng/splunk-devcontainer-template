# Splunk Dev Container Template

Docker-out-of-Docker development environment for building Splunk applications — Python custom search commands, React UI apps with Dashboard Studio, and multi-app workflows.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details: directory structure, commands, debugging, and how it works.

## Key Entry Points

- **`Taskfile.yml`** — all automation (`task dev:up`, `task app:create`, `task react:start`, etc.)
- **`.devcontainer/`** — devcontainer config, compose files
- **`splunk/`** — Dockerfile, entrypoint wrapper, app source, dependencies
- **`packages/`** — React/JS source (monorepo)

## Windloop

This project uses the `spec-driven-dev` skill for autonomous development. Run `/spec-help` to get started.
