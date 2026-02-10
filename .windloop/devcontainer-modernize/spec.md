# Specification: Devcontainer Modernize

## Overview

Modernize and simplify the splunk-devcontainer-template repository. The repo has grown organically and is now inconsistent. This spec covers: devcontainer configuration, Docker/docker-compose usage, a custom Splunk image with skip-provision support, multi-app development workflow, and React UI integration.

## Goals

- [ ] Clean separation: devcontainer (IDE + tools) vs runtime services (Splunk, UF, Cribl, DB)
- [ ] Fast inner dev loop: React HMR, live app mounts, no-rebuild restarts
- [ ] Multi-app support: develop and package multiple apps in one repo
- [ ] Idempotent dependency install: supporting apps (MLTK, etc.) installed reliably
- [ ] Custom Splunk image that skips Ansible re-provisioning on restart
- [ ] Modern tooling: Node 22, Python 3.12, current Splunk UI packages

## Architecture

### Tech Stack

- **Devcontainer base**: `mcr.microsoft.com/devcontainers/base:ubuntu-24.04`
- **Node**: 22 LTS (via devcontainer feature)
- **Python**: 3.12 (via devcontainer feature)
- **Docker**: docker-outside-of-docker feature (host daemon)
- **Task runner**: Go Task (via devcontainer feature)
- **Splunk image**: `splunk/splunk:9.4.0` with custom entrypoint wrapper
- **React UI**: `@splunk/react-ui` v5.x, `@splunk/create`, webpack-based

### Directory Structure

```
.devcontainer/
  devcontainer.json          # Tools-only dev container
  docker-compose.yml         # Runtime services (Splunk, UF, Cribl, DB)
  post-create.sh             # Minimal tool installation
splunk/
  Dockerfile                 # Custom Splunk image (stock + entrypoint wrapper)
  entrypoint-wrapper.sh      # Skip-provision logic
  config/
    apps/                    # Splunk app source directories (multi-app)
      splunk-config-dev/     # Dev-mode Splunk settings (web.conf, etc.)
      MyApp/                 # Example app
    deps.yml                 # Declarative dependency list (Splunkbase apps)
  stage/                     # Built tarballs (.tgz) — gitignored
packages/                    # React/JS source (monorepo-friendly)
  MyApp/                     # @splunk/create scaffolded app
Taskfile.yml                 # All automation
.env                         # Secrets/config — gitignored
splunk.env.example           # Template for .env
AGENTS.md                    # General repo info for AI agents
```

## Requirements

### R1: Devcontainer Configuration

**R1.1**: As a developer, I should be able to open the repo in VS Code and get a fully configured dev environment with Node 22, Python 3.12, Docker CLI, and Go Task, so that I can start working immediately.

**R1.2**: As a developer, I should have the devcontainer be tools-only (no Splunk runtime inside it), so that rebuilding the devcontainer doesn't destroy Splunk state.

**R1.3**: As a developer, I should have modern VS Code extensions and settings pre-configured, so that formatting, linting, and Splunk-specific tooling work out of the box.

**R1.4**: As a developer, I should have a minimal `post-create.sh` that installs only necessary tools (splunk-appinspect, ruff, pytest), so that container creation is fast and doesn't install dead/deprecated tools.

### R2: Custom Splunk Docker Image (Skip-Provision)

**R2.1**: As a developer, I should be able to restart the Splunk container without Ansible re-provisioning, so that restarts take ~15s instead of 60-120s and don't corrupt app installs.

**R2.2**: The custom image should wrap the official entrypoint with a script that detects whether Splunk has already been provisioned (marker file in a persistent volume) and skips Ansible on subsequent starts.

**R2.3**: On first start (no marker), the wrapper should delegate to the stock entrypoint for full Ansible provisioning.

**R2.4**: A `task splunk:reprovision` command should force full re-provisioning by removing the marker file and restarting.

**R2.5**: The custom Dockerfile should be minimal: FROM the official image, COPY the wrapper script, override ENTRYPOINT. No other modifications.

### R3: Docker Compose Services

**R3.1**: As a developer, I should be able to start Splunk and optional sidecars (UF, Cribl Edge, PostgreSQL) via `task splunk:up`, so that runtime services are managed independently from the devcontainer.

**R3.2**: The compose file should use the custom Splunk image (built from `splunk/Dockerfile`).

**R3.3**: App source directories should be bind-mounted directly into the Splunk container at `/opt/splunk/etc/apps/<app_name>`, so that config and static asset changes are picked up without tarball packaging.

**R3.4**: Named volumes (`splunk-var`) should persist indexed data. The `splunk-etc` volume should NOT be used when using direct bind mounts (bind mounts overlay the named volume for specific paths).

**R3.5**: The staging directory (`splunk/stage/`) should be mounted at `/tmp/apps` for tarball-based installs.

### R4: Multi-App Support

**R4.1**: As a developer, I should be able to develop multiple Splunk apps simultaneously in `splunk/config/apps/`, each with its own directory.

**R4.2**: As a developer, I should be able to scaffold a new app with `task app:create APP_NAME=foo`, which creates a proper Splunk app skeleton.

**R4.3**: As a developer, I should be able to package any app with `task app:package APP_NAME=foo`, producing `splunk/stage/foo.tgz`.

**R4.4**: As a developer, I should be able to provision all local apps (or a specific one) into a running Splunk instance with `task app:provision`.

**R4.5**: The compose file should dynamically mount all app directories from `splunk/config/apps/` into the Splunk container. Since docker-compose doesn't support globs, the Taskfile should generate or manage the mount list.

### R5: Dependency Management (Supporting Apps)

**R5.1**: As a developer, I should be able to declare Splunkbase app dependencies in a `splunk/config/deps.yml` file, so that required apps (MLTK, Python for Scientific Computing, etc.) are installed automatically.

**R5.2**: The dependency install should be idempotent: if an app is already installed at the correct version, skip it.

**R5.3**: Dependencies should be installed via `task deps:install` which downloads tarballs to `splunk/stage/` and installs them into the running Splunk instance.

**R5.4**: The `deps.yml` format should support both Splunkbase app IDs/versions and direct URLs.

### R6: React UI Development

**R6.1**: As a developer, I should be able to scaffold a new React-based Splunk app with `task react:create`, which runs `npx @splunk/create`.

**R6.2**: As a developer, I should be able to start a React dev server with HMR via `task react:start`, so that UI changes are reflected instantly without rebuilding.

**R6.3**: As a developer, I should be able to build and install a React app into Splunk with `task react:build-install`, which builds the JS, copies output to `appserver/static/`, packages, and installs.

**R6.4**: The React dev server should proxy API calls to the Splunk REST API (port 8089).

### R7: Taskfile Modernization

**R7.1**: All tasks should be organized into clear namespaces: `splunk:*`, `app:*`, `react:*`, `deps:*`, `python:*`.

**R7.2**: Tasks should load `APP_NAME` from `.env` as a default when not provided as a CLI argument.

**R7.3**: The `react:start` task must be implemented (currently missing).

**R7.4**: Legacy/dead tasks and aliases should be removed.

**R7.5**: A `splunk:reprovision` task should exist to force full Ansible re-provisioning.

### Non-Functional

**NF1**: Container creation (devcontainer rebuild) should complete in under 60 seconds.

**NF2**: Splunk restart (after initial provisioning) should complete in under 20 seconds.

**NF3**: The template should work on macOS (Docker Desktop) and Linux.

**NF4**: All configuration files should use modern schema (devcontainer.json `customizations.vscode.*`, not legacy `settings`/`extensions`).

## Data Models

### deps.yml schema

```yaml
dependencies:
  - name: python_for_scientific_computing
    splunkbase_id: 2882
    version: "4.2.3"
  - name: Splunk_ML_Toolkit
    splunkbase_id: 2890
    version: "5.6.1"
  - name: custom_app
    url: "https://example.com/custom_app-1.0.tgz"
```

## Testing Strategy

- **Property tests**: Shell script validation (shellcheck), YAML lint, docker-compose config validation
- **E2E tests**: Manual verification via `task splunk:up` + `task app:provision` + browser check
- **Test command**: `shellcheck .devcontainer/*.sh splunk/entrypoint-wrapper.sh`
- **Lint command**: `task lint` (runs shellcheck + yamllint)

## Constraints

- Must use the official `splunk/splunk` Docker image as the base (not a fork)
- The entrypoint wrapper must be forward-compatible with Splunk image updates
- No Yeoman (`yo`), no `splunk-packaging-toolkit` (slim) — both are deprecated
- `@splunk/create` should be used via `npx`, not installed globally
- The devcontainer must NOT run Splunk inside it
- Docker Compose services are started/stopped independently from the devcontainer lifecycle

## Out of Scope

- Splunk Cloud deployment
- CI/CD pipeline implementation (release.yml exists but is disabled)
- Splunk clustering (multi-node) configuration
- Custom search commands or modular input implementation
- Splunk Dashboard Studio (JSON dashboards) — focus is on React UI framework
