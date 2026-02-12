# Specification: Devcontainer Modernize

## Overview

Docker-out-of-Docker development environment for building Splunk applications. The devcontainer provides tools (Node, Python, Docker CLI, Go Task); Splunk and optional sidecars run as separate compose services with bind-mounted apps for live editing. A custom entrypoint wrapper skips Ansible re-provisioning on restart (~30s vs 90s). A staging environment runs alongside dev on different ports with apps baked into the image.

## Goals

- [x] Clean separation: devcontainer (IDE + tools) vs runtime services (Splunk, sidecars)
- [x] Fast inner dev loop: live app mounts, skip-provision restarts, React HMR
- [x] Multi-app support: develop and package multiple apps in one repo
- [x] Idempotent dependency install from Splunkbase
- [x] Modern tooling: Node 22, Python 3.12, ruff, Go Task
- [x] Simultaneous dev + staging environments on different ports
- [x] Integrated debugging: Python (debugpy) and React/JS (Chrome DevTools)
- [x] Automated E2E testing via `@devcontainers/cli`

## Requirements

### R1: Devcontainer Configuration

**R1.1**: Open the repo in VS Code → fully configured dev environment (Node 22, Python 3.12, Docker CLI, Go Task) ready immediately.

**R1.2**: Devcontainer is tools-only — no Splunk runtime. Rebuilding the devcontainer does not affect Splunk state.

**R1.3**: Modern VS Code extensions and settings pre-configured (ruff, Splunk extensions, Chrome debug). No deprecated settings.

**R1.4**: `post-create.sh` installs splunk-appinspect, ruff, pytest, then builds the Splunk dev image (`task splunk:build`) so `task splunk:up` is instant.

### R2: Skip-Provision Splunk Image

**R2.1**: Restart the Splunk container without Ansible re-provisioning (~30s instead of 90s). Marker file in `splunk-var` volume tracks provisioning state. The entrypoint wrapper writes the Docker healthcheck state file so `checkstate.sh` passes in skip-provision mode.

**R2.2**: First start (no marker) → full Ansible provisioning → write marker. Subsequent starts → skip Ansible, start splunkd directly.

**R2.3**: `task splunk:reprovision` forces full re-provisioning by removing the marker and restarting.

### R3: Docker Compose Services

**R3.1**: `task splunk:up` starts Splunk (and optional sidecars: UF, Cribl Edge, PostgreSQL). Runtime services are independent from the devcontainer lifecycle.

**R3.2**: App source directories in `splunk/config/apps/` are live-mounted into the Splunk container at `/opt/splunk/etc/apps/<app>`. Config and static asset changes are picked up without packaging or container recreation.

**R3.3**: `splunk-var` named volume persists indexed data and provisioning state. `splunk-etc` named volume persists `/opt/splunk/etc` (Splunk config, installed apps) across container recreations — this enables skip-provision on recreation.

**R3.4**: `splunk/stage/` is bind-mounted at `/tmp/apps` for tarball-based installs.

### R4: Multi-App Support

**R4.1**: Multiple Splunk apps developed simultaneously in `splunk/config/apps/`, each with its own directory.

**R4.2**: `task app:create APP_NAME=foo` scaffolds a new Splunk app skeleton. The generated `app.conf` uses `APP_NAME` as `[package] id` and derives `[ui] label` as a human-readable version (underscores replaced with spaces).

**R4.3**: `task app:package APP_NAME=foo` produces `splunk/stage/foo.tgz`.

**R4.4**: `task app:provision` packages and installs all local apps (or a specific one) into a running Splunk instance.

**R4.5**: `task app:sync-links` ensures every app in `splunk/config/apps/` is visible inside the running Splunk container. Runs automatically as part of `splunk:up` and after `app:create`.

**R4.6**: `task app:create` scaffolds the app and makes it immediately visible in Splunk without container recreation.

### R5: Dependency Management

**R5.1**: Splunkbase app dependencies declared in `splunk/config/deps.yml`. Supports both Splunkbase IDs/versions and direct URLs.

**R5.2**: `task deps:install` downloads and installs dependencies idempotently — skips already-installed apps at the correct version.

### R6: React UI Development

**R6.1**: `task react:create` scaffolds a new React-based Splunk app via `npx @splunk/create`. After scaffolding, it creates a corresponding Splunk app skeleton (if one doesn't exist) and syncs symlinks so the app is visible in Splunk without container recreation.

**R6.2**: `task react:start` runs a webpack dev server with HMR on port 3000, proxying API calls to Splunk REST API (port 8089).

**R6.3**: `task react:build-install` builds JS, copies output to `appserver/static/`, packages, and installs into Splunk.

> **Note**: The template supports embedding Dashboard Studio definitions via `@splunk/dashboard-core` and `@splunk/dashboard-presets`. This is a supported pattern, not a tested feature.

### R7: Taskfile

**R7.1**: All tasks organized into namespaces: `splunk:*`, `app:*`, `react:*`, `deps:*`, `python:*`, `test:*`.

**R7.2**: Tasks load `APP_NAME` from `.env` as default when not provided as a CLI argument.

**R7.3**: `task splunk:refresh` reloads app configs and restarts Splunk Web via the REST API (`services/apps/local/_reload` + `services/server/control/restart_webui`). This is the fastest way to pick up .conf and static asset changes (~2s, no splunkd restart).

**R7.4**: `task splunk:build` explicitly builds the dev Splunk image. `task splunk:up` does not rebuild — it only starts the container. Image builds happen in `post-create.sh` (first time) or explicitly via `task splunk:build` (after Dockerfile changes).

### R8: Debugging

**R8.1**: Python debugging via SA-VSCode add-on (declared in `deps.yml`) + debugpy on port 5678. VSCode launch config with path mappings from `splunk/config/apps/` to `/opt/splunk/etc/apps/`.

**R8.2**: VSCode launch configurations for: Python remote attach (debugpy), Chrome Splunk Web (dev :8000), Chrome React dev server (:3000), Chrome Staging Splunk Web (:18000).

**R8.3**: `splunk-config-dev` app sets `web.conf` options (`js_no_cache`, `minify_js`, `enableWebDebug`, `minify_css`) for JS debugging.

### R9: Simultaneous Dev + Staging

**R9.1**: Dev and staging Splunk instances run simultaneously. Staging uses `docker-compose.staging.yml` with ports 18000/18089/18088, `staging` Dockerfile target, and its own volumes (`splunk-staging-var`, `splunk-staging-etc`).

**R9.2**: `task splunk:up-staging`, `splunk:down-staging`, `splunk:clean-staging`, `splunk:logs-staging` manage the staging environment.

### R10: E2E Testing

**R10.1**: An automated E2E test script uses `@devcontainers/cli` to validate: devcontainer builds, post-create.sh completes, expected tools are on PATH, `task --list` works, and `docker compose config` validates for both dev and staging compose files.

**R10.2**: `task test:e2e` runs the E2E test script. `task test:all` runs lint + E2E.

**R10.3**: An expanded E2E test validates the full Splunk lifecycle inside the devcontainer: `splunk:up` starts Splunk and it becomes healthy, `app:create` scaffolds a custom search command app, `app:package` packages it, `app:provision` installs it into running Splunk, the app is visible via Splunk REST API, a React app scaffold exists in `packages/` and `react:build-install` can build and deploy it, `splunk:build-staging` builds a staging image, and all test artifacts are cleaned up so they don't leak to the main branch.

**R10.4**: Lifecycle tests are split into focused, independently-runnable scripts with shared helpers. A runner script orchestrates them and reports per-suite results. Each test script is self-contained: sets up what it needs, verifies, and cleans up.

**R10.5**: Skip-provision and reprovision behavior is validated end-to-end: (a) restarting a provisioned Splunk container recovers in ≤60s without running Ansible, (b) `task splunk:reprovision` removes the marker, runs full Ansible, and re-creates the marker.

### R11: Documentation

**R11.1**: `AGENTS.md` is a concise repo summary with key entry points and a windloop hint. No architecture details.

**R11.2**: `ARCHITECTURE.md` contains full documentation: directory structure, commands, debugging guides, how-it-works explanations, troubleshooting.

### Non-Functional

**NF1**: Devcontainer rebuild completes in under 60 seconds.

**NF2**: Splunk container recreation with skip-provision completes in under 40 seconds.

**NF3**: Works on macOS (Docker Desktop) and Linux.

**NF4**: Modern config schema throughout (`customizations.vscode.*`, not legacy top-level settings).

### R12: Developer Workflow

The natural developer workflow, from setup to daily iteration:

**R12.1 — First-time setup**:
1. Open repo in VS Code → "Reopen in Container"
2. `post-create.sh` installs tools + builds Splunk image
3. `task splunk:up` → Splunk starts, syncs app symlinks, full Ansible provisioning on first boot
4. `task deps:install` → install Splunkbase dependencies

**R12.2 — Creating a new app**:
1. `task app:create APP_NAME=foo` → scaffolds app, creates symlink, refreshes Splunk (~2-10s, no container recreation)
2. App is immediately visible in Splunk
3. Edit .conf files → `task splunk:refresh` to reload (~2s)

**R12.3 — Creating a React app**:
1. `task react:create` → scaffolds React app + Splunk app skeleton, syncs symlinks, refreshes Splunk (~2-10s, no container recreation)
2. `task react:start` → webpack dev server with HMR on :3000
3. `task react:build-install` → build, copy to appserver/static, install into Splunk

**R12.4 — Daily dev loop** (fastest to slowest):
- Edit .conf / dashboard → `task splunk:refresh` (~2s, no restart)
- Edit Python code → `task splunk:restartd` (~10s, restart splunkd process)
- New app → `task app:create` (~2-10s, symlink + refresh, no recreate)
- Dockerfile change → `task splunk:build` then `task splunk:up`
- Full reset → `task splunk:clean` then `task splunk:up` (full Ansible ~90s)

**R12.5 — Staging verification**:
1. `task app:package APP_NAME=foo` → package apps into `splunk/stage/`
2. `task splunk:build-staging` → build staging image with apps baked in
3. `task splunk:up-staging` → start staging on ports 18000/18089/18088

### R13: App Naming Model

**R13.1**: `APP_NAME` in `.env` identifies the **primary app** being developed. It is the folder name under `splunk/config/apps/` and must equal `[package] id` in `app.conf` (Splunk convention). For React-based apps, the same `APP_NAME` links the React source in `packages/<APP_NAME>/` to the Splunk app in `splunk/config/apps/<APP_NAME>/`.

**R13.2**: `app:create` generates `app.conf` with `[package] id = ${APP_NAME}` and `[ui] label` derived from `APP_NAME` (underscores → spaces). All Taskfile commands that accept `APP_NAME` use it as a folder name.

**R13.3**: `.env` comments and `splunk.env.example` clearly document that `APP_NAME` is the primary app folder name, used by `app:*`, `react:*`, and CI/CD workflows.

### R14: GitHub Actions CI/CD

**R14.1**: `.github/workflows/ci.yml` runs on push/PR to main: builds the primary app package from `splunk/config/apps/${APP_NAME}`, runs AppInspect validation via `VatsalJagani/splunk-app-action@v5` (local mode, no credentials required).

**R14.2**: `.github/workflows/release.yml` runs on tag push (`v*`): builds the primary app, runs AppInspect via Splunkbase API (cloud + SSAI checks using `SPLUNKBASE_USERNAME`/`SPLUNKBASE_PASSWORD` secrets), uploads the `.tgz` artifact.

**R14.3**: Workflows read `APP_NAME` from `.env` to identify the primary app folder. No fragile grep of `app.conf`.

**R14.4**: All GitHub Action versions are current: `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`.

**R14.5**: The old `_disabled.release.yml` is removed.

## Constraints

- App live-mounting (R3.2) uses a single bind mount (`splunk/config/apps/` → `/opt/splunk/dev-apps/`) plus symlinks (`/opt/splunk/etc/apps/<app>` → `/opt/splunk/dev-apps/<app>`). This avoids per-app bind mounts which would require container recreation for each new app. The symlink pattern is validated by Splunk’s own `@splunk/create` tooling (`yarn run link:app`).
- Symlinks live inside the `splunk-etc` named volume and persist across container restarts.
- `app:sync-links` runs via `docker exec` and requires the Splunk container to be running. If Splunk is not running, symlink creation is deferred to the next `splunk:up`.
- `app:sync-links` cleans up stale symlinks (pointing to deleted app directories).

## Out of Scope

- Splunk Cloud deployment (ACS API)
- Splunk clustering (multi-node)
- Internal logic of user-developed apps — only the dev workflow around them
- Automated Splunkbase publishing (release workflow builds + validates but does not auto-publish)
