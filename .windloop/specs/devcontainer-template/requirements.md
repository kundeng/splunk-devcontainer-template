# Requirements: Splunk Devcontainer Template

## Introduction

A reusable GitHub template repository for building Splunk applications. The template provides a uniform development environment across operating systems (macOS, Windows, Linux) by running developer tooling inside a VS Code devcontainer and Splunk as a separate Docker Compose service. This separation means contributors on any OS get identical tooling without manual setup, while Splunk runs in its own isolated container.

The devcontainer is the cross-platform equalizer: a Windows developer gets a Linux shell with the right Node/Python/Task versions; a macOS developer gets the same environment as a Linux CI runner. Splunk and optional sidecars run as compose services, independent of the devcontainer lifecycle.

## Glossary

- **Devcontainer**: VS Code dev container providing tools (Node, Python, Docker CLI, Go Task). Tools-only — no Splunk runtime inside.
- **Splunk container**: The `splunk/splunk` Docker image running as a compose service, accessed from the devcontainer via Docker-outside-of-Docker.
- **Skip-provision**: Restarting the Splunk container without re-running Ansible (~30s vs ~90s), enabled by a marker file in the `splunk-var` volume.
- **Symlink mount**: A single bind mount of `splunk/config/apps/` into the container at `/opt/splunk/dev-apps/`, with per-app symlinks in `/opt/splunk/etc/apps/` — avoids container recreation when adding new apps.
- **APP_NAME**: The primary app identifier. Used as the folder name, `[package] id` in `app.conf`, and React source directory name.
- **Staging environment**: A second Splunk instance with apps baked into the image, running on different ports (18000/18089/18088), for pre-release validation.

## Requirements

### Requirement 1: Uniform Cross-Platform Development Environment

**User Story:** As a developer on any OS (macOS, Windows, Linux), I want to open this repo in VS Code and get a fully configured development environment immediately, so that I don't need to manually install tools or worry about version mismatches.

#### Acceptance Criteria

1. WHEN a developer opens the repo in VS Code, THEN the devcontainer SHALL provide Node 22, Python 3.12, Docker CLI, and Go Task ready to use.
2. WHEN the devcontainer starts, THEN it SHALL contain no Splunk runtime — Splunk runs only in the compose stack.
3. WHEN the devcontainer rebuilds, THEN Splunk state SHALL be unaffected (volumes persist independently).
4. WHEN the devcontainer starts, THEN VS Code extensions SHALL be pre-configured (ruff, Splunk, Chrome debug, ESLint, Prettier, YAML).
5. WHEN the devcontainer is created, THEN `post-create.sh` SHALL install splunk-appinspect, ruff, and pytest, then build the Splunk dev image so `task splunk:up` is instant.
6. WHEN a developer uses the devcontainer on Windows or Linux, THEN the experience SHALL be identical to macOS — same tool versions, same shell, same task commands.

### Requirement 2: Fast Splunk Dev Loop (Skip-Provision)

**User Story:** As a developer, I want Splunk to restart quickly after Docker Desktop restarts or container recreation, so that I'm not blocked waiting 90 seconds for Ansible to re-run.

#### Acceptance Criteria

1. WHEN the Splunk container restarts and a `.provisioned` marker exists in `splunk-var`, THEN the entrypoint wrapper SHALL skip Ansible and start splunkd directly in ~30s.
2. WHEN the Splunk container starts for the first time (no marker), THEN full Ansible provisioning SHALL run and write the marker on success.
3. WHEN `task splunk:reprovision` is run, THEN the marker SHALL be removed and full Ansible provisioning SHALL run on next start.
4. WHEN skip-provision mode runs, THEN the Docker healthcheck state file SHALL be written so `checkstate.sh` passes.

### Requirement 3: Splunk Compose Services

**User Story:** As a developer, I want Splunk and optional sidecars to run as independent compose services, so that I can start/stop them without affecting my editor or tooling.

#### Acceptance Criteria

1. WHEN `task splunk:up` is run, THEN Splunk SHALL start as a compose service independent of the devcontainer lifecycle.
2. WHEN app source files in `splunk/config/apps/` are edited on the host, THEN changes SHALL be visible inside the Splunk container without packaging or container recreation.
3. WHEN the Splunk container is recreated, THEN indexed data and provisioning state SHALL persist via `splunk-var` and `splunk-etc` named volumes.
4. WHEN `splunk/stage/` contains tarballs, THEN they SHALL be bind-mounted at `/tmp/apps` inside the container for tarball-based installs.

### Requirement 4: Multi-App Development

**User Story:** As a developer, I want to develop and test multiple Splunk apps simultaneously in one repo, so that I can work on related apps together.

#### Acceptance Criteria

1. WHEN multiple app directories exist in `splunk/config/apps/`, THEN all SHALL be live-mounted and visible in Splunk simultaneously.
2. WHEN `task app:create APP_NAME=foo` is run, THEN a Splunk app skeleton SHALL be scaffolded with `app.conf` using `APP_NAME` as `[package] id` and a human-readable `[ui] label` (underscores → spaces).
3. WHEN `task app:create` completes, THEN the new app SHALL be immediately visible in Splunk without container recreation (~2–10s).
4. WHEN `task app:package APP_NAME=foo` is run, THEN `splunk/stage/foo.tgz` SHALL be produced.
5. WHEN `task app:provision` is run, THEN all local apps (or a specific one) SHALL be packaged and installed into the running Splunk instance.
6. WHEN `task app:sync-links` is run, THEN every app directory in `splunk/config/apps/` SHALL have a corresponding symlink in `/opt/splunk/etc/apps/` inside the container.

### Requirement 5: Splunkbase Dependency Management

**User Story:** As a developer, I want to declare Splunkbase app dependencies in a config file and install them idempotently, so that my team can reproduce the same Splunk environment.

#### Acceptance Criteria

1. WHEN `splunk/config/deps.yml` is populated with Splunkbase IDs/versions or direct URLs, THEN `task deps:install` SHALL download and install each dependency into the running Splunk instance.
2. WHEN `task deps:install` is run and a dependency is already installed at the correct version, THEN it SHALL be skipped (idempotent).

### Requirement 6: React UI Development

**User Story:** As a developer building a React-based Splunk app, I want scaffolding, a live dev loop via symlinked stage/, and a packaging pipeline for staging, so that I can iterate on UI quickly.

#### Acceptance Criteria

1. WHEN `task react:create` is run, THEN a React app SHALL be scaffolded via `npx @splunk/create`, the app package detected, `APP_NAME` synced in `.env`, and an initial build run so `stage/` exists.
2. WHEN `task react:link` is run, THEN `stage/` SHALL be symlinked into the dev Splunk container's `etc/apps/`. If `stage/` is missing, it SHALL auto-build first.
3. WHEN `task react:start` is run, THEN webpack watch SHALL start, updating `stage/` live. Splunk reads changes via the symlink from `react:link`.
4. WHEN `task react:build` is run, THEN a production webpack build SHALL always rebuild `stage/`.
5. WHEN `task react:package` is run, THEN `react:build` SHALL run first, then `stage/` SHALL be packaged as `splunk/stage/<APP_NAME>.tgz` for staging image bake-in.
6. WHEN `task react:add-page` is run, THEN a new page/component SHALL be added interactively via `@splunk/create`.

### Requirement 7: Task Automation

**User Story:** As a developer, I want all common operations available as short `task` commands, so that I don't need to remember long docker or curl commands.

#### Acceptance Criteria

1. WHEN `task --list` is run, THEN all tasks SHALL be organized into namespaces: `splunk:*`, `app:*`, `react:*`, `deps:*`, `python:*`, `test:*`.
2. WHEN a task accepts `APP_NAME`, THEN it SHALL load the value from `.env` as default when not provided as a CLI argument.
3. WHEN `task splunk:refresh` is run, THEN app configs and Splunk Web SHALL reload via REST API in ~2s without restarting splunkd.
4. WHEN `task splunk:build` is run, THEN the dev Splunk image SHALL be built explicitly. `task splunk:up` SHALL NOT rebuild — it only starts the container.

### Requirement 8: Debugging

**User Story:** As a developer, I want to attach a debugger to Python code running inside Splunk and to React code in the browser, so that I can step through code without print-debugging.

#### Acceptance Criteria

1. WHEN a Python script calls `enable_debugging()` from SA-VSCode, THEN VSCode SHALL be able to attach via debugpy on port 5678 using the "Python: Attach to Splunk" launch config.
2. WHEN VSCode opens the repo, THEN `launch.json` SHALL provide four debug configurations: Python remote attach, Chrome Splunk Web (dev :8000), Chrome React dev server (:3000), Chrome Staging Splunk Web (:18000).
3. WHEN the `splunk-config-dev` app is installed, THEN `web.conf` SHALL set `js_no_cache`, `minify_js=false`, `enableWebDebug=true`, `minify_css=false` for JS debugging.

### Requirement 9: Simultaneous Dev + Staging Environments

**User Story:** As a developer, I want to run a staging Splunk instance alongside dev on different ports, so that I can validate baked-in app builds without disrupting my dev environment.

#### Acceptance Criteria

1. WHEN `task splunk:up-staging` is run, THEN a staging Splunk instance SHALL start on ports 18000/18089/18088 with apps baked into the image, using separate volumes (`splunk-staging-var`, `splunk-staging-etc`) isolated from dev.
2. WHEN both dev and staging are running, THEN they SHALL not conflict (different ports, volumes, container names).

### Requirement 10: E2E Testing

**User Story:** As a maintainer, I want automated E2E tests that validate the full devcontainer and Splunk lifecycle, so that template changes don't silently break the developer experience.

#### Acceptance Criteria

1. WHEN `task test:e2e` is run, THEN it SHALL validate: devcontainer builds, `post-create.sh` completes, expected tools are on PATH, `task --list` works, and `docker compose config` validates for both compose files.
2. WHEN the lifecycle E2E tests run, THEN they SHALL validate: `splunk:up` starts a healthy Splunk, `app:create` scaffolds and makes an app visible via REST API, `app:package` produces a tarball, `app:provision` installs it, `react:build` and `react:package` produce a valid tgz, and `splunk:build-staging` builds a staging image.
3. WHEN lifecycle tests run, THEN they SHALL be split into focused, independently-runnable scripts with shared helpers and per-suite pass/fail reporting.
4. WHEN skip-provision E2E runs, THEN it SHALL verify: restarting a provisioned container recovers in ≤60s without Ansible, and `task splunk:reprovision` runs full Ansible and re-creates the marker.

### Requirement 11: Documentation

**User Story:** As a developer cloning this template, I want clear documentation on the architecture and all available commands, so that I can understand and use the template without reading the source.

#### Acceptance Criteria

1. WHEN a developer reads `AGENTS.md`, THEN it SHALL provide a concise repo summary with key entry points and a windloop hint. No architecture details.
2. WHEN a developer reads `ARCHITECTURE.md`, THEN it SHALL contain full documentation: directory structure, commands, debugging guides, how-it-works explanations, and troubleshooting.

### Requirement 12: Developer Workflow

**User Story:** As a developer using this template, I want a documented end-to-end workflow from first clone to daily iteration, so that I can get productive quickly.

#### Acceptance Criteria

1. WHEN a developer first opens the repo, THEN the documented workflow SHALL be: open in VS Code → "Reopen in Container" → `task splunk:up` → `task deps:install`.
2. WHEN a developer creates a new app, THEN the documented workflow SHALL be: `task app:create APP_NAME=foo` → app visible in Splunk → edit `.conf` → `task splunk:refresh`.
3. WHEN a developer does daily iteration, THEN the documented speed tiers SHALL be: refresh (~2s) → restartd (~10s) → new app (~2–10s) → image rebuild → full reset (~90s).
4. WHEN a developer wants to verify a release build, THEN the documented workflow SHALL be: `task app:package` → `task splunk:build-staging` → `task splunk:up-staging`.

### Requirement 13: App Naming Model

**User Story:** As a developer, I want a single `APP_NAME` identifier to tie together the folder name, Splunk package ID, and React source directory, so that there's no ambiguity or manual synchronization.

#### Acceptance Criteria

1. WHEN `APP_NAME` is set in `.env`, THEN it SHALL be the folder name under `splunk/config/apps/`, equal `[package] id` in `app.conf`, and (for React apps) match the directory under `packages/`.
2. WHEN `task app:create` runs, THEN `app.conf` SHALL have `[package] id = APP_NAME` and `[ui] label` derived by replacing underscores with spaces.
3. WHEN `.env` and `splunk.env.example` are read, THEN they SHALL clearly document that `APP_NAME` is the primary app folder name used by `app:*`, `react:*`, and CI/CD.

### Requirement 14: GitHub Actions CI/CD

**User Story:** As a maintainer, I want CI/CD workflows that build and validate the primary app on every push and publish a release artifact on tag, so that quality is enforced automatically.

#### Acceptance Criteria

1. WHEN a push or PR targets `main`, THEN `ci.yml` SHALL build the primary app from `splunk/config/apps/${APP_NAME}` and run local AppInspect via `VatsalJagani/splunk-app-action@v5`.
2. WHEN a tag matching `v*` is pushed, THEN `release.yml` SHALL build the primary app, run API AppInspect (cloud + SSAI checks), and upload the `.tgz` artifact.
3. WHEN workflows run, THEN they SHALL read `APP_NAME` from `.env` — no fragile grep of `app.conf`.
4. WHEN workflows are defined, THEN they SHALL use current action versions: `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`.

### Non-Functional

**NF 1**: Devcontainer rebuild completes in under 60 seconds.

**NF 2**: Splunk container restart in skip-provision mode completes in under 40 seconds.

**NF 3**: The template works on macOS (Docker Desktop) and Linux. The devcontainer provides a uniform environment for Windows developers via Docker Desktop + WSL2.

**NF 4**: All VS Code config uses the modern `customizations.vscode.*` schema — no deprecated top-level `settings` or `extensions` keys.

## Out of Scope

- Native (non-devcontainer) local development on macOS — planned as a future enhancement
- Splunk Cloud deployment (ACS API)
- Splunk clustering (multi-node)
- Internal logic of user-developed apps — only the dev workflow around them
- Automated Splunkbase publishing (release workflow builds + validates but does not auto-publish)
