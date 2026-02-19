# Tasks: Splunk Devcontainer Template

## Overview

Implementation proceeded in phases: core Splunk image + entrypoint wrapper → devcontainer + compose → app workflow → React + debugging → staging → CI/CD → E2E testing → documentation + hygiene. All tasks are complete as of the initial implementation.

## Tasks

- [x] 1. Core Splunk Image + Skip-Provision
  - [x] 1.1 Custom Splunk Dockerfile with entrypoint wrapper
    - Multi-stage Dockerfile (base → dev / staging); `entrypoint-wrapper.sh` skips Ansible when `.provisioned` marker exists in `splunk-var`; writes healthcheck state file in skip-provision mode
    - **Depends**: —
    - **Requirements**: 2.1, 2.2, 2.4
    - **Properties**: 1, 2

  - [x] 1.2 Write property test: skip-provision and first-run provisioning
    - Properties 1 and 2 (validates Requirement 2.1, 2.2, 2.4)
    - **Depends**: 1.1
    - **Properties**: 1, 2

- [x] 2. Devcontainer + Compose Services
  - [x] 2.1 Devcontainer configuration (devcontainer.json + post-create.sh)
    - Ubuntu 24.04 base; features: docker-outside-of-docker, node 22, python 3.12, go-task; `post-create.sh` installs appinspect/ruff/pytest and builds Splunk image; modern `customizations.vscode.*` schema
    - **Depends**: 1.1
    - **Requirements**: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, NF 1, NF 4
    - **Properties**: 6

  - [x] 2.2 Docker Compose dev services
    - `docker-compose.yml` with custom image, `platform: linux/amd64`, single bind mount (`splunk/config/apps/` → `/opt/splunk/dev-apps/`), named volumes (`splunk-var`, `splunk-etc`), optional sidecars commented out
    - **Depends**: 1.1
    - **Requirements**: 3.1, 3.2, 3.3, 3.4
    - **Properties**: 7, 8

  - [x] 2.3 Write property test: devcontainer tools-only
    - Property 6 (validates Requirement 1.1, 1.2, 1.3, 1.6, NF 4)
    - **Depends**: 2.1
    - **Properties**: 6

- [x] 3. App Workflow
  - [x] 3.1 Taskfile — all namespaces
    - All `splunk:*`, `app:*`, `react:*`, `deps:*`, `python:*`, `test:*` tasks; `APP_NAME` loaded from `.env`; `splunk:refresh` via REST API; `splunk:build` separate from `splunk:up`
    - **Depends**: 2.2
    - **Requirements**: 7.1, 7.2, 7.3, 7.4
    - **Properties**: 11

  - [x] 3.2 App symlink manager (`app:sync-links`)
    - Scans `splunk/config/apps/`, creates symlinks in `/opt/splunk/etc/apps/` via `docker exec --user root`, removes stale symlinks; runs automatically after `splunk:up` and `app:create`
    - **Depends**: 2.2, 3.1
    - **Requirements**: 4.1, 4.3, 4.6
    - **Properties**: 4, 7

  - [x] 3.3 App scaffold (`app:create`, `app:package`, `app:provision`)
    - `app:create` scaffolds skeleton with correct `app.conf`; `app:package` produces `.tgz`; `app:provision` installs into running Splunk; all use `APP_NAME`
    - **Depends**: 3.1, 3.2
    - **Requirements**: 4.2, 4.3, 4.4, 4.5, 13.1, 13.2, 13.3
    - **Properties**: 13

  - [x] 3.4 Write property test: multi-app symlink creation and app naming
    - Properties 4 and 13 (validates Requirement 4.1, 4.3, 4.6, 13.1, 13.2, 13.3)
    - **Depends**: 3.2, 3.3
    - **Properties**: 4, 13

- [x] 4. Dependency Manager
  - [x] 4.1 `deps.yml` schema + `deps-install.py`
    - Stdlib-only Python script; regex-based YAML parser; Splunkbase auth + REST install; URL fallback; idempotent (skip already-installed at correct version)
    - **Depends**: 3.1
    - **Requirements**: 5.1, 5.2
    - **Properties**: 5

  - [x] 4.2 Write property test: idempotent dependency install
    - Property 5 (validates Requirement 5.1, 5.2)
    - **Depends**: 4.1
    - **Properties**: 5

- [x] 5. React UI + Debugging
  - [x] 5.1 React workflow tasks (`react:create`, `react:start`, `react:build-install`)
    - `react:create` via `npx @splunk/create` + Splunk skeleton + sync-links; `react:start` webpack HMR on :3000; `react:build-install` builds and installs into Splunk
    - **Depends**: 3.2, 3.3
    - **Requirements**: 6.1, 6.2, 6.3
    - **Properties**: 13

  - [x] 5.2 Debugging support (launch.json + splunk-config-dev + debugpy port)
    - Four VS Code launch configs; port 5678 exposed in compose; `splunk-config-dev` app with web.conf debug options
    - **Depends**: 2.2
    - **Requirements**: 8.1, 8.2, 8.3
    - **Properties**: 9

  - [x] 5.3 Write property test: debug configurations complete
    - Property 9 (validates Requirement 8.1, 8.2, 8.3)
    - **Depends**: 5.2
    - **Properties**: 9

- [x] 6. Staging Environment
  - [x] 6.1 Staging compose + Taskfile staging tasks
    - `docker-compose.staging.yml` on ports 18000/18089/18088, `staging` Dockerfile target, separate volumes; `splunk:up-staging`, `down-staging`, `clean-staging`, `logs-staging`
    - **Depends**: 1.1, 3.1
    - **Requirements**: 9.1, 9.2
    - **Properties**: 8

  - [x] 6.2 Write property test: staging compose isolation
    - Property 8 (validates Requirement 9.1, 9.2)
    - **Depends**: 6.1
    - **Properties**: 8

- [x] 7. CI/CD Workflows
  - [x] 7.1 GitHub Actions CI workflow (`ci.yml`)
    - Push/PR to main: shellcheck + local AppInspect via `splunk-app-action@v5`; reads `APP_NAME` from `.env`; current action versions
    - **Depends**: 3.3
    - **Requirements**: 14.1, 14.3, 14.4
    - **Properties**: 14

  - [x] 7.2 GitHub Actions release workflow (`release.yml`)
    - Tag push `v*`: API AppInspect (cloud + SSAI) + artifact upload; reads `APP_NAME` from `.env`; Splunkbase credentials from secrets
    - **Depends**: 7.1
    - **Requirements**: 14.2, 14.3, 14.4
    - **Properties**: 14

  - [x] 7.3 Remove deprecated `_disabled.release.yml`
    - Deleted outdated workflow (deprecated `::set-output`, old action versions, stub publish step)
    - **Depends**: 7.2
    - **Requirements**: 14.4
    - **Properties**: 14

  - [x] 7.4 Write property test: CI/CD workflows valid
    - Property 14 (validates Requirement 14.1, 14.2, 14.3, 14.4)
    - **Depends**: 7.1, 7.2
    - **Properties**: 14

- [x] 8. E2E Tests
  - [x] 8.1 Static E2E test (`devcontainer-test.sh`)
    - Validates: devcontainer build, post-create.sh, tools on PATH, `task --list`, `docker compose config` for both compose files
    - **Depends**: 2.1, 3.1, 6.1
    - **Requirements**: 10.1
    - **Properties**: 10, 11

  - [x] 8.2 Lifecycle E2E tests (focused scripts + shared helpers)
    - `helpers.sh` (shared functions); `test-boot.sh`, `test-app-lifecycle.sh`, `test-deps-install.sh`, `test-react-build.sh`, `test-staging.sh`; `run-lifecycle.sh` runner with per-suite reporting
    - **Depends**: 3.3, 4.1, 5.1, 6.1
    - **Requirements**: 10.2, 10.3
    - **Properties**: 4, 5, 7, 8, 13

  - [x] 8.3 Skip-provision E2E test (`test-skip-provision.sh`)
    - Validates restart recovery ≤60s without Ansible; validates `splunk:reprovision` runs full Ansible and re-creates marker
    - **Depends**: 1.1, 8.2
    - **Requirements**: 10.4
    - **Properties**: 12

- [x] 9. Documentation + Template Hygiene
  - [x] 9.1 AGENTS.md (concise repo summary + windloop hint)
    - Concise entry points, key commands, windloop snippet — no architecture details
    - **Depends**: —
    - **Requirements**: 11.1

  - [x] 9.2 ARCHITECTURE.md (full architecture docs)
    - Directory structure, commands reference, debugging guides, how-it-works, troubleshooting
    - **Depends**: —
    - **Requirements**: 11.2, 12.1, 12.2, 12.3, 12.4

  - [x] 9.3 Template hygiene cleanup
    - Removed concrete app (`Splunk_App_for_Anomaly_Detection`) from git; updated `.gitignore` to ignore user apps except `splunk-config-dev/`; deleted stale dirs; fixed spec path drift; replaced concrete app references with `<your_app>/` placeholders in docs
    - **Depends**: —
    - **Requirements**: 1.2, 11.2

## Notes

- All tasks complete as of Feb 2026. The spec was migrated from `.windloop/devcontainer-modernize/` to this location in the new format.
- Native macOS local development (without devcontainer) is planned as a future spec — see Out of Scope in `requirements.md`.
- The `splunk/splunk` image is x86-64 only; `platform: linux/amd64` is required in compose files for Apple Silicon Macs.
