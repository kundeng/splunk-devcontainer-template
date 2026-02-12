# Tasks: Devcontainer Modernize

<!--
STATUS: [ ] pending | [x] done | [~] partial/skipped | [!] blocked
OPTIONAL: add * after bracket to mark optional, e.g. [ ]* or [x]*
PRIORITY: P0 critical | P1 important | P2 nice-to-have
DEPENDS: task IDs that must complete first
REQUIREMENTS: requirement IDs from spec.md this task fulfills
PROPERTIES: property IDs from design.md this task should satisfy
-->

## Phase 1: Core Infrastructure

### T1: Custom Splunk Image + Entrypoint Wrapper `P0`
- **Description**: Create `splunk/entrypoint-wrapper.sh` and update `splunk/Dockerfile`. The wrapper checks for a `.provisioned` marker in `/opt/splunk/var/`. If present and `FORCE_PROVISION` is not set, skip Ansible and start splunkd directly. Otherwise delegate to stock `/sbin/entrypoint.sh` and write the marker on success.
- **Requirements**: R2.1, R2.2, R2.3
- **Properties**: P1, P2, P3
- **Acceptance criteria**:
  - [x] `splunk/entrypoint-wrapper.sh` exists and is executable
  - [x] `splunk/Dockerfile` is minimal: FROM official image, COPY wrapper, override ENTRYPOINT
  - [x] First start runs full Ansible provisioning and writes marker
  - [x] Subsequent starts skip Ansible and start splunkd directly
  - [x] `FORCE_PROVISION=true` forces full re-provisioning
  - [x] SIGTERM is trapped and splunkd stops gracefully
- **Files**: `splunk/entrypoint-wrapper.sh`, `splunk/Dockerfile`
- **Verify**: `shellcheck splunk/entrypoint-wrapper.sh`
- **Status**: [x]

### T2: Modernize devcontainer.json `P0`
- **Description**: Rewrite `.devcontainer/devcontainer.json` with modern schema. Ubuntu 24.04 base, Node 22, Python 3.12, docker-outside-of-docker, go-task features. Modern `customizations.vscode.*` format. Remove deprecated settings. Clean extension list.
- **Requirements**: R1.1, R1.3, NF4
- **Properties**: P6
- **Acceptance criteria**:
  - [x] Uses `mcr.microsoft.com/devcontainers/base:ubuntu-24.04`
  - [x] Node 22, Python 3.12 features
  - [x] `customizations.vscode.extensions` and `customizations.vscode.settings` (not legacy top-level)
  - [x] No deprecated `pwa-chrome`, `python.linting.*`, `python.formatting.*` settings
  - [x] Ports: 8000, 8089, 8088, 3000
  - [x] No `shutdownAction: stopCompose`
- **Files**: `.devcontainer/devcontainer.json`
- **Verify**: JSON syntax valid
- **Status**: [x]

### T3: Modernize docker-compose.yml `P0`
- **Description**: Rewrite `.devcontainer/docker-compose.yml` to use the custom Splunk image (built from `splunk/Dockerfile`). Add `SPLUNK_GENERAL_TERMS` env var (required by newer Splunk images). Include commented-out sidecar templates (UF, Cribl, PostgreSQL). Mount `splunk/stage` at `/tmp/apps`. Use `splunk-var` named volume only. *(Originally included `-f docker-compose.apps.yml` support — superseded by T19 symlink strategy.)*
- **Depends**: T1
- **Requirements**: R3.1, R3.2, R3.3, R3.4
- **Properties**: P7
- **Acceptance criteria**:
  - [x] Uses `build: context: ../splunk` (custom image)
  - [x] `SPLUNK_GENERAL_TERMS` env var included
  - [x] `splunk-var` named volume for persistence
  - [x] `splunk/stage:/tmp/apps` bind mount
  - [x] Commented-out sidecar service templates
  - [x] `docker compose config` validates successfully
- **Files**: `.devcontainer/docker-compose.yml`
- **Verify**: `docker compose -f .devcontainer/docker-compose.yml config`
- **Status**: [x]

### T4: Minimal post-create.sh `P0`
- **Description**: Rewrite `.devcontainer/post-create.sh` to be minimal. Install only: splunk-appinspect, ruff, pytest. Set up PATH. Create `.env` from example if missing. Create directory structure. Remove: yo, generator-splunk-app, splunk-packaging-toolkit, global @splunk/create, isort, black, flake8 (replaced by ruff).
- **Requirements**: R1.4
- **Acceptance criteria**:
  - [x] Installs only: splunk-appinspect, ruff, pytest
  - [x] No yo, slim, global @splunk/create
  - [x] Creates `.env` from `splunk.env.example` if missing
  - [x] Creates `splunk/config/apps/`, `splunk/stage/`, `packages/` dirs
  - [x] Completes in <30s
- **Files**: `.devcontainer/post-create.sh`
- **Verify**: `shellcheck .devcontainer/post-create.sh`
- **Status**: [x]

## Phase 2: Multi-App & Dependencies

### T5: App Mount Generator `P0` *(superseded by T19)*
- **Description**: *(Originally)* Added `app:generate-mounts` task that scanned `splunk/config/apps/` and generated `docker-compose.apps.yml` with per-app bind mounts. *(Superseded by T19: replaced with single parent bind mount + symlinks via `app:sync-links`.)*
- **Depends**: T3
- **Requirements**: R4.5, R3.3
- **Properties**: P4, P7
- **Acceptance criteria**:
  - [x] `task app:generate-mounts` creates valid `docker-compose.apps.yml`
  - [x] Each directory in `splunk/config/apps/` gets a bind mount entry
  - [x] `splunk:up` calls `app:generate-mounts` first
  - [x] Empty `splunk/config/apps/` produces valid empty override
- **Files**: `Taskfile.yml`, `.devcontainer/docker-compose.apps.yml` (generated, gitignored)
- **Verify**: Create test app dirs, run task, validate YAML
- **Status**: [x]

### T6: Dependency Manager (deps.yml + task) `P1`
- **Description**: Create `splunk/config/deps.yml` schema and `deps:install` task. The task reads the YAML, checks if each app is installed at the correct version, downloads missing tarballs from Splunkbase API or direct URL, and installs them idempotently.
- **Depends**: T3
- **Requirements**: R5.1, R5.2
- **Properties**: P5
- **Acceptance criteria**:
  - [x] `splunk/config/deps.yml` exists with example entries
  - [x] `task deps:install` downloads and installs missing deps
  - [x] Already-installed deps are skipped
  - [x] Supports both Splunkbase IDs and direct URLs
  - [x] Clear error messages on auth/network failures
- **Files**: `Taskfile.yml`, `splunk/config/deps.yml`
- **Verify**: Run `task deps:install` twice, verify idempotent
- **Status**: [x]

## Phase 3: Taskfile Modernization

### T7: Rewrite Taskfile.yml `P0`
- **Description**: Modernize the entire Taskfile. Add missing `react:start` task. Add `splunk:reprovision` task. Remove legacy aliases (`splunk:start`, `splunk:stop`). Clean up help output. Fix `PACKAGES_DIR` and `REACT_PACKAGES_DIR` paths. *(Originally integrated `app:generate-mounts` into `splunk:up` — superseded by T19 which uses `app:sync-links`.)*
- **Depends**: T1, T3, T5, T6
- **Requirements**: R7.1, R7.2, R6.1, R6.2, R6.3, R4.2, R4.3, R4.4
- **Acceptance criteria**:
  - [x] `react:start` task exists and works
  - [x] `splunk:reprovision` task exists
  - [x] No legacy aliases
  - [x] `splunk:up` generates mounts first
  - [x] All compose commands use both compose files
  - [x] `deps:install` task integrated
  - [x] Help output is clean and accurate
- **Files**: `Taskfile.yml`
- **Verify**: `task --list` shows all tasks
- **Status**: [x]

## Phase 4: Documentation & Cleanup

### T8: Rewrite AGENTS.md `P1`
- **Description**: Rewrite `AGENTS.md` as general repository information for AI agents. Move project requirements and design details to `.windloop/` (already done via spec). AGENTS.md should describe: what the repo is, directory structure, how to use it (commands), architecture overview, and key patterns. Add windloop snippet.
- **Depends**: T7
- **Requirements**: R11.1
- **Acceptance criteria**:
  - [x] AGENTS.md is concise general repo info
  - [x] No project requirements or design specs in AGENTS.md
  - [x] Includes windloop snippet
  - [x] Accurately reflects the modernized repo structure
- **Files**: `AGENTS.md`
- **Status**: [x]

### T9: Update splunk.env.example and .gitignore `P1`
- **Description**: Update `splunk.env.example` with current Splunk version (9.4.0), add `SPLUNK_GENERAL_TERMS`, clean up comments. Remove `custom_setup.sh` (dead code). *(Originally added `docker-compose.apps.yml` to `.gitignore` — removed by T19.)*
- **Depends**: T3
- **Requirements**: (cleanup)
- **Acceptance criteria**:
  - [x] `splunk.env.example` has `SPLUNK_VERSION=9.4.0` and `SPLUNK_GENERAL_TERMS`
  - [x] ~~`.gitignore` includes `docker-compose.apps.yml`~~ *(removed by T19)*
  - [x] `custom_setup.sh` deleted
- **Files**: `splunk.env.example`, `.gitignore`, `.devcontainer/custom_setup.sh` (delete)
- **Status**: [x]

### T10: Update launch.json `P2`
- **Description**: Fix `.vscode/launch.json` to use modern `chrome` debug type instead of deprecated `pwa-chrome`. Update `webRoot` paths to match new directory structure.
- **Requirements**: R1.3
- **Acceptance criteria**:
  - [x] Debug type is `chrome` not `pwa-chrome`
  - [x] Paths match modernized structure
- **Files**: `.vscode/launch.json`
- **Status**: [x]

### T11: Update README.md `P2`*
- **Description**: Update README to reflect modernized architecture, commands, and workflow. Optional — can be deferred.
- **Depends**: T7, T8
- **Acceptance criteria**:
  - [x]* README reflects current architecture
- **Files**: `README.md`
- **Status**: [x]

## Phase 5: Integration Testing

### T12: Devcontainer Build + Compose Validation `P0`
- **Description**: Verify the full devcontainer builds successfully using the devcontainer CLI. Validate docker-compose config parses. Run shellcheck on shell scripts. Confirm Taskfile parses with `task --list`. *(Originally validated `app:generate-mounts` — superseded by T19 `app:sync-links`.)*
- **Depends**: T1, T2, T3, T4, T5, T6, T7
- **Requirements**: R11.2
- **Acceptance criteria**:
  - [x] `devcontainer build --workspace-folder .` succeeds
  - [x] `docker compose -f .devcontainer/docker-compose.yml config` validates
  - [x] `shellcheck splunk/entrypoint-wrapper.sh` passes (or only minor warnings)
  - [x] `shellcheck .devcontainer/post-create.sh` passes
  - [x] `task --list` shows all expected tasks
  - [x] ~~`task app:generate-mounts` produces valid docker-compose.apps.yml~~ *(superseded by T19: `app:sync-links`)*
  - [x] JSON syntax of devcontainer.json and launch.json is valid
- **Verify**: Run all checks above
- **Status**: [x]

### T13: Multi-Stage Dockerfile (dev + staging targets) `P1`
- **Description**: Refactor `splunk/Dockerfile` into a multi-stage build following the kveditor pattern. A `base` stage installs acl and the entrypoint wrapper. A `dev` stage (default for docker-compose) uses the skip-provision wrapper with bind-mounted apps. A `staging` stage bakes all tarballs from `splunk/stage/` into the image for a self-contained deployment. Add `target: dev` to docker-compose.yml. Add `splunk:build-staging` and `splunk:up-staging` tasks.
- **Depends**: T1, T3, T7
- **Requirements**: R2.1, R3.2
- **Acceptance criteria**:
  - [x] Dockerfile has `base`, `dev`, and `staging` stages
  - [x] `docker-compose.yml` specifies `target: dev`
  - [x] `task splunk:build-staging` builds the staging image via `docker build --target staging`
  - [x] `task splunk:up-staging` runs a self-contained staging container
  - [x] Dev stage uses entrypoint-wrapper (skip-provision)
  - [x] Staging stage uses entrypoint-wrapper.sh (same as dev, auto-discovers /tmp/apps/*.tgz) and bakes in tarballs
  - [x] `docker compose config` still validates
- **Files**: `splunk/Dockerfile`, `.devcontainer/docker-compose.yml`, `Taskfile.yml`
- **Verify**: `docker compose config`, `task --list`
- **Status**: [x]

### T14: Dual Compose, Debugging, Scope & Layout Overhaul `P1`
- **Description**: Correct the template scope — React+Dashboard Studio apps and Python custom search commands are the primary use cases, not out of scope. Add `docker-compose.staging.yml` for simultaneous dev+staging (different ports). Add Python debugging support (SA-VSCode in deps.yml, port 5678, debugpy launch config). Add Chrome launch configs for React dev server, Splunk Web, and staging. Fix `splunk-config-dev` app structure (move files into `default/`, add `metadata/`). Rewrite AGENTS.md with correct scope, example layout (anomaly detection suite), debugging docs, and dev+staging architecture. Update spec.md and design.md to reflect expanded scope.
- **Depends**: T13
- **Requirements**: R8.1, R8.2, R8.3, R9.1, R9.2
- **Properties**: P8, P9
- **Tests**:
  - [x] Property test — P8: Staging compose isolation (validates R9.1)
  - [x] Property test — P9: Debug configurations complete (validates R8.1, R8.2)
- **Acceptance criteria**:
  - [x] `docker-compose.staging.yml` exists with staging target, ports 18000/18089/18088
  - [x] Dev and staging can run simultaneously (different container names, ports, volumes)
  - [x] Port 5678 exposed in dev compose for Python debugpy
  - [x] SA-VSCode added to `deps.yml`
  - [x] `launch.json` has Python attach, Chrome Splunk Web, Chrome React dev, Chrome staging configs
  - [x] `splunk-config-dev` has proper `default/` and `metadata/` structure
  - [x] Taskfile has `splunk:up-staging`, `splunk:down-staging`, `splunk:clean-staging`, `splunk:logs-staging` using compose
  - [x] AGENTS.md documents: scope, layout, example app suite, debugging (Python + React), dev+staging architecture
  - [x] `docker compose config` validates for both dev and staging compose files
  - [x] `task --list` shows all staging tasks
  - [x] spec.md updated: R8 (debugging), R9 (staging), R6.5 (Dashboard Studio), Out of Scope corrected
  - [x] design.md updated: Module 7 (staging), Module 8 (debugging), properties P8-P11, architecture diagram
- **Files**: `.devcontainer/docker-compose.yml`, `.devcontainer/docker-compose.staging.yml`, `.vscode/launch.json`, `splunk/config/deps.yml`, `Taskfile.yml`, `AGENTS.md`, `splunk/config/apps/splunk-config-dev/`, `.windloop/devcontainer-modernize/spec.md`, `.windloop/devcontainer-modernize/design.md`
- **Verify**: `docker compose -f .devcontainer/docker-compose.yml config`, `docker compose -f .devcontainer/docker-compose.staging.yml config`, `task --list`
- **Status**: [x]

## Phase 6: E2E Testing with Devcontainer CLI

### T15: E2E Test Suite with `@devcontainers/cli` `P0`
- **Description**: Create an automated E2E test script that uses `@devcontainers/cli` to validate the full devcontainer lifecycle. The script should: (1) `devcontainer build` the workspace, (2) `devcontainer up` to start the container, (3) `devcontainer exec` to verify tools are present (node, python3, docker, task, ruff, splunk-appinspect), (4) `devcontainer exec` to run `task --list` and verify `app:sync-links` exists, (5) `devcontainer exec` to run `docker compose config` for both dev and staging compose files, (6) clean up the container. The test should exit non-zero on any failure and produce clear output. *(Originally validated `app:generate-mounts` — updated by T19.)*
- **Depends**: T12, T14
- **Requirements**: R10.1
- **Properties**: P10, P11
- **Tests**:
  - [x] Property test — P10: Devcontainer CLI build + up succeeds (validates R10.1)
  - [x] Property test — P11: Devcontainer tools and automation available (validates R10.1)
- **Acceptance criteria**:
  - [x] Test script exists at `tests/e2e/devcontainer-test.sh` (or similar)
  - [x] `devcontainer build --workspace-folder .` succeeds
  - [x] `devcontainer up --workspace-folder .` succeeds and post-create.sh completes
  - [x] All expected tools verified via `devcontainer exec`: node, python3, docker, task, ruff
  - [x] `task --list` succeeds inside the devcontainer
  - [x] ~~`task app:generate-mounts` produces valid YAML~~ → `task app:sync-links` exists *(updated by T19)*
  - [x] `docker compose config` validates for both dev and staging compose files
  - [x] Test cleans up containers on exit (trap handler)
  - [x] Script is shellcheck-clean
- **Files**: `tests/e2e/devcontainer-test.sh`
- **Verify**: `shellcheck tests/e2e/devcontainer-test.sh`, run the test locally
- **Status**: [x]

### T16: `task test:e2e` Runner `P1`
- **Description**: Add a `test:e2e` task to `Taskfile.yml` that runs the E2E test script from T15. Ensure `@devcontainers/cli` is available (install via npx or check for global install). Add a `test:all` task that runs both lint and E2E tests.
- **Depends**: T15
- **Requirements**: R10.2
- **Acceptance criteria**:
  - [x] `task test:e2e` runs the devcontainer E2E test script
  - [x] `task test:all` runs lint + E2E
  - [x] Works without globally installed `@devcontainers/cli` (uses npx fallback)
  - [x] `task test:e2e` executes and all E2E checks pass (exit 0)
  - [x] `task test:all` executes and all checks pass (exit 0)
- **Files**: `Taskfile.yml`
- **Verify**: `task test:e2e` passes, `task test:all` passes
- **Status**: [x]

### T17: Expanded E2E — Splunk Lifecycle, App Workflows, Staging `P0`
- **Description**: Create `tests/e2e/splunk-lifecycle-test.sh` that runs inside an already-started devcontainer (from T15). It starts Splunk via `task splunk:up`, waits for healthy, then exercises the full app workflow: (1) `task app:create` a test custom search command app with a Python script, (2) verify symlink creation via `app:sync-links`, (3) `task app:package` + `task app:provision` to install it, (4) verify the app is visible via Splunk REST API, (5) scaffold a minimal React app in `packages/`, run `task react:build-install` to build and deploy it, (6) `task splunk:build-staging` to verify staging image builds, (7) clean up all test artifacts. *(Originally used `app:generate-mounts` — updated by T19 to use symlink verification.)*
- **Depends**: T15, T16
- **Requirements**: R10.3
- **Properties**: P10, P11
- **Acceptance criteria**:
  - [x] `task splunk:up` starts Splunk and it becomes healthy (REST API responds)
  - [x] `task app:create APP_NAME=test_cmd_app` scaffolds app with correct structure
  - [x] Test adds a Python custom search command script to the app
  - [x] ~~`task app:generate-mounts` includes the new test app~~ → symlink verified in container *(updated by T19)*
  - [x] `task app:package APP_NAME=test_cmd_app` creates a tarball
  - [x] `task app:provision APP_NAME=test_cmd_app` installs it into running Splunk
  - [x] Splunk REST API confirms the test app is installed
  - [x] A minimal React app is scaffolded in `packages/test_react_app/` (non-interactive)
  - [x] `task react:build-install` builds and deploys the React app into Splunk
  - [x] `task splunk:build-staging` builds the staging image successfully
  - [x] All test artifacts are cleaned up (test apps, packages, stage tarballs removed)
  - [x] `task test:e2e` runs both static and lifecycle tests end-to-end
  - [x] Script is shellcheck-clean
  - [x] No test artifacts leak to main branch (cleanup trap removes all test apps/packages/tarballs)
- **Files**: `tests/e2e/splunk-lifecycle-test.sh`, `tests/e2e/devcontainer-test.sh` (updated), `Taskfile.yml` (updated)
- **Verify**: `task test:e2e` passes with all lifecycle checks
- **Status**: [x]

## Phase 7: Workflow Refinement

### T18: Dev Workflow Refinement — Build/Up Separation, Refresh, Auto-Mount `P0`
- **Description**: Refine the developer workflow: (1) separate `splunk:build` from `splunk:up` so image builds only happen in `post-create.sh` or explicitly, (2) add `splunk:refresh` task for config/static reload via REST API without restart, (3) persist `/opt/splunk/etc` as a named volume so skip-provision works on container recreation, (4) fix Docker healthcheck in skip-provision path, (5) make `app:create` and `react:create` auto-regenerate mounts and recreate Splunk if running, (6) fix `LOCAL_WORKSPACE_FOLDER` default for compose paths, (7) update README.md, ARCHITECTURE.md, spec.md, and design.md to document the developer workflow (R12).
- **Depends**: T17
- **Requirements**: R4.6, R7.3, R7.4, R12.1, R12.2, R12.3, R12.4, R12.5
- **Acceptance criteria**:
  - [x] `task splunk:up` does not rebuild the image
  - [x] `task splunk:build` explicitly builds the dev image
  - [x] `task splunk:refresh` reloads configs via REST API (200/200)
  - [x] `splunk-etc` named volume persists across container recreation
  - [x] Docker healthcheck passes in skip-provision mode
  - [x] `task app:create` auto-regenerates mounts and recreates Splunk if running
  - [x] `task react:create` creates Splunk app skeleton + auto-remounts
  - [x] `post-create.sh` builds Splunk image and generates mounts
  - [x] README.md documents developer workflow (R12)
  - [x] ARCHITECTURE.md documents developer workflow, build/up separation, refresh, persistence
  - [x] spec.md and design.md updated with R4.6, R7.3, R7.4, R12, NF2
  - [x] `task test:e2e` passes (46 checks, 0 failures)
- **Files**: `Taskfile.yml`, `.devcontainer/docker-compose.yml`, `.devcontainer/docker-compose.staging.yml`, `.devcontainer/post-create.sh`, `splunk/entrypoint-wrapper.sh`, `tests/e2e/splunk-lifecycle-test.sh`, `README.md`, `ARCHITECTURE.md`, `.windloop/devcontainer-modernize/spec.md`, `.windloop/devcontainer-modernize/design.md`
- **Verify**: `task test:e2e` passes
- **Status**: [x]

### T19: Symlink Mount Strategy — Replace Per-App Bind Mounts `P0`
- **Description**: Replace per-app bind mounts with a single parent bind mount (`splunk/config/apps/` → `/opt/splunk/dev-apps/`) + symlinks (`/opt/splunk/etc/apps/<app>` → `/opt/splunk/dev-apps/<app>`). This eliminates container recreation when adding new apps. (1) Add single `dev-apps` bind mount to `docker-compose.yml`, remove `docker-compose.apps.yml` override mechanism, (2) create `app:sync-links` task that scans `/opt/splunk/dev-apps/` and creates symlinks via `docker exec`, cleans up stale symlinks, (3) update `splunk:up` to call `app:sync-links` after container start, (4) update `app:create` to call `app:sync-links` + `splunk:refresh` instead of recreating the container, (5) update `react:create` similarly, (6) remove `app:generate-mounts` task and `__remount-if-running` helper, (7) remove `docker-compose.apps.yml` from `.gitignore`, (8) update `post-create.sh` to remove `app:generate-mounts` call, (9) update E2E tests to validate symlink-based flow, (10) update README.md, ARCHITECTURE.md with new mount strategy.
- **Depends**: T18
- **Requirements**: R3.2, R3.3, R4.5, R4.6, R6.1, R12.2, R12.3, R12.4
- **Properties**: P4, P7
- **Acceptance criteria**:
  - [x] `docker-compose.yml` has single bind mount: `splunk/config/apps/` → `/opt/splunk/dev-apps/`
  - [x] No `docker-compose.apps.yml` — removed entirely
  - [x] `task app:sync-links` creates symlinks in `/opt/splunk/etc/apps/` for each app in `dev-apps/`
  - [x] `task app:sync-links` cleans up stale symlinks (app dir deleted on host)
  - [x] `task app:sync-links` is idempotent (existing symlinks preserved)
  - [x] `task splunk:up` calls `app:sync-links` after container start
  - [x] `task app:create` calls `app:sync-links` + `splunk:refresh` (no container recreation)
  - [x] `task react:create` calls `app:sync-links` + `splunk:refresh` (no container recreation)
  - [x] `app:generate-mounts` task removed
  - [x] `__remount-if-running` helper removed
  - [x] `post-create.sh` no longer calls `app:generate-mounts`
  - [x] File changes in `splunk/config/apps/<app>/` are visible at `/opt/splunk/etc/apps/<app>/` through symlink
  - [x] `task test:e2e` passes with updated tests
  - [x] README.md updated: no mention of `docker-compose.apps.yml` or container recreation for new apps
  - [x] ARCHITECTURE.md updated: symlink strategy documented
- **Files**: `Taskfile.yml`, `.devcontainer/docker-compose.yml`, `.devcontainer/post-create.sh`, `.gitignore`, `tests/e2e/devcontainer-test.sh`, `tests/e2e/splunk-lifecycle-test.sh`, `README.md`, `ARCHITECTURE.md`
- **Verify**: `task test:e2e` passes (18 static + 25 lifecycle = 43 checks, 0 failures)
- **Status**: [x]

## Phase 8: Test Restructuring

### T20: Split Monolithic Lifecycle Test into Focused Scripts `P1`
- **Description**: Replace the 425-line `splunk-lifecycle-test.sh` monolith with focused, independently-runnable test scripts. Extract shared helpers (log, pass, fail, check_app_rest, wait_for_splunk, splunk_exec) into `tests/e2e/helpers.sh`. Split test logic into: `test-boot.sh` (build, up, healthy), `test-app-lifecycle.sh` (create, symlink, package, provision, REST verify), `test-deps-install.sh` (parse, install, idempotency), `test-react-build.sh` (scaffold, build-install, REST verify), `test-staging.sh` (build staging, up, verify baked apps). Create `tests/e2e/run-lifecycle.sh` runner that sources helpers, starts Splunk once, runs each suite, reports per-suite results. Update `devcontainer-test.sh` to call the new runner. Remove old `splunk-lifecycle-test.sh`.
- **Depends**: T17
- **Requirements**: R10.3, R10.4
- **Properties**: P10, P11
- **Acceptance criteria**:
  - [x] `tests/e2e/helpers.sh` contains shared test functions
  - [x] Each test script is self-contained and sources helpers
  - [x] `tests/e2e/run-lifecycle.sh` orchestrates all suites and reports per-suite pass/fail
  - [x] `devcontainer-test.sh` calls `run-lifecycle.sh` instead of `splunk-lifecycle-test.sh`
  - [x] Old `splunk-lifecycle-test.sh` is removed
  - [x] All existing test coverage is preserved (no regressions)
  - [x] Each script is shellcheck-clean
- **Files**: `tests/e2e/helpers.sh`, `tests/e2e/test-boot.sh`, `tests/e2e/test-app-lifecycle.sh`, `tests/e2e/test-deps-install.sh`, `tests/e2e/test-react-build.sh`, `tests/e2e/test-staging.sh`, `tests/e2e/test-skip-provision.sh`, `tests/e2e/run-lifecycle.sh`, `tests/e2e/devcontainer-test.sh`
- **Verify**: `bash tests/e2e/run-lifecycle.sh` passes all suites
- **Status**: [x]

### T21: Skip-Provision and Reprovision E2E Test `P1`
- **Description**: Add `tests/e2e/test-skip-provision.sh` that validates the entrypoint-wrapper marker logic end-to-end. Test 1 (skip-provision): verify marker exists, restart Splunk via `task splunk:restart`, time recovery (must be ≤60s), verify marker preserved and apps still accessible via REST API. Test 2 (reprovision): run `task splunk:reprovision`, wait for full Ansible to complete, verify marker is re-created, verify apps still work. Integrate into `run-lifecycle.sh` runner.
- **Depends**: T20
- **Requirements**: R10.5
- **Properties**: P1, P2, P3, P12
- **Acceptance criteria**:
  - [x] `tests/e2e/test-skip-provision.sh` exists and is shellcheck-clean
  - [x] Skip-provision test: restart recovers in ≤60s, marker preserved, apps accessible
  - [x] Reprovision test: marker removed, full Ansible runs, marker re-created, apps work
  - [x] `run-lifecycle.sh` includes the new test suite
  - [x] All tests pass when run inside the devcontainer
- **Files**: `tests/e2e/test-skip-provision.sh`, `tests/e2e/run-lifecycle.sh`
- **Verify**: `bash tests/e2e/run-lifecycle.sh` passes all suites including skip-provision
- **Status**: [x]

---

## Phase 9: App Naming & CI/CD

### T22: Clarify App Naming Model `P13`
- **Description**: Update `app:create` in `Taskfile.yml` to generate `[ui] label` as a human-readable version of `APP_NAME` (underscores → spaces) instead of using the raw folder name. Update `.env` comments and `splunk.env.example` to clearly document that `APP_NAME` is the primary app folder name (= `[package] id`), used by `app:*`, `react:*`, and CI/CD. Update `ARCHITECTURE.md` with the naming model.
- **Depends**: none
- **Requirements**: R13.1, R13.2, R13.3
- **Properties**: P13
- **Acceptance criteria**:
  - [x] `app:create` generates `[ui] label` with underscores replaced by spaces
  - [x] `app:create` generates `[package] id = ${APP_NAME}` (already true, verified)
  - [x] `splunk.env.example` documents APP_NAME as primary app folder name
  - [x] `ARCHITECTURE.md` has a section explaining the app naming model
- **Files**: `Taskfile.yml`, `splunk.env.example`, `ARCHITECTURE.md`
- **Verify**: `task app:create APP_NAME=Test_My_App` → check `app.conf` has `label = Test My App` and `id = Test_My_App`
- **Status**: [x]

### T23: GitHub Actions CI Workflow `P14`
- **Description**: Create `.github/workflows/ci.yml` that runs on push/PR to main. Steps: checkout, read `APP_NAME` from `.env`, run `VatsalJagani/splunk-app-action@v5` with `app_dir` pointing to `splunk/config/apps/${APP_NAME}`, local AppInspect mode (no credentials needed). Also run shellcheck on `splunk/*.sh` and `tests/**/*.sh`.
- **Depends**: T22
- **Requirements**: R14.1, R14.3, R14.4
- **Properties**: P14
- **Acceptance criteria**:
  - [x] `.github/workflows/ci.yml` exists and is valid YAML
  - [x] Uses `actions/checkout@v4`
  - [x] Reads `APP_NAME` from `.env`
  - [x] Uses `VatsalJagani/splunk-app-action@v5` with `local_app_inspect: true`
  - [x] Includes shellcheck linting step
  - [x] Triggers on push to main and pull_request
- **Files**: `.github/workflows/ci.yml`
- **Verify**: `cat .github/workflows/ci.yml | python3 -c "import sys,yaml; yaml.safe_load(sys.stdin)"` exits 0
- **Status**: [x]

### T24: GitHub Actions Release Workflow `P14`
- **Description**: Create `.github/workflows/release.yml` that runs on tag push (`v*`). Steps: checkout, read `APP_NAME` from `.env`, run `splunk-app-action@v5` with Splunkbase API AppInspect (cloud + SSAI checks using `SPLUNKBASE_USERNAME`/`SPLUNKBASE_PASSWORD` secrets), upload `.tgz` artifact via `actions/upload-artifact@v4`.
- **Depends**: T23
- **Requirements**: R14.2, R14.3, R14.4
- **Properties**: P14
- **Acceptance criteria**:
  - [x] `.github/workflows/release.yml` exists and is valid YAML
  - [x] Uses `actions/checkout@v4`, `actions/upload-artifact@v4`
  - [x] Reads `APP_NAME` from `.env`
  - [x] Uses `splunk-app-action@v5` with Splunkbase API mode (secrets for creds)
  - [x] Triggers on tag push matching `v*`
  - [x] Uploads `.tgz` artifact with app name and version
- **Files**: `.github/workflows/release.yml`
- **Verify**: `cat .github/workflows/release.yml | python3 -c "import sys,yaml; yaml.safe_load(sys.stdin)"` exits 0
- **Status**: [x]

### T25: Remove Disabled Release Workflow `P14`
- **Description**: Delete `.github/workflows/_disabled.release.yml` (outdated: uses deprecated `::set-output`, old action versions, stub publish step, fragile app.conf grep).
- **Depends**: T24
- **Requirements**: R14.5
- **Properties**: P14
- **Acceptance criteria**:
  - [x] `.github/workflows/_disabled.release.yml` no longer exists
  - [x] No references to `_disabled.release.yml` remain in docs (only in spec task descriptions)
- **Files**: `.github/workflows/_disabled.release.yml`
- **Verify**: `test ! -f .github/workflows/_disabled.release.yml`
- **Status**: [x]

## Phase 10: Audit Cleanup

### T26: Template Hygiene + Spec Drift Fixes `P0`
- **Description**: Clean up files that should not be committed in a template repo, fix spec↔disk path drift, and reorder spec sections. Found during `/spec-audit`. (1) Remove `Splunk_App_for_Anomaly_Detection/` from git — concrete app, not part of template. (2) Remove tracked `splunk-config-dev/metadata/local.meta` (Splunk runtime artifact, gitignore rule exists but file was tracked before rule). (3) Remove tracked `splunk-config-dev/default.old.*/` backup dirs. (4) Update `.gitignore` to ignore all apps under `splunk/config/apps/` except `splunk-config-dev/` and `.gitkeep`. (5) Delete stale dirs on disk: `default.old.20260212-182201/`, `.devcontainer/splunk/`. (6) Fix design.md and tasks.md: `tests/e2e/lib/helpers.sh` → `tests/e2e/helpers.sh`. (7) Reorder R12 before R13 in spec.md. (8) Update ARCHITECTURE.md to remove concrete app references.
- **Depends**: T25
- **Requirements**: R11.1, R11.2
- **Acceptance criteria**:
  - [x] `Splunk_App_for_Anomaly_Detection/` not tracked in git
  - [x] `splunk-config-dev/metadata/local.meta` not tracked in git
  - [x] `splunk-config-dev/default.old.*/` dirs not tracked and not on disk
  - [x] `.gitignore` ignores `splunk/config/apps/*` except `splunk-config-dev/` and `.gitkeep`
  - [x] `.devcontainer/splunk/` stale dir deleted from disk
  - [x] design.md and tasks.md reference correct path `tests/e2e/helpers.sh` (no `lib/` subdir)
  - [x] R12 appears before R13 in spec.md
  - [x] ARCHITECTURE.md uses generic `<your_app>/` placeholder instead of concrete app name
  - [x] `git status` is clean after commit (no unintended changes)
- **Files**: `.gitignore`, `ARCHITECTURE.md`, `.windloop/devcontainer-modernize/spec.md`, `.windloop/devcontainer-modernize/design.md`, `.windloop/devcontainer-modernize/tasks.md`
- **Verify**: `git ls-files splunk/config/apps/ | grep -v splunk-config-dev | grep -v .gitkeep` returns empty
- **Status**: [x]
