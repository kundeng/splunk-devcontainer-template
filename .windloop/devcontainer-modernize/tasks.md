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
- **Requirements**: R2.1, R2.2, R2.3, R2.4, R2.5
- **Properties**: P1, P2, P3
- **Acceptance criteria**:
  - [ ] `splunk/entrypoint-wrapper.sh` exists and is executable
  - [ ] `splunk/Dockerfile` is minimal: FROM official image, COPY wrapper, override ENTRYPOINT
  - [ ] First start runs full Ansible provisioning and writes marker
  - [ ] Subsequent starts skip Ansible and start splunkd directly
  - [ ] `FORCE_PROVISION=true` forces full re-provisioning
  - [ ] SIGTERM is trapped and splunkd stops gracefully
- **Files**: `splunk/entrypoint-wrapper.sh`, `splunk/Dockerfile`
- **Verify**: `shellcheck splunk/entrypoint-wrapper.sh`
- **Status**: [x]

### T2: Modernize devcontainer.json `P0`
- **Description**: Rewrite `.devcontainer/devcontainer.json` with modern schema. Ubuntu 24.04 base, Node 22, Python 3.12, docker-outside-of-docker, go-task features. Modern `customizations.vscode.*` format. Remove deprecated settings. Clean extension list.
- **Requirements**: R1.1, R1.3, NF4
- **Properties**: P6
- **Acceptance criteria**:
  - [ ] Uses `mcr.microsoft.com/devcontainers/base:ubuntu-24.04`
  - [ ] Node 22, Python 3.12 features
  - [ ] `customizations.vscode.extensions` and `customizations.vscode.settings` (not legacy top-level)
  - [ ] No deprecated `pwa-chrome`, `python.linting.*`, `python.formatting.*` settings
  - [ ] Ports: 8000, 8089, 8088, 3000
  - [ ] No `shutdownAction: stopCompose`
- **Files**: `.devcontainer/devcontainer.json`
- **Verify**: JSON syntax valid
- **Status**: [x]

### T3: Modernize docker-compose.yml `P0`
- **Description**: Rewrite `.devcontainer/docker-compose.yml` to use the custom Splunk image (built from `splunk/Dockerfile`). Add `SPLUNK_GENERAL_TERMS` env var (required by newer Splunk images). Include commented-out sidecar templates (UF, Cribl, PostgreSQL). Mount `splunk/stage` at `/tmp/apps`. Use `splunk-var` named volume only. Include `-f docker-compose.apps.yml` support for dynamic app mounts.
- **Depends**: T1
- **Requirements**: R3.1, R3.2, R3.4, R3.5
- **Properties**: P7
- **Acceptance criteria**:
  - [ ] Uses `build: context: ../splunk` (custom image)
  - [ ] `SPLUNK_GENERAL_TERMS` env var included
  - [ ] `splunk-var` named volume for persistence
  - [ ] `splunk/stage:/tmp/apps` bind mount
  - [ ] Commented-out sidecar service templates
  - [ ] `docker compose config` validates successfully
- **Files**: `.devcontainer/docker-compose.yml`
- **Verify**: `docker compose -f .devcontainer/docker-compose.yml config`
- **Status**: [x]

### T4: Minimal post-create.sh `P0`
- **Description**: Rewrite `.devcontainer/post-create.sh` to be minimal. Install only: splunk-appinspect, ruff, pytest. Set up PATH. Create `.env` from example if missing. Create directory structure. Remove: yo, generator-splunk-app, splunk-packaging-toolkit, global @splunk/create, isort, black, flake8 (replaced by ruff).
- **Requirements**: R1.4
- **Acceptance criteria**:
  - [ ] Installs only: splunk-appinspect, ruff, pytest
  - [ ] No yo, slim, global @splunk/create
  - [ ] Creates `.env` from `splunk.env.example` if missing
  - [ ] Creates `splunk/config/apps/`, `splunk/stage/`, `packages/` dirs
  - [ ] Completes in <30s
- **Files**: `.devcontainer/post-create.sh`
- **Verify**: `shellcheck .devcontainer/post-create.sh`
- **Status**: [x]

## Phase 2: Multi-App & Dependencies

### T5: App Mount Generator `P0`
- **Description**: Add a `app:generate-mounts` task to Taskfile that scans `splunk/config/apps/` and generates `.devcontainer/docker-compose.apps.yml` with bind mount entries for each app directory. Update `splunk:up` to run this before `docker compose up`. Update compose command to include `-f docker-compose.apps.yml`.
- **Depends**: T3
- **Requirements**: R4.5, R3.3
- **Properties**: P4, P7
- **Acceptance criteria**:
  - [ ] `task app:generate-mounts` creates valid `docker-compose.apps.yml`
  - [ ] Each directory in `splunk/config/apps/` gets a bind mount entry
  - [ ] `splunk:up` calls `app:generate-mounts` first
  - [ ] Empty `splunk/config/apps/` produces valid empty override
- **Files**: `Taskfile.yml`, `.devcontainer/docker-compose.apps.yml` (generated, gitignored)
- **Verify**: Create test app dirs, run task, validate YAML
- **Status**: [x]

### T6: Dependency Manager (deps.yml + task) `P1`
- **Description**: Create `splunk/config/deps.yml` schema and `deps:install` task. The task reads the YAML, checks if each app is installed at the correct version, downloads missing tarballs from Splunkbase API or direct URL, and installs them idempotently.
- **Depends**: T3
- **Requirements**: R5.1, R5.2, R5.3, R5.4
- **Properties**: P5
- **Acceptance criteria**:
  - [ ] `splunk/config/deps.yml` exists with example entries
  - [ ] `task deps:install` downloads and installs missing deps
  - [ ] Already-installed deps are skipped
  - [ ] Supports both Splunkbase IDs and direct URLs
  - [ ] Clear error messages on auth/network failures
- **Files**: `Taskfile.yml`, `splunk/config/deps.yml`
- **Verify**: Run `task deps:install` twice, verify idempotent
- **Status**: [x]

## Phase 3: Taskfile Modernization

### T7: Rewrite Taskfile.yml `P0`
- **Description**: Modernize the entire Taskfile. Add missing `react:start` task. Add `splunk:reprovision` task. Remove legacy aliases (`splunk:start`, `splunk:stop`). Clean up help output. Update compose commands to include apps override file. Fix `PACKAGES_DIR` and `REACT_PACKAGES_DIR` paths. Integrate `app:generate-mounts` into `splunk:up`.
- **Depends**: T1, T3, T5, T6
- **Requirements**: R7.1, R7.2, R7.3, R7.4, R7.5, R6.1, R6.2, R6.3, R4.2, R4.3, R4.4
- **Acceptance criteria**:
  - [ ] `react:start` task exists and works
  - [ ] `splunk:reprovision` task exists
  - [ ] No legacy aliases
  - [ ] `splunk:up` generates mounts first
  - [ ] All compose commands use both compose files
  - [ ] `deps:install` task integrated
  - [ ] Help output is clean and accurate
- **Files**: `Taskfile.yml`
- **Verify**: `task --list` shows all tasks
- **Status**: [x]

## Phase 4: Documentation & Cleanup

### T8: Rewrite AGENTS.md `P1`
- **Description**: Rewrite `AGENTS.md` as general repository information for AI agents. Move project requirements and design details to `.windloop/` (already done via spec). AGENTS.md should describe: what the repo is, directory structure, how to use it (commands), architecture overview, and key patterns. Add windloop snippet.
- **Depends**: T7
- **Requirements**: (documentation)
- **Acceptance criteria**:
  - [ ] AGENTS.md is concise general repo info
  - [ ] No project requirements or design specs in AGENTS.md
  - [ ] Includes windloop snippet
  - [ ] Accurately reflects the modernized repo structure
- **Files**: `AGENTS.md`
- **Status**: [x]

### T9: Update splunk.env.example and .gitignore `P1`
- **Description**: Update `splunk.env.example` with current Splunk version (9.4.0), add `SPLUNK_GENERAL_TERMS`, clean up comments. Update `.gitignore` to include `docker-compose.apps.yml` (generated file). Remove `custom_setup.sh` (dead code).
- **Depends**: T3
- **Requirements**: (cleanup)
- **Acceptance criteria**:
  - [ ] `splunk.env.example` has `SPLUNK_VERSION=9.4.0` and `SPLUNK_GENERAL_TERMS`
  - [ ] `.gitignore` includes `docker-compose.apps.yml`
  - [ ] `custom_setup.sh` deleted
- **Files**: `splunk.env.example`, `.gitignore`, `.devcontainer/custom_setup.sh` (delete)
- **Status**: [x]

### T10: Update launch.json `P2`
- **Description**: Fix `.vscode/launch.json` to use modern `chrome` debug type instead of deprecated `pwa-chrome`. Update `webRoot` paths to match new directory structure.
- **Requirements**: R1.3
- **Acceptance criteria**:
  - [ ] Debug type is `chrome` not `pwa-chrome`
  - [ ] Paths match modernized structure
- **Files**: `.vscode/launch.json`
- **Status**: [x]

### T11: Update README.md `P2`*
- **Description**: Update README to reflect modernized architecture, commands, and workflow. Optional — can be deferred.
- **Depends**: T7, T8
- **Acceptance criteria**:
  - [ ]* README reflects current architecture
- **Files**: `README.md`
- **Status**: [x]

## Phase 5: Integration Testing

### T12: Devcontainer Build + Compose Validation `P0`
- **Description**: Verify the full devcontainer builds successfully using the devcontainer CLI. Validate docker-compose config parses. Run shellcheck on shell scripts. Confirm Taskfile parses with `task --list`. Run `task app:generate-mounts` and verify output.
- **Depends**: T1, T2, T3, T4, T5, T6, T7
- **Requirements**: (verification of all implementation tasks)
- **Acceptance criteria**:
  - [ ] `devcontainer build --workspace-folder .` succeeds
  - [ ] `docker compose -f .devcontainer/docker-compose.yml config` validates
  - [ ] `shellcheck splunk/entrypoint-wrapper.sh` passes (or only minor warnings)
  - [ ] `shellcheck .devcontainer/post-create.sh` passes
  - [ ] `task --list` shows all expected tasks
  - [ ] `task app:generate-mounts` produces valid docker-compose.apps.yml
  - [ ] JSON syntax of devcontainer.json and launch.json is valid
- **Verify**: Run all checks above
- **Status**: [x]
