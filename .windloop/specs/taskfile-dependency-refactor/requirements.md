# Requirements: Taskfile Dependency Refactor

## Introduction

Refactor the Taskfile to use proper Taskfile `deps` (declarative dependencies) instead of sequential `cmds` calls for task orchestration. Add missing internal guard tasks (`__dev:ensure-running`, `__stage:ensure-running`) so that tasks which require a running container declare that as a dependency. Restructure the staging lifecycle into composable tasks (`stage:package`, `stage:up`, `stage:install`, `stage:deploy`) with correct dependency chains. Extract a `__stage-compose` DRY helper.

This spec does NOT rename tasks or change user-facing behavior — it restructures the internal wiring so that dependency relationships are explicit and correct. It also eliminates the `.task/app-env` file-based state passing for `APP_NAME`, replacing it with a simple guard task that validates the env var is set.

## Glossary

- **`deps`**: Taskfile keyword — tasks listed here run *before* `cmds`, in parallel if independent. Expresses "I need this to be true before I can run."
- **`cmds`**: Taskfile keyword — sequential steps that *are* the task's work.
- **Guard task**: An internal task that asserts a precondition (e.g. container is running) and fails fast with a clear message if not met.
- **Reverse dependency principle**: If task B requires the output of task A, then B should declare A as a `dep`, not call A inside its `cmds`.

## Requirements

### Requirement 1: Internal Guard Tasks

**User Story:** As a developer, I want tasks that require a running container to fail fast with a clear message if the container isn't running, so that I don't get cryptic Docker errors.

#### Acceptance Criteria

1. WHEN any task that requires the dev container (e.g. `dev:refresh`, `dev:restartd`, `dev:ensure-links`, `deps:install`) is run and the dev container is NOT running, THEN the guard task `__dev:ensure-running` SHALL fail with a message like "Dev container is not running. Start with: task dev:up".
2. WHEN any task that requires the staging container (e.g. `stage:install`) is run and the staging container is NOT running, THEN the guard task `__stage:ensure-running` SHALL fail with a message like "Staging container is not running. Start with: task stage:up".
3. WHEN the container IS running, THEN the guard tasks SHALL succeed silently (no output).

### Requirement 2: Dev Tasks Use Proper Dependencies

**User Story:** As a developer, I want dev tasks to declare their preconditions as `deps`, so that the dependency graph is explicit and Taskfile can parallelize independent preconditions.

#### Acceptance Criteria

1. WHEN `dev:ensure-links` is run, THEN it SHALL declare `deps: [__dev:ensure-running]` and delegate to `app:sync-links` in `cmds`.
2. WHEN `dev:refresh` is run, THEN it SHALL declare `deps: [__dev:ensure-running]`.
3. WHEN `dev:restartd` is run, THEN it SHALL declare `deps: [__dev:ensure-running]`.
4. WHEN `dev:reprovision` is run, THEN it SHALL declare `deps: [__dev:ensure-running]`.
5. WHEN `app:sync-links` is run, THEN it SHALL declare `deps: [__dev:ensure-running]` and remove its inline "is container running?" check.
6. WHEN `react:link` is run, THEN it SHALL declare `deps: [__react:ensure-build, __dev:ensure-running]`.
7. WHEN `deps:install` is run, THEN it SHALL declare `deps: [__dev:ensure-running]` and the Python script's own container check SHALL be removed.

### Requirement 3: Staging Lifecycle Decomposition

**User Story:** As a developer, I want the staging lifecycle broken into composable tasks with correct dependency chains, so that I can run individual steps or the full pipeline.

#### Acceptance Criteria

1. WHEN `stage:package` is run, THEN it SHALL package all app directories in `splunk/config/apps/` into `splunk/stage/<app>.tgz`. No container dependency.
2. WHEN `stage:up` is run, THEN it SHALL start the staging container and wait for it to become healthy (health check loop, like `dev:up`).
3. WHEN `stage:install` is run, THEN it SHALL declare `deps: [stage:package, stage:up]` and install each `.tgz` from `splunk/stage/` into the running staging container via `splunk install app`.
4. WHEN `stage:deploy` is run, THEN it SHALL call `stage:install` (which pulls in `stage:package` and `stage:up` via deps). It is a convenience alias.
5. WHEN `stage:install` runs and a tgz is already installed, THEN it SHALL skip it or overwrite gracefully (idempotent).

### Requirement 4: DRY Staging Compose Helper

**User Story:** As a maintainer, I want staging compose commands to use a shared internal helper (like `__dev-compose` for dev), so that the compose file path and env file aren't repeated in every staging task.

#### Acceptance Criteria

1. WHEN any staging task needs to run `docker compose`, THEN it SHALL use `__stage-compose` with a `COMPOSE_ARGS` variable, mirroring the `__dev-compose` pattern.
2. WHEN `__stage-compose` is defined, THEN `stage:up`, `stage:down`, `stage:clean`, and `stage:logs` SHALL use it instead of inline `docker compose` commands.

### Requirement 5: dev:up Sequencing

**User Story:** As a developer, I want `dev:up` to correctly sequence its steps: start container → wait for health → sync links, without misusing `deps` for things that must happen in order.

#### Acceptance Criteria

1. WHEN `dev:up` is run, THEN it SHALL use `cmds` (not `deps`) for the sequence: compose up → health wait → `dev:ensure-links`, because each step depends on the previous one completing.
2. WHEN `dev:up` health wait times out, THEN it SHALL print a message suggesting `task dev:ensure-links` and exit gracefully (not fail).

### Requirement 6: Backward Compatibility

**User Story:** As a developer, I want all existing user-facing task names and behavior to remain unchanged, so that my muscle memory and scripts still work.

#### Acceptance Criteria

1. WHEN any user-facing task is run, THEN its name, arguments, and observable output SHALL remain the same as before this refactor.
2. WHEN `app:sync-links` is run directly, THEN it SHALL still work (it gains a dep on `__dev:ensure-running` but remains callable).

### Requirement 7: Eliminate .task/app-env State File

**User Story:** As a developer, I want APP_NAME resolution to use environment variables (already loaded via `dotenv`) instead of writing state to disk, so that there's no hidden file-based coupling between tasks.

#### Acceptance Criteria

1. WHEN `APP_NAME` is set in `.env`, THEN all `app:*` tasks SHALL read it directly from the environment (loaded by `dotenv: ['.env']`). No `.task/app-env` file SHALL be written.
2. WHEN `APP_NAME=foo` is passed as a CLI argument, THEN it SHALL override the `.env` value via Taskfile variable precedence.
3. WHEN `APP_NAME` is neither in `.env` nor passed as a CLI argument, THEN `__ensure-app-name` SHALL fail with: "APP_NAME is required (pass APP_NAME=x or set in .env)".
4. WHEN `__app:resolve` and `.task/app-env` are removed, THEN `app:create` and `app:package` SHALL use `deps: [__ensure-app-name]` and reference `{{.APP_NAME}}` directly in their `cmds`.
5. WHEN React tasks need `REACT_APP_PATH` (a computed value), THEN `__react:resolve` MAY continue to use a temp file or inline `sh:` var, since the path requires directory scanning.

### Non-Functional

**NF 1**: All 6 E2E test suites SHALL continue to pass after the refactor.

**NF 2**: No new user-facing tasks are added except `stage:package` and `stage:install`. `stage:deploy` already exists but changes behavior.

## Out of Scope

- Renaming tasks (already done in previous refactor)
- Changing the React workflow
- Modifying the entrypoint wrapper or Dockerfile
- Adding new features beyond dependency restructuring
- Refactoring `__react:resolve` path computation (may use temp file for computed REACT_APP_PATH)
