# Tasks: Taskfile Dependency Refactor

## Overview

Restructure Taskfile dependency wiring in phases: guard tasks first, then dev wiring, then APP_NAME cleanup, then staging decomposition, then DRY helpers, then E2E verification.

## Tasks

- [x] 1. Guard Tasks + DRY Helpers
  - [x] 1.1 Add __dev:ensure-running, __stage:ensure-running, __stage-compose, __ensure-app-name
    - Add `__dev:ensure-running` internal task (checks dev container running, fails with clear message)
    - Add `__stage:ensure-running` internal task (checks staging container running)
    - Add `__stage-compose` internal helper (mirrors `__dev-compose` for staging)
    - Add `__ensure-app-name` internal task (validates `{{.APP_NAME}}` is set, no file write)
    - **Depends**: —
    - **Requirements**: 1.1, 1.2, 1.3, 4.1, 7.3
    - **Properties**: 1, 2

- [x] 2. Dev Task Dependency Wiring
  - [x] 2.1 Wire dev tasks to use __dev:ensure-running
    - Add `deps: [__dev:ensure-running]` to: `dev:ensure-links`, `dev:refresh`, `dev:restartd`, `dev:reprovision`
    - Add `deps: [__dev:ensure-running]` to `app:sync-links`, remove inline container check
    - Add `__dev:ensure-running` to `react:link` deps (alongside existing `__react:ensure-build`)
    - Add `deps: [__dev:ensure-running]` to `deps:install`, remove Python-side container check
    - `dev:up` stays as sequential `cmds` (container doesn't exist yet when it starts)
    - Verify `dev:up` health timeout still exits gracefully with suggestion message (R5.2 — no change needed, just verify)
    - **Depends**: 1.1
    - **Requirements**: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 5.1, 5.2
    - **Properties**: 1, 2, 4

- [x] 3. APP_NAME Cleanup
  - [x] 3.1 Replace __app:resolve with __ensure-app-name
    - Remove `__app:resolve` task entirely
    - Change `app:create` deps from `[__app:resolve]` to `[__ensure-app-name]`
    - Change `app:package` deps from `[__app:resolve]` to `[__ensure-app-name]`
    - Remove `. .task/app-env` sourcing from `app:create` and `app:package`
    - Replace `${APP_NAME}` (shell var from sourced file) with `{{.APP_NAME}}` (Taskfile var) in both tasks
    - Update `__react:resolve` to use `{{.APP_NAME}}` instead of resolving APP_NAME itself
    - **Depends**: 1.1
    - **Requirements**: 7.1, 7.2, 7.3, 7.4, 7.5
    - **Properties**: 5

- [x] 4. Staging Lifecycle Decomposition
  - [x] 4.1 Restructure staging tasks: stage:package, stage:up (health wait), stage:install, stage:deploy
    - Extract packaging logic from `stage:deploy` into new `stage:package` task
    - Add health wait loop to `stage:up` (like `dev:up`)
    - Create `stage:install` with `deps: [stage:package, stage:up]`, installs tgz via `splunk install app`
    - Rewrite `stage:deploy` to just call `stage:install`
    - Convert `stage:up`, `stage:down`, `stage:clean`, `stage:logs` to use `__stage-compose`
    - **Depends**: 1.1
    - **Requirements**: 3.1, 3.2, 3.3, 3.4, 3.5, 4.2
    - **Properties**: 3, 6

- [x] 5. Update Help Text + Docs
  - [x] 5.1 Update help task, ARCHITECTURE.md, README.md
    - Update `help` task to list `stage:package`, `stage:install`
    - Update ARCHITECTURE.md commands section with new staging tasks
    - Update README.md commands section with new staging tasks
    - Document the dependency chain in ARCHITECTURE.md
    - **Depends**: 4.1
    - **Requirements**: 6.1, NF 2

- [x] 6. E2E Tests
  - [x] 6.1 Update E2E tests for staging restructure
    - Update `test-staging.sh` to exercise `stage:deploy` (full chain)
    - Verify `test-app-lifecycle.sh` works with `__ensure-app-name` (uses `APP_NAME=x` CLI syntax)
    - Run full `task test:native` — all 6 suites must pass
    - **Depends**: 2.1, 3.1, 4.1, 5.1
    - **Requirements**: NF 1
    - **Properties**: 3, 4, 5

## Notes

- `dev:up` intentionally stays as sequential `cmds` — the container doesn't exist when it starts, so it can't dep on `__dev:ensure-running`.
- `app:create` keeps `dev:ensure-links` and `dev:refresh` as sequential `cmds` (post-scaffold steps, not preconditions).
- `__react:resolve` keeps its temp file for `REACT_APP_PATH` (computed value requiring directory scan). Only the APP_NAME portion switches to `{{.APP_NAME}}`.
- The `deps-install.py` container check removal is a Python file edit, not just Taskfile.
