# Requirements: Test Improvements

## Introduction

Revise the test suite so that the same lifecycle tests run identically on any host (Mac, Linux, or inside the devcontainer). Currently there are two separate entry points (`test:e2e` builds a devcontainer then runs tests inside it; `test:native` runs the same tests directly on the host). This split is unnecessary — the lifecycle tests target Docker containers and should work the same everywhere.

This spec also renames the confusing `test:native` task, adds `expect`-based automation for the interactive `react:create` wizard (replacing the current fake scaffold), and revises the devcontainer static checks to reflect the dependency refactor.

## Glossary

- **`expect`**: Unix tool for automating interactive CLI programs by scripting responses to prompts.
- **Lifecycle tests**: Tests that exercise the full Splunk task lifecycle (guards, boot, app, deps, react, staging, skip-provision). Run via `run-lifecycle.sh`.
- **Devcontainer static checks**: Verify the devcontainer image is correctly configured (tools on PATH, env vars, compose config). Only meaningful when run inside the devcontainer.

## Requirements

### Requirement 1: Unified Test Entry Points

**User Story:** As a developer, I want one set of lifecycle tests that runs the same way regardless of where I invoke them, so there's no confusion about which test command to use.

#### Acceptance Criteria

1. WHEN `task test:lifecycle` is run (on any host with Docker + task), THEN it SHALL execute `run-lifecycle.sh` — the single lifecycle test runner. This replaces `test:native`.
2. WHEN `task test:devcontainer` is run, THEN it SHALL: (a) build the devcontainer, (b) run devcontainer-specific static checks (tools, env vars, compose config), (c) run `task test:lifecycle` inside the devcontainer. The lifecycle tests are the same code path — no separate test logic.
3. WHEN `task test:all` is run, THEN it SHALL run `python:lint` + `test:devcontainer`.
4. WHEN the old `test:native` or `test:e2e` names are used, THEN they SHALL be removed (no aliases — clean break).
5. WHEN the help text lists tests, THEN it SHALL show `test:lifecycle` and `test:devcontainer` with clear descriptions.

### Requirement 2: Revise Devcontainer Static Checks

**User Story:** As a developer, I want the devcontainer static checks to reflect the current task structure after the dependency refactor.

#### Acceptance Criteria

1. WHEN `devcontainer-test.sh` checks for expected tasks, THEN it SHALL verify the new tasks exist: `stage:package`, `stage:install`, `stage:deploy`, `test:lifecycle`.
2. WHEN `devcontainer-test.sh` runs static checks, THEN it SHALL verify guard tasks are present (via `task --list-all`): `__ensure-image`, `__dev:ensure-running`, `__stage:ensure-running`, `__ensure-app-name`, `__wait-healthy`.
3. WHEN any stale task references exist in test scripts (e.g. `test:native`, `app:provision`, old `splunk:*` names), THEN they SHALL be removed.

### Requirement 3: Expect-Based Interactive Test for react:create

**User Story:** As a developer, I want the React E2E test to exercise the real `react:create` flow (which runs `npx @splunk/create` interactively), so that we test the actual wizard rather than a fake scaffold.

#### Acceptance Criteria

1. WHEN `test-react-build.sh` runs, THEN it SHALL use `expect` (or equivalent) to automate the `react:create` interactive wizard, providing app name and selecting the Splunk app type.
2. WHEN `react:create` completes via expect, THEN the test SHALL verify: React app directory exists, `APP_NAME` was synced to `.env`, `stage/` was created by the initial build.
3. WHEN `expect` is not available, THEN the test SHALL fall back to the current fake scaffold approach with a warning.
4. WHEN the expect-based test runs, THEN it SHALL also test `react:link` and verify the symlink exists in the dev container.

### Requirement 4: Install expect in Devcontainer

**User Story:** As a developer, I want `expect` available inside the devcontainer so that interactive task tests can run in CI.

#### Acceptance Criteria

1. WHEN the devcontainer is built, THEN `expect` SHALL be available on PATH.
2. WHEN `expect` is installed, THEN it SHALL be added via the devcontainer Dockerfile or feature, not as a runtime install in the test script.

### Non-Functional

**NF 1**: All 7 lifecycle test suites SHALL pass after the revisions (guards, boot, app-lifecycle, deps-install, react-build, staging, skip-provision).

**NF 2**: `test:devcontainer` SHALL pass end-to-end (static checks + lifecycle tests inside devcontainer).

**NF 3**: The `expect` dependency SHALL be lightweight and not significantly increase devcontainer build time.

## Out of Scope

- Adding new Taskfile workflow tasks (dependency refactor is complete)
- Changing the React or app workflow logic
- CI/CD pipeline setup (tests are local/devcontainer only for now)
- Performance benchmarking of tests
