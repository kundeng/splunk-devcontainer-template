# Requirements: Test Improvements

## Introduction

Revise the devcontainer E2E tests and lifecycle test suite to reflect the dependency refactor changes, improve naming, and add `expect`-based automation for interactive tasks (`react:create`, `app:create` wizard). The current test suite has stale references, a confusing `test:native` name, a fake React scaffold instead of testing the real `react:create` flow, and no coverage of the interactive `@splunk/create` wizard.

## Glossary

- **`expect`**: Unix tool for automating interactive CLI programs by scripting responses to prompts.
- **Lifecycle tests**: Tests that exercise the full Splunk task lifecycle (boot, app, deps, react, staging, skip-provision, guards). Currently called `test:native`.
- **Devcontainer tests**: Tests that build the devcontainer, verify tools, then run lifecycle tests inside it. Currently called `test:e2e`.
- **Static checks**: Non-Splunk checks (tool presence, compose config, env vars) run inside the devcontainer before lifecycle tests.

## Requirements

### Requirement 1: Rename test:native → test:lifecycle

**User Story:** As a developer, I want the test task name to describe what it tests, not where it runs.

#### Acceptance Criteria

1. WHEN `task test:lifecycle` is run, THEN it SHALL execute `run-lifecycle.sh` (same behavior as current `test:native`).
2. WHEN `task test:native` is run, THEN it SHALL still work (alias or error with suggestion).
3. WHEN the help text lists tests, THEN it SHALL show `test:lifecycle` instead of `test:native`.

### Requirement 2: Revise Devcontainer E2E Tests

**User Story:** As a developer, I want the devcontainer E2E tests to reflect the current task structure and verify the dependency refactor works inside the devcontainer.

#### Acceptance Criteria

1. WHEN `devcontainer-test.sh` checks for expected tasks, THEN it SHALL verify the new staging tasks (`stage:package`, `stage:install`, `stage:deploy`) exist.
2. WHEN `devcontainer-test.sh` runs static checks, THEN it SHALL verify `__ensure-image`, `__dev:ensure-running`, and `__ensure-app-name` guards are present (via `task --list-all` or grep).
3. WHEN `devcontainer-test.sh` runs lifecycle tests, THEN it SHALL call `run-lifecycle.sh` which includes the `guards` suite (7 suites total).
4. WHEN any stale task references exist in test scripts (e.g. old `splunk:*` names, `app:provision`), THEN they SHALL be removed.

### Requirement 3: Expect-Based Interactive Test for react:create

**User Story:** As a developer, I want the React E2E test to exercise the real `react:create` flow (which runs `npx @splunk/create` interactively), so that we test the actual wizard rather than a fake scaffold.

#### Acceptance Criteria

1. WHEN `test-react-build.sh` runs, THEN it SHALL use `expect` (or equivalent) to automate the `react:create` interactive wizard, providing app name and selecting "Splunk app" type.
2. WHEN `react:create` completes via expect, THEN the test SHALL verify: React app directory exists, `APP_NAME` was synced to `.env`, `stage/` was created by the initial build.
3. WHEN `expect` is not available, THEN the test SHALL fall back to the current fake scaffold approach with a warning.
4. WHEN the expect-based test runs, THEN it SHALL also test `react:link` and verify the symlink exists in the dev container.

### Requirement 4: Expect-Based Interactive Test for app:create (optional)

**User Story:** As a developer, I want to verify that `app:create` works end-to-end including the guard chain (`__ensure-app-name` → scaffold → `dev:ensure-links` → `dev:refresh`).

#### Acceptance Criteria

1. WHEN `test-app-lifecycle.sh` runs `app:create`, THEN it SHALL continue to use the `APP_NAME=x` CLI syntax (not interactive — `app:create` is not interactive, it takes APP_NAME as an argument).
2. WHEN `app:create` is tested, THEN the test SHALL verify the full guard chain: `__ensure-app-name` validates, scaffold runs, `dev:ensure-links` syncs symlinks, `dev:refresh` reloads configs.
3. WHEN `app:create` is run without APP_NAME, THEN the test SHALL verify the `__ensure-app-name` guard fails with the expected error (already covered by `test-guards.sh`).

### Requirement 5: Revise test:e2e and test:all

**User Story:** As a developer, I want the test task hierarchy to be clear and consistent.

#### Acceptance Criteria

1. WHEN `task test:e2e` is run, THEN it SHALL build the devcontainer and run all tests inside it (static + lifecycle).
2. WHEN `task test:lifecycle` is run, THEN it SHALL run lifecycle tests directly on the host (requires Docker + running Splunk).
3. WHEN `task test:all` is run, THEN it SHALL run `python:lint` + `test:e2e` (unchanged).
4. WHEN the help text lists tests, THEN it SHALL clearly distinguish between `test:lifecycle` (host) and `test:e2e` (devcontainer).

### Requirement 6: Install expect in Devcontainer

**User Story:** As a developer, I want `expect` available inside the devcontainer so that interactive task tests can run in CI.

#### Acceptance Criteria

1. WHEN the devcontainer is built, THEN `expect` SHALL be available on PATH.
2. WHEN `expect` is installed, THEN it SHALL be added via the devcontainer Dockerfile or feature, not as a runtime install in the test script.

### Non-Functional

**NF 1**: All 7 lifecycle test suites SHALL pass after the revisions.

**NF 2**: Devcontainer E2E (`test:e2e`) SHALL pass end-to-end including the revised lifecycle tests.

**NF 3**: The `expect` dependency SHALL be lightweight and not significantly increase devcontainer build time.

## Out of Scope

- Adding new Taskfile tasks (dependency refactor is complete)
- Changing the React or app workflow logic
- CI/CD pipeline setup (tests are local/devcontainer only for now)
- Performance benchmarking of tests
