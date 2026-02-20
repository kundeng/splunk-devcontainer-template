# Requirements: React Workflow Audit

## Introduction

Audit and update the Splunk devcontainer template after the React task restructure. The Taskfile was refactored (react:build-install → react:build + react:package, added __react:resolve/__react:ensure-build internals, proper deps), but the spec, docs, E2E tests, and requirements still reference the old task names and workflow model.

Additionally: enrich task descriptions, add idempotency tests, and simplify convoluted shell code in the Taskfile.

## Glossary

- **stage/**: Webpack output directory inside `react/packages/<app>/` — a complete Splunk app
- **react:link**: Symlinks stage/ into dev Splunk's etc/apps/ for live dev
- **react:build**: Production webpack build (always rebuilds stage/)
- **react:package**: react:build + tar stage/ into splunk/stage/<APP_NAME>.tgz
- **__react:resolve**: Internal task that resolves APP_NAME + REACT_PATH → .task/react-env
- **__react:ensure-build**: Internal task that builds stage/ only if missing

## Requirements

### Requirement 1: Spec Consistency

**User Story:** As a maintainer, I want the spec (requirements.md, design.md, tasks.md) to accurately reflect the current Taskfile implementation, so that the spec remains a reliable source of truth.

#### Acceptance Criteria

1. WHEN a developer reads requirements.md, THEN Requirement 6 (React UI) SHALL reference `react:create`, `react:link`, `react:start`, `react:build`, `react:package` — not `react:build-install` or HMR on port 3000.
2. WHEN a developer reads design.md, THEN Module 6 (Taskfile) SHALL list `react:*` as `create`, `add-page`, `link`, `start`, `build`, `package` — not `build-install`.
3. WHEN a developer reads design.md, THEN Module 8 (Debugging) SHALL describe `react:start` as webpack watch (not HMR dev server on :3000).
4. WHEN a developer reads tasks.md, THEN task 5.1 SHALL reference the current task names.

### Requirement 2: Documentation Consistency

**User Story:** As a developer cloning this template, I want ARCHITECTURE.md to accurately describe the React workflow and available commands, so that I can follow the docs without confusion.

#### Acceptance Criteria

1. WHEN a developer reads the React UI commands section, THEN it SHALL list `react:build` and `react:package` — not `react:build-install`.
2. WHEN a developer reads the React dev loop section, THEN it SHALL describe: `react:create` → `react:link` → `react:start`, with `react:package` for staging.
3. WHEN a developer reads the "Which one do I need?" table, THEN `app:provision` SHALL NOT reference `react:build-install`.
4. WHEN a developer reads the Debugging section, THEN `react:start` SHALL be described as webpack watch updating stage/ via symlink — not as a dev server on :3000 with HMR.
5. WHEN a developer reads the Daily Dev Loop table, THEN React production build SHALL reference `react:package` not `react:build-install`.

### Requirement 3: E2E Test Accuracy

**User Story:** As a maintainer, I want E2E tests to validate the current React workflow (react:build + react:package + stage/ symlink model), so that regressions are caught.

#### Acceptance Criteria

1. WHEN `test-react-build.sh` runs, THEN it SHALL test `react:build` and `react:package` — not `react:build-install`.
2. WHEN `test-react-build.sh` runs, THEN it SHALL validate that `react:package` produces a valid tgz in `splunk/stage/`.
3. WHEN `test-react-build.sh` runs, THEN it SHALL use `react/packages/` (not `packages/`) as the React source directory.
4. WHEN `run-lifecycle.sh` cleans up, THEN it SHALL clean `react/` artifacts (not `packages/`).

### Requirement 4: React Workflow Correctness

**User Story:** As a developer, I want react:link to auto-build if stage/ is missing and react:package to depend on react:build, so that tasks are standalone and idempotent.

#### Acceptance Criteria

1. WHEN `task react:link` is run and stage/ does not exist, THEN __react:ensure-build SHALL run yarn build automatically before creating the symlink.
2. WHEN `task react:package` is run, THEN react:build SHALL run first (via deps), then the tgz SHALL be produced.
3. WHEN `task react:build` is run, THEN it SHALL always rebuild stage/ (not skip if exists).

### Requirement 5: Enriched Task Descriptions

**User Story:** As a developer running `task --list`, I want each task description to clearly convey what it does, its key behavior, and any important side effects, so that I can choose the right command without reading the Taskfile source.

#### Acceptance Criteria

1. WHEN a developer runs `task --list`, THEN every task `desc` SHALL be a concise but informative sentence — not just restating the task name.
2. WHEN a task has notable behavior (e.g. health wait, idempotency, interactive prompt), THEN the `desc` SHALL mention it.
3. WHEN a task depends on other tasks, THEN the `desc` SHALL hint at the dependency chain where useful (e.g. "auto-builds stage/ if missing").

### Requirement 6: Idempotency Testing

**User Story:** As a maintainer, I want E2E tests to verify that key tasks are idempotent (running twice produces the same result without errors), so that developers can safely re-run tasks.

#### Acceptance Criteria

1. WHEN `test-app-lifecycle.sh` runs, THEN it SHALL run `app:create` twice and verify the second run succeeds with a "skipping" message.
2. WHEN `test-react-build.sh` runs, THEN it SHALL run `react:package` twice and verify both runs succeed.
3. WHEN `test-app-lifecycle.sh` runs, THEN it SHALL run `app:sync-links` twice and verify the second run reports "0 created".

### Requirement 7: Code Simplification

**User Story:** As a maintainer, I want the Taskfile shell scripts to be clean and DRY, so that they are easy to read, debug, and modify.

#### Acceptance Criteria

1. WHEN APP_NAME resolution is needed, THEN it SHALL use a shared internal helper (`__app:resolve`) — not copy-pasted 3-line blocks in every task.
2. WHEN `app:create` generates config files, THEN it SHALL NOT use the heredoc+sed dedent pattern. It SHALL use a cleaner approach (e.g. unindented heredoc with `<<-EOF` and tabs, or write without indentation).
3. WHEN `app:provision` iterates over apps, THEN it SHALL use a single readable loop — not IFS juggling with two separate loops.
4. WHEN `app:sync-links` runs inline shell in docker exec, THEN the script SHALL be readable and well-structured (acceptable as-is if comments are clear, but reduce if possible).

## Out of Scope

- Changing the @splunk/create scaffolding behavior
- Adding new React tasks beyond what's already implemented
- Devcontainer configuration changes
- CI/CD workflow updates (separate spec)
