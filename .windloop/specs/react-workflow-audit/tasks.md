# Tasks: React Workflow Audit

## Overview

Update all documentation, specs, and E2E tests to match the restructured React tasks (react:build + react:package replacing react:build-install, proper dependency graph with __react:resolve and __react:ensure-build).

## Tasks

- [ ] 1. Spec Updates
  - [ ] 1.1 Update parent spec requirements.md — Requirement 6 (React UI)
    - Replace react:build-install references, fix HMR/port 3000 description, add react:link/react:add-page
    - **Depends**: —
    - **Requirements**: 1.1, 1.2

  - [ ] 1.2 Update parent spec design.md — Module 6 + Module 8
    - Update react:* namespace list, fix debugging description
    - **Depends**: —
    - **Requirements**: 1.2, 1.3

  - [ ] 1.3 Update parent spec tasks.md — Task 5.1
    - Update task name references
    - **Depends**: —
    - **Requirements**: 1.4

- [ ] 2. Documentation Updates
  - [ ] 2.1 Update ARCHITECTURE.md — all react:build-install references
    - Commands section, React dev loop, "Which one do I need?" table, Debugging section, Daily Dev Loop, port 3000
    - **Depends**: —
    - **Requirements**: 2.1, 2.2, 2.3, 2.4, 2.5

- [ ] 3. E2E Test Updates
  - [ ] 3.1 Rewrite test-react-build.sh for new workflow
    - Use react/packages/, test react:build + react:package, validate tgz contents
    - **Depends**: —
    - **Requirements**: 3.1, 3.2, 3.3

  - [ ] 3.2 Fix run-lifecycle.sh cleanup paths
    - packages/ → react/packages/
    - **Depends**: —
    - **Requirements**: 3.4

- [ ] 4. Code Simplification
  - [ ] 4.1 Extract __app:resolve internal helper (mirror __react:resolve)
    - Writes APP_NAME to .task/app-env; app:create, app:package, app:provision use deps + source
    - **Depends**: —
    - **Requirements**: 7.1

  - [ ] 4.2 Fix app:create heredoc — remove sed dedent pattern
    - Use unindented heredoc so no post-processing needed
    - **Depends**: 4.1
    - **Requirements**: 7.2

  - [ ] 4.3 Simplify app:provision — single loop, no IFS juggling
    - Replace OLD_IFS/IFS pattern with tr + while read
    - **Depends**: 4.1
    - **Requirements**: 7.3

- [ ] 5. Task Description Enrichment
  - [ ] 5.1 Improve desc fields across all tasks
    - Add idempotency hints, interactive flags, dependency chain info
    - **Depends**: 4.1, 4.2, 4.3
    - **Requirements**: 5.1, 5.2, 5.3

- [ ] 6. Idempotency Tests
  - [ ] 6.1 Add idempotency checks to test-app-lifecycle.sh
    - app:create twice (second shows "skipping"), app:sync-links twice (second shows "0 created")
    - **Depends**: 3.1
    - **Requirements**: 6.1, 6.3

  - [ ] 6.2 Add idempotency check to test-react-build.sh
    - react:package twice, both exit 0
    - **Depends**: 3.1
    - **Requirements**: 6.2

- [ ] 7. Verification
  - [ ] 7.1 Grep for stale references
    - Zero hits for react:build-install across all .md, .sh, .yml files
    - **Depends**: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2
    - **Requirements**: 1, 2, 3

  - [ ] 7.2 Run task test:native on Mac
    - All lifecycle suites pass including rewritten react-build and idempotency checks
    - **Depends**: 3.1, 3.2, 6.1, 6.2
    - **Requirements**: 3, 4, 6

## Notes

- The Taskfile.yml React tasks are already correct — this audit adds code cleanup for app:* tasks, enriches descriptions, updates docs/specs/tests.
- The fresh template extract at /tmp/splunk-template-test can be used for isolated testing.
- app:sync-links inline docker exec script is long but well-structured — leave as-is per design decision 5d.
