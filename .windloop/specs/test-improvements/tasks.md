# Tasks: Test Improvements

## Overview

Unify test entry points, revise devcontainer static checks, add expect-based react:create automation, install expect in devcontainer. Order: task renaming → static checks → expect infra → react test revision → E2E verification.

## Tasks

- [ ] 1. Unified Test Entry Points
  - [ ] 1.1 Rename test:native → test:lifecycle, test:e2e → test:devcontainer
    - Rename `test:native` to `test:lifecycle` in Taskfile.yml
    - Rename `test:e2e` to `test:devcontainer` in Taskfile.yml
    - Update `test:all` to call `test:devcontainer`
    - Update help text to show `test:lifecycle` and `test:devcontainer`
    - Remove old task names (clean break, no aliases)
    - Update ARCHITECTURE.md and README.md references
    - **Depends**: —
    - **Requirements**: R1.1, R1.2, R1.3, R1.4, R1.5

- [ ] 2. Revise Devcontainer Static Checks
  - [ ] 2.1 Update devcontainer-test.sh for current task structure
    - Add `stage:package`, `stage:install`, `stage:deploy`, `test:lifecycle` to task existence checks
    - Add guard task checks via `task --list-all`: `__ensure-image`, `__dev:ensure-running`, `__stage:ensure-running`, `__ensure-app-name`, `__wait-healthy`
    - Remove stale references (`test:native`, `app:provision`, any `splunk:*` names)
    - Change lifecycle invocation from `bash run-lifecycle.sh` to `task test:lifecycle`
    - **Depends**: 1.1
    - **Requirements**: R2.1, R2.2, R2.3
    - **Properties**: 2

- [ ] 3. Expect Infrastructure
  - [ ] 3.1 Install expect in devcontainer
    - Add `sudo apt-get install -y expect` to `.devcontainer/post-create.sh`
    - Verify expect is on PATH after devcontainer build
    - **Depends**: —
    - **Requirements**: R4.1, R4.2

  - [ ] 3.2 Create expect script for react:create wizard
    - Create `tests/e2e/expect/react-create.exp`
    - Handle 3 prompts: app name (text), page name (text), page type (arrow-key selection)
    - Use loose pattern matching and 60s timeouts per prompt
    - Accept app_name as CLI argument
    - Exit 0 on success, 1 on timeout/failure
    - **Depends**: 3.1
    - **Requirements**: R3.1
    - **Properties**: 4

- [ ] 4. Revise React Build Test
  - [ ] 4.1 Rewrite test-react-build.sh with expect/fallback
    - If `command -v expect`: call `react-create.exp` → verify directory, `.env` sync, `stage/` exists → `react:link` → verify symlink → `react:build` → `react:package` → tgz validation
    - If no expect: warn, use current fake scaffold → `react:build` → `react:package` → tgz validation
    - Keep idempotency check for react:package
    - **Depends**: 3.2
    - **Requirements**: R3.1, R3.2, R3.3, R3.4
    - **Properties**: 3, 4

- [ ] 5. E2E Verification
  - [ ] 5.1 Verify test:lifecycle passes (7 suites)
    - Run `task test:lifecycle` on host
    - All 7 suites must pass: guards, boot, app-lifecycle, deps-install, react-build, staging, skip-provision
    - **Depends**: 1.1, 2.1, 4.1
    - **Requirements**: NF 1
    - **Properties**: 1

  - [ ] 5.2 Verify test:devcontainer passes
    - Run `task test:devcontainer`
    - Static checks + lifecycle tests inside devcontainer must pass
    - **Depends**: 5.1
    - **Requirements**: NF 2
    - **Properties**: 1, 2

## Notes

- `app:create` is NOT interactive (takes `APP_NAME=x` as argument), so no expect needed for it. Already tested in `test-app-lifecycle.sh`.
- The expect script for `react:create` may need adjustment after running the wizard once to confirm exact prompt text and arrow-key behavior for page type selection.
- `test:lifecycle` and `test:devcontainer` run the exact same lifecycle code — the only difference is where the test runner sits and whether static checks run first.
