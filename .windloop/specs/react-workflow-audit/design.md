# Design: React Workflow Audit

## Tech Stack

No new tech — this is a documentation/test audit of existing code.

- **Automation**: Go Task (`Taskfile.yml`) — already implemented
- **E2E testing**: bash scripts in `tests/e2e/`
- **Docs**: Markdown (ARCHITECTURE.md, spec files)

## Architecture Overview

No architecture changes. The Taskfile React tasks were already restructured:

```
__react:resolve       → .task/react-env (APP_NAME + REACT_PATH)
__react:ensure-build  → yarn build only if stage/ missing
react:link            → deps: __react:ensure-build → symlink stage/ into Splunk
react:start           → deps: __react:ensure-build → webpack watch
react:build           → deps: __react:resolve → always rebuilds stage/
react:package         → deps: react:build → tgz into splunk/stage/
```

## Scope of Changes

### 1. Parent Spec Updates (`.windloop/specs/devcontainer-template/`)

**requirements.md** — Requirement 6 (React UI):
- Replace `react:build-install` with `react:build` + `react:package`
- Replace "webpack dev server with HMR on port 3000" with "webpack watch updating stage/ via symlink"
- Add `react:link` and `react:add-page` to acceptance criteria

**design.md** — Module 6 (Taskfile):
- Update `react:*` namespace list: `create`, `add-page`, `link`, `start`, `build`, `package`
- Module 8 (Debugging): `react:start` is webpack watch, not HMR dev server

**tasks.md** — Task 5.1:
- Update task name references

### 2. ARCHITECTURE.md Updates

All `react:build-install` references → appropriate replacement:
- Commands section: list `react:build` and `react:package`
- React dev loop: `react:create` → `react:link` → `react:start`, staging via `react:package`
- "Which one do I need?" table: remove `react:build-install` reference from `app:provision`
- Debugging section: `react:start` = webpack watch, not dev server on :3000
- Daily Dev Loop: React production → `react:package`
- Port 3000: clarify it's not used by default (webpack watch writes to stage/, no dev server)

### 3. E2E Test Updates

**test-react-build.sh** — full rewrite:
- Use `react/packages/` not `packages/`
- Test `react:build` (produces stage/) and `react:package` (produces tgz)
- Validate tgz contents (has APP_NAME/ root with expected subdirs)
- Remove old `appserver/static/` copy check

**run-lifecycle.sh** — cleanup paths:
- `packages/` → `react/packages/`

### 4. Task Description Enrichment

Review every `desc:` field in Taskfile.yml. Key improvements:

| Task | Current desc | Improved desc |
|------|-------------|---------------|
| `splunk:up` | "Start dev Splunk, wait for health, then sync app symlinks" | Good — keep |
| `splunk:refresh` | "Reload app configs + static assets without restarting splunkd" | Good — keep |
| `app:create` | "Scaffold a new Splunk app, sync symlinks, and refresh Splunk" | Add "(idempotent — skips if exists)" |
| `app:provision` | "Package and install local app(s) into running Splunk" | Add "via splunk install app CLI (-update 1)" |
| `react:create` | Long desc | Add "(interactive)" |
| `react:link` | Already mentions auto-build | Good — keep |
| `react:start` | Already clear | Good — keep |
| `react:build` | "Production webpack build (always rebuilds stage/)" | Good — keep |
| `react:package` | Already clear | Good — keep |

### 5. Code Simplification

**5a. Extract `__app:resolve`** — mirror `__react:resolve` pattern:
- Writes `APP_NAME` to `.task/app-env`
- `app:create`, `app:package`, `app:provision` source it via `deps: [__app:resolve]`
- Eliminates the 3-line copy-paste in each task

**5b. Fix `app:create` heredoc** — replace `cat << EOF` + `sed dedent` with:
```bash
cat > "$APP_DIR/default/app.conf" <<EOF
[install]
state = enabled
...
EOF
```
No indentation in the heredoc = no sed needed. The `cmds` block in Taskfile allows unindented heredocs inside `- |` blocks.

**5c. Simplify `app:provision` loop** — replace IFS juggling with:
```bash
echo "$APPS" | tr ',' '\n' | while read -r APP; do
  ...
done
```
Single loop that packages + installs each app sequentially.

**5d. `app:sync-links`** — the inline docker exec script is long but well-commented and functional. Leave as-is but ensure comments are clear. It's a single atomic operation that needs to run inside the container.

### 6. Idempotency Test Additions

Add to existing E2E test scripts:

**test-app-lifecycle.sh**:
- After `app:create`, run it again → check output contains "skipping"
- After `app:sync-links`, run it again → check output contains "0 created"

**test-react-build.sh**:
- After `react:package`, run it again → check both runs exit 0

## Testing Strategy

- **Verification**: Run `task test:native` on Mac to execute full lifecycle suite
- **Manual check**: Grep for `react:build-install` across all files — should return 0 hits
- **Manual check**: Grep for `packages/${TEST_REACT_APP}` (without `react/` prefix) — should return 0 hits

## Correctness Properties

### Property 1: No stale references
- **Statement**: After all updates, `grep -r 'react:build-install'` across the repo returns zero matches.
- **Validates**: Requirement 1, 2
- **Test approach**: `grep -r 'react:build-install' . --include='*.md' --include='*.sh' --include='*.yml'`

### Property 2: E2E tests pass
- **Statement**: `task test:native` completes with all suites passing, including the rewritten react-build suite.
- **Validates**: Requirement 3, 4, 6
- **Test approach**: Run `task test:native` on Mac native

### Property 3: DRY APP_NAME resolution
- **Statement**: APP_NAME resolution logic appears exactly once (in `__app:resolve`), not duplicated across `app:create`, `app:package`, `app:provision`.
- **Validates**: Requirement 7.1
- **Test approach**: `grep -c 'grep -E .^APP_NAME=.' Taskfile.yml` returns ≤2 (one in `__app:resolve`, one in `__react:resolve`)

### Property 4: Idempotent tasks
- **Statement**: Running `app:create`, `app:sync-links`, and `react:package` twice each produces no errors on the second run.
- **Validates**: Requirement 6
- **Test approach**: E2E tests run each twice and check exit codes + output
