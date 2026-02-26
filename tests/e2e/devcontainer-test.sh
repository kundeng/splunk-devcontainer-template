#!/bin/bash
# E2E test for the devcontainer — validates build, tools, and automation.
# Uses @devcontainers/cli (installed via npx if not available).
#
# Usage: bash tests/e2e/devcontainer-test.sh
# Or:    task test:devcontainer

set -euo pipefail

WORKSPACE_FOLDER="${1:-.}"
PASS=0
FAIL=0
DEVCONTAINER=""

# ── Helpers ───────────────────────────────────────────────────────────

log()  { echo "=== $1"; }
pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

cleanup() {
    log "Cleanup"
    if [ -n "$DEVCONTAINER" ]; then
        $DEVCONTAINER down --workspace-folder "$WORKSPACE_FOLDER" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Resolve devcontainer CLI
if command -v devcontainer &>/dev/null; then
    DEVCONTAINER="devcontainer"
else
    DEVCONTAINER="npx --yes @devcontainers/cli"
fi

dc_exec() {
    $DEVCONTAINER exec --workspace-folder "$WORKSPACE_FOLDER" "$@"
}

# ── Build ─────────────────────────────────────────────────────────────

log "Build devcontainer"
if $DEVCONTAINER build --workspace-folder "$WORKSPACE_FOLDER" 2>&1 | tail -5; then
    pass "devcontainer build"
else
    fail "devcontainer build"
    echo "FATAL: build failed, cannot continue"
    exit 1
fi

# ── Up ────────────────────────────────────────────────────────────────

log "Start devcontainer"
if $DEVCONTAINER up --workspace-folder "$WORKSPACE_FOLDER" 2>&1 | tail -5; then
    pass "devcontainer up"
else
    fail "devcontainer up"
    echo "FATAL: up failed, cannot continue"
    exit 1
fi

# ── Tool checks ───────────────────────────────────────────────────────

log "Verify tools on PATH"

for tool in node python3 docker task ruff expect; do
    if dc_exec which "$tool" &>/dev/null; then
        VERSION=$(dc_exec "$tool" --version 2>/dev/null | head -1 || echo "ok")
        pass "$tool ($VERSION)"
    else
        fail "$tool not found"
    fi
done

# splunk-appinspect is a pip package — check via pip show (module name != package name)
if dc_exec pip show splunk-appinspect &>/dev/null; then
    pass "splunk-appinspect"
else
    fail "splunk-appinspect not found"
fi

# ── Taskfile ──────────────────────────────────────────────────────────

log "Verify Taskfile"

TASK_LIST=$(dc_exec task --list 2>&1) || true
if echo "$TASK_LIST" | grep -q "Available tasks"; then
    pass "task --list"
else
    fail "task --list"
fi

# Check expected user-facing tasks exist
for ns in dev:up dev:ensure-links app:create react:start deps:install python:lint \
          stage:package stage:install stage:deploy test:lifecycle; do
    if echo "$TASK_LIST" | grep -qF "$ns"; then
        pass "task $ns exists"
    else
        fail "task $ns missing"
    fi
done

# ── Internal guard tasks ─────────────────────────────────────────────

log "Verify internal guard tasks"

TASK_LIST_ALL=$(dc_exec task --list-all 2>&1) || true
for guard in __ensure-image __dev:ensure-running __stage:ensure-running __ensure-app-name __wait-healthy; do
    if echo "$TASK_LIST_ALL" | grep -qF "$guard"; then
        pass "internal task $guard exists"
    else
        fail "internal task $guard missing"
    fi
done

# ── Environment variable checks ──────────────────────────────────────

log "Verify environment variables"

LWF=$(dc_exec bash -c 'echo $LOCAL_WORKSPACE_FOLDER' 2>/dev/null || true)
if [ -n "$LWF" ]; then
    pass "LOCAL_WORKSPACE_FOLDER set ($LWF)"
else
    fail "LOCAL_WORKSPACE_FOLDER not set — runArgs injection may have failed"
fi

# ── Compose config validation ─────────────────────────────────────────

log "Verify docker compose config"

if dc_exec docker compose -f .devcontainer/docker-compose.yml config &>/dev/null; then
    pass "docker compose config (dev)"
else
    fail "docker compose config (dev)"
fi

if dc_exec docker compose -f .devcontainer/docker-compose.staging.yml config &>/dev/null; then
    pass "docker compose config (staging)"
else
    fail "docker compose config (staging)"
fi

# ── Static check summary ──────────────────────────────────────────────

log "Static Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    echo "Static checks failed — skipping lifecycle tests"
    exit 1
fi

# ── Lifecycle tests (same code path as host) ────────────────────────

log "Running lifecycle tests inside devcontainer (this takes several minutes)..."

if dc_exec task test:lifecycle 2>&1; then
    pass "lifecycle tests"
else
    fail "lifecycle tests"
fi

# ── Final summary ─────────────────────────────────────────────────────

log "Total Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

echo "All E2E tests passed."
