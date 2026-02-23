#!/bin/bash
# Test suite: React build + package workflow.
# If `expect` is available: exercises the real react:create wizard via expect,
# then tests react:link, react:build, react:package, tgz validation, and idempotency.
# If `expect` is not available: falls back to a fake scaffold (same as before).
# Expects Splunk to be running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

USE_EXPECT=false

if command -v expect &>/dev/null; then
    USE_EXPECT=true
fi

# ── Scaffold: expect (real wizard) or fake ───────────────────────────

if [ "$USE_EXPECT" = true ]; then
    log "Scaffold React app via expect (real react:create wizard)"

    # Clean any previous scaffold
    rm -rf react/packages react/package.json

    if expect "${SCRIPT_DIR}/expect/react-create.exp" "${TEST_REACT_APP}" 2>&1 | tail -20; then
        pass "react:create wizard completed (via expect)"
    else
        fail "react:create via expect failed"
    fi

    # Detect the actual app directory created by @splunk/create
    # The wizard creates the app directly in react/packages/ (not nested)
    REACT_APP_DIR=$(find react/packages -maxdepth 1 -type d -name "*test*" 2>/dev/null | grep -v "^react/packages$" | head -1)
    if [ -n "$REACT_APP_DIR" ]; then
        pass "react:create scaffolded app in ${REACT_APP_DIR}"
    else
        fail "react:create did not create app directory"
    fi

    # Verify APP_NAME was synced to .env
    DETECTED_APP_NAME=$(grep "^APP_NAME=" .env 2>/dev/null | cut -d= -f2)
    if [ -n "$DETECTED_APP_NAME" ]; then
        pass "react:create synced APP_NAME=${DETECTED_APP_NAME} in .env"
        # Use the detected app name for subsequent tests
        TEST_REACT_APP="$DETECTED_APP_NAME"
    else
        fail "react:create did not sync APP_NAME in .env"
    fi

    # Verify initial build created stage/
    STAGE_DIR="${REACT_APP_DIR}/stage"
    if [ -d "${STAGE_DIR}" ]; then
        pass "react:create initial build created stage/"
    else
        fail "react:create initial build did not create stage/"
    fi

    # ── react:link ───────────────────────────────────────────────────
    log "Test react:link"

    if task react:link APP_NAME="${TEST_REACT_APP}" 2>&1 | tail -5; then
        pass "react:link completed"
    else
        fail "react:link failed"
    fi

    # Verify symlink exists in dev container
    if docker exec "${SPLUNK_CONTAINER}" test -d "/opt/splunk/etc/apps/${TEST_REACT_APP}" 2>/dev/null; then
        pass "react:link symlink exists in dev container"
    else
        fail "react:link symlink missing in dev container"
    fi

else
    log "WARNING: expect not available — using fake scaffold (install expect for full react:create test)"

    log "Scaffold minimal React test app in react/packages/"

    REACT_APP_DIR="react/packages/${TEST_REACT_APP}"
    STAGE_DIR="${REACT_APP_DIR}/stage"

    mkdir -p "${REACT_APP_DIR}/src/main/webapp/pages"

    # Root package.json (monorepo)
    cat > "react/package.json" <<'EOF'
{ "private": true, "workspaces": ["packages/*"] }
EOF

    # App package.json with build script that creates a stage/ directory
    cat > "${REACT_APP_DIR}/package.json" <<EOF
{
  "name": "@splunk/${TEST_REACT_APP}",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "build": "mkdir -p stage/default stage/appserver/static/pages stage/appserver/templates && echo '[package]' > stage/default/app.conf && echo 'id = ${TEST_REACT_APP}' >> stage/default/app.conf && echo 'console.log(1)' > stage/appserver/static/pages/TestPage.js && echo 'Build complete'"
  }
}
EOF

    # Discriminator: src/main/webapp/pages/ marks this as a Splunk app (not a component lib)
    touch "${REACT_APP_DIR}/src/main/webapp/pages/.gitkeep"

    if [ -f "${REACT_APP_DIR}/package.json" ]; then
        pass "React app scaffolded in react/packages/${TEST_REACT_APP}/ (fake)"
    else
        fail "React app scaffold failed"
    fi
fi

# ── react:build ──────────────────────────────────────────────────────

log "Test react:build"

# Ensure no stale stage/ (only if STAGE_DIR is set)
if [ -n "${STAGE_DIR:-}" ]; then
    rm -rf "${STAGE_DIR}"
fi

if task react:build APP_NAME="${TEST_REACT_APP}" 2>&1 | tail -3; then
    if [ -d "${STAGE_DIR}" ]; then
        pass "react:build created stage/"
    else
        fail "react:build did not create stage/"
    fi
else
    fail "task react:build exited non-zero"
fi

# ── react:package ────────────────────────────────────────────────────

log "Test react:package"

TGZ="splunk/stage/${TEST_REACT_APP}.tgz"
rm -f "${TGZ}"

if task react:package APP_NAME="${TEST_REACT_APP}" 2>&1 | tail -3; then
    if [ -f "${TGZ}" ]; then
        pass "react:package created ${TGZ}"
    else
        fail "react:package did not create tgz"
    fi
else
    fail "task react:package exited non-zero"
fi

# Validate tgz contents
if tar -tzf "${TGZ}" | grep -q "^${TEST_REACT_APP}/default/app.conf"; then
    pass "tgz contains ${TEST_REACT_APP}/default/app.conf"
else
    fail "tgz missing expected app.conf"
fi

if tar -tzf "${TGZ}" | grep -q "^${TEST_REACT_APP}/appserver/static/pages/"; then
    pass "tgz contains appserver/static/pages/"
else
    fail "tgz missing appserver/static/pages/"
fi

# ── Idempotency: run react:package again ─────────────────────────────

log "Test react:package idempotency"

if task react:package APP_NAME="${TEST_REACT_APP}" 2>&1 | tail -3; then
    pass "react:package second run succeeded"
else
    fail "react:package second run failed"
fi

results
