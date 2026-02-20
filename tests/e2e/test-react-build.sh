#!/bin/bash
# Test suite: React build + package workflow.
# Scaffolds a minimal fake React app (mimics @splunk/create output),
# then tests react:build, react:package, tgz validation, and idempotency.
# Expects Splunk to be running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

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
    pass "React app scaffolded in react/packages/${TEST_REACT_APP}/"
else
    fail "React app scaffold failed"
fi

# ── react:build ──────────────────────────────────────────────────────

log "Test react:build"

# Ensure no stale stage/
rm -rf "${STAGE_DIR}"

if APP_NAME="${TEST_REACT_APP}" task react:build 2>&1 | tail -3; then
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

if APP_NAME="${TEST_REACT_APP}" task react:package 2>&1 | tail -3; then
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

if APP_NAME="${TEST_REACT_APP}" task react:package 2>&1 | tail -3; then
    pass "react:package second run succeeded"
else
    fail "react:package second run failed"
fi

results
