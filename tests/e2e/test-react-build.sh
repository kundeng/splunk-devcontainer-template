#!/bin/bash
# Test suite: React scaffold + build-install + REST verify.
# Expects Splunk to be running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Scaffold minimal React test app"

REACT_DIR="packages/${TEST_REACT_APP}"
mkdir -p "${REACT_DIR}/src"

cat > "${REACT_DIR}/package.json" << 'PKGJSON'
{
  "name": "test_react_app",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "build": "mkdir -p dist && echo '<html><body><h1>Test React App</h1></body></html>' > dist/index.html && echo 'Build complete'",
    "start": "echo 'Dev server would start here'"
  }
}
PKGJSON

cat > "${REACT_DIR}/src/index.js" << 'JSEOF'
// Minimal React app entry point for E2E testing
console.log('Test React app loaded');
JSEOF

if [ -f "${REACT_DIR}/package.json" ]; then
    pass "React app scaffolded in packages/${TEST_REACT_APP}/"
else
    fail "React app scaffold failed"
fi

# Create the Splunk app so it can be bind-mounted
if [ ! -d "splunk/config/apps/${TEST_REACT_APP}" ]; then
    APP_NAME="${TEST_REACT_APP}" task app:create 2>/dev/null || true
fi

# Verify React app symlink was created
if docker exec "${SPLUNK_CONTAINER}" test -L "/opt/splunk/etc/apps/${TEST_REACT_APP}"; then
    pass "symlink exists: /opt/splunk/etc/apps/${TEST_REACT_APP}"
else
    fail "symlink missing: /opt/splunk/etc/apps/${TEST_REACT_APP}"
fi

# Run react:build-install
if APP_NAME="${TEST_REACT_APP}" REACT_PATH="${REACT_DIR}" \
    task react:build-install 2>&1 | tail -5; then
    pass "task react:build-install"
    if [ -d "splunk/config/apps/${TEST_REACT_APP}/appserver/static" ]; then
        pass "React build output deployed to appserver/static/"
    else
        fail "React build output not found in appserver/static/"
    fi
else
    fail "task react:build-install"
fi

log "Verify React app via Splunk REST API"

check_app_rest "${SPLUNK_CONTAINER}" "${TEST_REACT_APP}" "symlinked React app"

results
