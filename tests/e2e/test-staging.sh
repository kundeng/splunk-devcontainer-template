#!/bin/bash
# Test suite: Staging — verify stage:deploy packages, starts, and installs apps.
# Expects Splunk dev to be running (for packaging). Starts/stops staging itself.
# stage:deploy handles: package → start + health wait → CLI install.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Deploy and start staging Splunk"

# stage:deploy → stage:install → deps: [stage:package, stage:up] → CLI install
# stage:up includes __wait-healthy, so staging is ready when deploy finishes.
if task stage:deploy 2>&1 | tail -20; then
    pass "task stage:deploy"
else
    fail "task stage:deploy"
fi

log "Verify apps installed via CLI"

# stage:install already ran splunk install app for each tgz.
# Give Splunk a moment to register the apps, then verify via REST.
sleep 5

check_app_rest "${STAGING_CONTAINER}" "${TEST_CMD_APP}" "staging: CLI-installed custom cmd app"

# Use actual React app name from .env (set by react:create during react-build suite)
REACT_APP_NAME=$(grep "^APP_NAME=" .env 2>/dev/null | cut -d= -f2)
if [ -z "$REACT_APP_NAME" ]; then
    REACT_APP_NAME="${TEST_REACT_APP}"
fi
check_app_rest "${STAGING_CONTAINER}" "${REACT_APP_NAME}" "staging: CLI-installed React app"

results
