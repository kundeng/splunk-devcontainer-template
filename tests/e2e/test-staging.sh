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
check_app_rest "${STAGING_CONTAINER}" "${TEST_REACT_APP}" "staging: CLI-installed React app"

results
