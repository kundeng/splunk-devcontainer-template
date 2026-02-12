#!/bin/bash
# Test suite: Splunk boot — build, start, wait for healthy, verify dev-mounted apps.
# Expects to run inside a devcontainer. Splunk should NOT be running yet.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Build and start Splunk"

task splunk:build 2>&1 | tail -3 || true

if task splunk:up 2>&1 | tail -5; then
    pass "task splunk:up"
else
    fail "task splunk:up"
    echo "FATAL: splunk:up failed, cannot continue"
    results; exit 1
fi

if wait_for_splunk; then
    pass "Splunk healthy (REST API responds)"
else
    fail "Splunk healthy (REST API responds)"
    echo "FATAL: Splunk not healthy, cannot continue"
    task splunk:logs 2>&1 | tail -30 || true
    results; exit 1
fi

log "Verify dev-mounted apps recognized by Splunk"

check_app_rest "${SPLUNK_CONTAINER}" "splunk-config-dev" "dev-mounted config app"
check_app_rest "${SPLUNK_CONTAINER}" "Splunk_App_for_Anomaly_Detection" "dev-mounted anomaly detection app"

results
