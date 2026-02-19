#!/bin/bash
# Test suite: Splunk boot — build, start, wait for healthy, verify dev-mounted apps.
# Splunk should NOT be running yet. Works on native Mac or inside devcontainer.

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

log "Verify SPLUNK_PASSWORD auth works"

if docker exec "${SPLUNK_CONTAINER}" bash -lc \
    "curl -sk -o /dev/null -w '%{http_code}' https://localhost:8089/services/apps/local -u admin:\$SPLUNK_PASSWORD" \
    2>/dev/null | grep -q '200'; then
    pass "REST API auth with SPLUNK_PASSWORD"
else
    fail "REST API auth with SPLUNK_PASSWORD — check password in .env"
fi

log "Verify dev-mounted apps recognized by Splunk"

check_app_rest "${SPLUNK_CONTAINER}" "splunk-config-dev" "dev-mounted config app"

results
