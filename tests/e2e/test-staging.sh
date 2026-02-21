#!/bin/bash
# Test suite: Staging — verify packaged apps are auto-installed from splunk/stage/.
# Expects Splunk dev to be running (for packaging). Starts/stops staging itself.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

STAGING_CONTAINER="splunk-staging"

log "Deploy and start staging Splunk"

# stage:deploy packages all apps in splunk/config/apps/ then calls stage:up
if task stage:deploy 2>&1 | tail -10; then
    pass "task stage:deploy"
else
    fail "task stage:deploy"
fi

log "Verify mounted apps are auto-installed"

# Wait for staging Splunk to become healthy
STAGING_HEALTHY=false
for i in $(seq 1 60); do
    if docker exec "${STAGING_CONTAINER}" bash -lc \
        "curl -sk -o /dev/null -w '%{http_code}' https://localhost:8089/services/server/health -u admin:${SPLUNK_PASSWORD}" 2>/dev/null | grep -q "200"; then
        STAGING_HEALTHY=true
        pass "Staging Splunk healthy (after $((i * 10))s)"
        break
    fi
    sleep 10
    echo "  staging ... $((i * 10))s"
done

if [ "$STAGING_HEALTHY" = "true" ]; then
    # Ansible may still be installing apps after Splunk reports healthy.
    # Wait until the first test app appears (up to 120s), then check all.
    STAGING_APPS_READY=false
    for j in $(seq 1 12); do
        PROBE=$(docker exec "${STAGING_CONTAINER}" bash -lc \
            "curl -sk 'https://localhost:8089/services/apps/local/${TEST_CMD_APP}?output_mode=json' -u admin:${SPLUNK_PASSWORD}" \
            2>/dev/null || echo "EXEC_FAILED")
        if echo "$PROBE" | grep -q "\"name\":\"${TEST_CMD_APP}\""; then
            STAGING_APPS_READY=true
            break
        fi
        echo "  waiting for staging apps ... $((j * 10))s"
        sleep 10
    done

    if [ "$STAGING_APPS_READY" = "true" ]; then
        check_app_rest "${STAGING_CONTAINER}" "${TEST_CMD_APP}" "staging: auto-installed custom cmd app"
        check_app_rest "${STAGING_CONTAINER}" "${TEST_REACT_APP}" "staging: auto-installed React app"
    else
        fail "Staging apps not installed after waiting 120s"
    fi
else
    fail "Staging Splunk did not become healthy"
fi

results
