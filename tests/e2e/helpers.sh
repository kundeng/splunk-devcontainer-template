#!/bin/bash
# Shared test helpers for E2E lifecycle tests.
# Source this file from each test script.

SPLUNK_PASSWORD="${SPLUNK_PASSWORD:-admin123}"
SPLUNK_CONTAINER="${SPLUNK_CONTAINER:-splunk-dev}"
STAGING_CONTAINER="${STAGING_CONTAINER:-splunk-staging}"
TEST_CMD_APP="${TEST_CMD_APP:-test_cmd_app}"
TEST_REACT_APP="${TEST_REACT_APP:-test_react_app}"

_PASS=0
_FAIL=0

log()  { echo "=== $1"; }
pass() { echo "  ✓ $1"; _PASS=$((_PASS + 1)); }
fail() { echo "  ✗ $1"; _FAIL=$((_FAIL + 1)); }

results() {
    echo "  Results: ${_PASS} passed, ${_FAIL} failed"
    return "$_FAIL"
}

splunk_exec() {
    docker exec "${SPLUNK_CONTAINER}" bash -lc "$*"
}

# Check if an app is recognized by Splunk via REST API
# Usage: check_app_rest <container> <app_name> <description>
check_app_rest() {
    local container="$1"
    local app_name="$2"
    local desc="$3"
    local response
    response=$(docker exec "${container}" bash -lc \
        "curl -sk 'https://localhost:8089/services/apps/local/${app_name}?output_mode=json' -u admin:${SPLUNK_PASSWORD}" \
        2>/dev/null || echo "EXEC_FAILED")

    if echo "$response" | grep -q "\"name\":\"${app_name}\""; then
        local disabled
        disabled=$(echo "$response" | grep -o '"disabled":[^,}]*' | head -1 | cut -d: -f2)
        if [ "$disabled" = "false" ] || [ "$disabled" = "0" ]; then
            pass "REST API: ${desc} (${app_name}) — enabled"
        else
            pass "REST API: ${desc} (${app_name}) — present (disabled=$disabled)"
        fi
        return 0
    else
        fail "REST API: ${desc} (${app_name}) — not found"
        echo "    Response (first 300 chars): ${response:0:300}"
        return 1
    fi
}

wait_for_splunk() {
    local container="${1:-$SPLUNK_CONTAINER}"
    local max_wait="${2:-600}"
    local interval=10
    local elapsed=0
    echo "  Waiting for Splunk to become healthy (max ${max_wait}s)..."
    while [ $elapsed -lt "$max_wait" ]; do
        if docker exec "${container}" bash -lc \
            "curl -sk -o /dev/null -w '%{http_code}' https://localhost:8089/services/server/health -u admin:${SPLUNK_PASSWORD}" 2>/dev/null | grep -q "200"; then
            echo "  Splunk healthy after ${elapsed}s"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        echo "  ... ${elapsed}s"
    done
    echo "  Splunk did not become healthy within ${max_wait}s"
    return 1
}
