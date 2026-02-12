#!/bin/bash
# Test suite: Skip-provision timing and reprovision.
# Validates entrypoint-wrapper marker logic end-to-end.
# Expects Splunk to be running and healthy (marker already exists).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

# ── Skip-provision: restart should be fast ────────────────────────────

log "Skip-provision: restart timing"

if docker exec "${SPLUNK_CONTAINER}" test -f /opt/splunk/var/.provisioned; then
    pass "Provisioning marker exists before restart"
else
    fail "Provisioning marker missing before restart"
    results; exit 1
fi

RESTART_START=$(date +%s)
task splunk:restart 2>&1 | tail -3 || true

SKIP_HEALTHY=false
for i in $(seq 1 12); do
    if docker exec "${SPLUNK_CONTAINER}" bash -lc \
        "curl -sk -o /dev/null -w '%{http_code}' https://localhost:8089/services/server/health -u admin:${SPLUNK_PASSWORD}" 2>/dev/null | grep -q "200"; then
        SKIP_HEALTHY=true
        break
    fi
    sleep 5
    echo "  skip-provision restart ... $((i * 5))s"
done

RESTART_END=$(date +%s)
RESTART_ELAPSED=$((RESTART_END - RESTART_START))

if [ "$SKIP_HEALTHY" = "true" ]; then
    if [ "$RESTART_ELAPSED" -le 60 ]; then
        pass "Skip-provision restart: healthy in ${RESTART_ELAPSED}s (≤60s)"
    else
        fail "Skip-provision restart too slow: ${RESTART_ELAPSED}s (expected ≤60s)"
    fi
else
    fail "Splunk not healthy after skip-provision restart (waited 60s)"
fi

if docker exec "${SPLUNK_CONTAINER}" test -f /opt/splunk/var/.provisioned; then
    pass "Marker preserved after skip-provision restart"
else
    fail "Marker lost after skip-provision restart"
fi

check_app_rest "${SPLUNK_CONTAINER}" "splunk-config-dev" "config app (after skip-provision restart)"

# ── Reprovision: remove marker, full Ansible ──────────────────────────

log "Reprovision: full Ansible run"

REPROV_START=$(date +%s)
task splunk:reprovision 2>&1 | tail -5 || true

REPROV_HEALTHY=false
for i in $(seq 1 60); do
    if docker exec "${SPLUNK_CONTAINER}" bash -lc \
        "curl -sk -o /dev/null -w '%{http_code}' https://localhost:8089/services/server/health -u admin:${SPLUNK_PASSWORD}" 2>/dev/null | grep -q "200"; then
        REPROV_HEALTHY=true
        break
    fi
    sleep 10
    echo "  reprovision ... $((i * 10))s"
done

REPROV_END=$(date +%s)
REPROV_ELAPSED=$((REPROV_END - REPROV_START))

if [ "$REPROV_HEALTHY" = "true" ]; then
    pass "Reprovision: Splunk healthy after full Ansible (${REPROV_ELAPSED}s)"
else
    fail "Splunk not healthy after reprovision (waited 600s)"
    task splunk:logs 2>&1 | tail -30 || true
fi

if docker exec "${SPLUNK_CONTAINER}" test -f /opt/splunk/var/.provisioned; then
    pass "Marker re-created after reprovision"
else
    fail "Marker not re-created after reprovision"
fi

check_app_rest "${SPLUNK_CONTAINER}" "splunk-config-dev" "config app (after reprovision)"

results
