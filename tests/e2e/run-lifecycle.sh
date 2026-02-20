#!/bin/bash
# Runner for lifecycle test suites.
# Starts Splunk once, runs each focused test suite, reports per-suite results.
#
# Usage: bash tests/e2e/run-lifecycle.sh
# Or called from devcontainer-test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

SUITES_PASS=0
SUITES_FAIL=0
SUITES_RUN=0

# Pre-cleanup (remove leftover artifacts from previous failed runs)
log "Pre-cleanup"
rm -rf "splunk/config/apps/${TEST_CMD_APP}"
rm -rf "splunk/config/apps/${TEST_REACT_APP}"
rm -rf "react/packages/${TEST_REACT_APP}"
rm -f "react/package.json"
rm -f "splunk/stage/${TEST_CMD_APP}.tgz"
rm -f "splunk/stage/${TEST_REACT_APP}.tgz"
rm -rf ".task"

cleanup_all() {
    log "Final cleanup"
    task dev:down 2>/dev/null || true
    task stage:down 2>/dev/null || true
    rm -rf "splunk/config/apps/${TEST_CMD_APP}"
    rm -rf "splunk/config/apps/${TEST_REACT_APP}"
    rm -rf "react/packages/${TEST_REACT_APP}"
    rm -f "react/package.json"
    rm -f "splunk/stage/${TEST_CMD_APP}.tgz"
    rm -f "splunk/stage/${TEST_REACT_APP}.tgz"
    rm -rf ".task"
    echo "  Artifacts cleaned up"
}
trap cleanup_all EXIT

run_suite() {
    local name="$1"
    local script="$2"
    SUITES_RUN=$((SUITES_RUN + 1))

    echo ""
    log "Suite: ${name}"

    if bash "${script}"; then
        echo "  Suite ${name}: PASS"
        SUITES_PASS=$((SUITES_PASS + 1))
    else
        echo "  Suite ${name}: FAIL"
        SUITES_FAIL=$((SUITES_FAIL + 1))
    fi
}

# ── Run suites in order ──────────────────────────────────────────────
# Boot must run first (starts Splunk). Others depend on a running instance.

run_suite "boot"           "${SCRIPT_DIR}/test-boot.sh"
run_suite "app-lifecycle"  "${SCRIPT_DIR}/test-app-lifecycle.sh"
run_suite "deps-install"   "${SCRIPT_DIR}/test-deps-install.sh"
run_suite "react-build"    "${SCRIPT_DIR}/test-react-build.sh"
run_suite "staging"        "${SCRIPT_DIR}/test-staging.sh"
run_suite "skip-provision" "${SCRIPT_DIR}/test-skip-provision.sh"

# ── Summary ──────────────────────────────────────────────────────────

echo ""
log "Lifecycle Results: ${SUITES_PASS}/${SUITES_RUN} suites passed, ${SUITES_FAIL} failed"

if [ "$SUITES_FAIL" -gt 0 ]; then
    exit 1
fi

echo "All lifecycle suites passed."
