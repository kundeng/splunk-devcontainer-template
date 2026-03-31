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
task dev:down 2>/dev/null || true
task stage:clean 2>/dev/null || true
rm -rf splunk/config/apps/test_cmd_app splunk/config/apps/test-react-app splunk/config/apps/testReactApp
rm -rf react/packages react/package.json react/node_modules react/yarn.lock
rm -rf ucc/test_ucc_addon ucc/output/*
rm -f splunk/stage/*.tgz splunk/stage/*.tar.gz
rm -rf ".task"
# Clear APP_NAME from .env so guards test works cleanly
if [ -f .env ]; then
    sed -i.bak '/^APP_NAME=/d' .env && rm -f .env.bak
fi

cleanup_all() {
    log "Final cleanup"
    task dev:down 2>/dev/null || true
    task stage:clean 2>/dev/null || true
    rm -rf splunk/config/apps/test_cmd_app splunk/config/apps/test-react-app splunk/config/apps/testReactApp
    rm -rf react/packages react/package.json react/node_modules react/yarn.lock
    rm -rf ucc/test_ucc_addon ucc/output/*
    rm -f splunk/stage/*.tgz splunk/stage/*.tar.gz
    rm -rf ".task"
    # Clear APP_NAME from .env so next run starts clean
    if [ -f .env ]; then
        sed -i.bak '/^APP_NAME=/d' .env && rm -f .env.bak
    fi
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

run_suite "guards"         "${SCRIPT_DIR}/test-guards.sh"
run_suite "boot"           "${SCRIPT_DIR}/test-boot.sh"
run_suite "app-lifecycle"  "${SCRIPT_DIR}/test-app-lifecycle.sh"
run_suite "deps-install"   "${SCRIPT_DIR}/test-deps-install.sh"
run_suite "react-build"    "${SCRIPT_DIR}/test-react-build.sh"
run_suite "ucc-lifecycle"  "${SCRIPT_DIR}/test-ucc-lifecycle.sh"
run_suite "staging"        "${SCRIPT_DIR}/test-staging.sh"
run_suite "skip-provision" "${SCRIPT_DIR}/test-skip-provision.sh"

# ── Summary ──────────────────────────────────────────────────────────

echo ""
log "Lifecycle Results: ${SUITES_PASS}/${SUITES_RUN} suites passed, ${SUITES_FAIL} failed"

if [ "$SUITES_FAIL" -gt 0 ]; then
    exit 1
fi

echo "All lifecycle suites passed."
