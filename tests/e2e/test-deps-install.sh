#!/bin/bash
# Test suite: Dependency install — parse deps.yml, install, idempotency.
# Expects Splunk to be running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Test deps:install with splunk/config/deps.yml"

# First run: downloads and installs all deps listed in deps.yml.
DEPS_OUTPUT=$(task deps:install 2>&1) || true
echo "  deps:install output (last 15 lines):"
echo "$DEPS_OUTPUT" | tail -15

# Verify the YAML parser ran (should check each dependency)
if echo "$DEPS_OUTPUT" | grep -q "Checking dependency:"; then
    pass "deps:install parsed deps.yml"
else
    fail "deps:install did not parse deps.yml"
    echo "    Full output: ${DEPS_OUTPUT}"
fi

# Verify at least one dep was downloaded+installed or already present
if echo "$DEPS_OUTPUT" | grep -qi "Installing\|Already installed\|skipping"; then
    pass "deps:install processed dependencies (installed or skipped)"
else
    fail "deps:install did not install or skip any dependencies"
    echo "    Output: ${DEPS_OUTPUT}"
fi

# Verify installed deps via REST API
sleep 5
check_app_rest "${SPLUNK_CONTAINER}" "Splunk_SA_Scientific_Python_linux_x86_64" "deps:install Splunkbase dep (PSC)"

# Second run: should be idempotent (all deps already installed → skip)
DEPS_OUTPUT2=$(task deps:install 2>&1) || true
echo "  deps:install (2nd run, last 10 lines):"
echo "$DEPS_OUTPUT2" | tail -10

if echo "$DEPS_OUTPUT2" | grep -qi "Already installed\|skipping"; then
    pass "deps:install idempotent (skipped already-installed deps)"
else
    fail "deps:install not idempotent on second run"
    echo "    Output: ${DEPS_OUTPUT2}"
fi

results
