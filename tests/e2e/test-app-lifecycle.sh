#!/bin/bash
# Test suite: App lifecycle — create, symlink, package, provision, REST verify.
# Expects Splunk to be running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Create test custom search command app"

if task app:create APP_NAME="${TEST_CMD_APP}" 2>&1 | tail -3; then
    pass "task app:create APP_NAME=${TEST_CMD_APP}"
else
    fail "task app:create APP_NAME=${TEST_CMD_APP}"
fi

# Verify structure
if [ -f "splunk/config/apps/${TEST_CMD_APP}/default/app.conf" ] && \
   [ -d "splunk/config/apps/${TEST_CMD_APP}/bin" ]; then
    pass "app scaffold has correct structure (default/app.conf, bin/)"
else
    fail "app scaffold missing expected structure"
fi

# Add a Python custom search command
cat > "splunk/config/apps/${TEST_CMD_APP}/bin/test_command.py" << 'PYEOF'
#!/usr/bin/env python3
"""Test custom search command — echoes input events with a test field."""
import sys
import csv

def main():
    reader = csv.DictReader(sys.stdin)
    writer = None
    for row in reader:
        row['test_field'] = 'hello_from_test_cmd_app'
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row.keys())
            writer.writeheader()
        writer.writerow(row)

if __name__ == '__main__':
    main()
PYEOF

mkdir -p "splunk/config/apps/${TEST_CMD_APP}/default"
cat > "splunk/config/apps/${TEST_CMD_APP}/default/commands.conf" << 'CMDEOF'
[testcommand]
filename = test_command.py
chunked = false
CMDEOF

if [ -f "splunk/config/apps/${TEST_CMD_APP}/bin/test_command.py" ]; then
    pass "Python custom search command added"
else
    fail "Python custom search command not created"
fi

log "Idempotency: app:create again (should skip)"

OUTPUT=$(task app:create APP_NAME="${TEST_CMD_APP}" 2>&1)
if echo "$OUTPUT" | grep -qi "skipping"; then
    pass "app:create idempotent — second run skipped"
else
    fail "app:create idempotent — expected 'skipping' in output"
    echo "    Output: ${OUTPUT:0:200}"
fi

log "Verify app symlink created"

if docker exec "${SPLUNK_CONTAINER}" test -L "/opt/splunk/etc/apps/${TEST_CMD_APP}"; then
    pass "symlink exists: /opt/splunk/etc/apps/${TEST_CMD_APP}"
else
    fail "symlink missing: /opt/splunk/etc/apps/${TEST_CMD_APP}"
fi

if docker exec "${SPLUNK_CONTAINER}" test -f "/opt/splunk/etc/apps/${TEST_CMD_APP}/default/app.conf"; then
    pass "app.conf visible through symlink"
else
    fail "app.conf not visible through symlink"
fi

log "Idempotency: dev:ensure-links again (should report 0 created)"

OUTPUT=$(task dev:ensure-links 2>&1)
if echo "$OUTPUT" | grep -q "0 created"; then
    pass "dev:ensure-links idempotent — 0 created on second run"
else
    fail "dev:ensure-links idempotent — expected '0 created'"
    echo "    Output: ${OUTPUT:0:200}"
fi

log "Package test app"

if task app:package APP_NAME="${TEST_CMD_APP}" 2>&1 | tail -3; then
    if [ -f "splunk/stage/${TEST_CMD_APP}.tgz" ]; then
        pass "task app:package created tarball"
    else
        fail "tarball not found after app:package"
    fi
else
    fail "task app:package"
fi

log "Verify apps visible via symlink (dev model — no CLI install)"

# app:create already called sync-links + dev:refresh; just verify via REST
check_app_rest "${SPLUNK_CONTAINER}" "${TEST_CMD_APP}" "symlinked custom cmd app"
check_app_rest "${SPLUNK_CONTAINER}" "splunk-config-dev" "dev-mounted config app"

results
