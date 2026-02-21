#!/bin/bash
# Test suite: Guards — verify guard tasks fail fast with clear messages
# when preconditions are not met. Must run BEFORE boot (no container running).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

log "Guard: __dev:ensure-running (container not running)"

OUTPUT=$(task dev:refresh 2>&1 || true)
if echo "$OUTPUT" | grep -q "ERROR: Dev container is not running"; then
    pass "dev:refresh fails fast when container not running"
else
    fail "dev:refresh should fail with clear error when container not running"
    echo "    Output: ${OUTPUT:0:300}"
fi

log "Guard: __ensure-app-name (no APP_NAME)"

OUTPUT=$(APP_NAME= task app:create 2>&1 || true)
if echo "$OUTPUT" | grep -q "ERROR: APP_NAME is required"; then
    pass "app:create fails fast when APP_NAME empty"
else
    fail "app:create should fail with clear error when APP_NAME empty"
    echo "    Output: ${OUTPUT:0:300}"
fi

log "Guard: __ensure-image (image not built)"

# Only test if image doesn't exist — skip if already built from a previous run
IMAGE=$(docker compose --env-file .env -f .devcontainer/docker-compose.yml images -q splunk 2>/dev/null || true)
if [ -z "$IMAGE" ]; then
    OUTPUT=$(task dev:up 2>&1 || true)
    if echo "$OUTPUT" | grep -q "ERROR: Splunk image not built"; then
        pass "dev:up fails fast when image not built"
    else
        fail "dev:up should fail with clear error when image not built"
        echo "    Output: ${OUTPUT:0:300}"
    fi
else
    echo "  (skipped — image already exists from previous build)"
    pass "ensure-image guard (skipped — image present)"
fi

results
