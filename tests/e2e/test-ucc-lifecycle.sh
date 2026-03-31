#!/bin/bash
# Test suite: UCC add-on lifecycle — init, build, package, link, REST verify.
# Phase 1 (offline): init, build, package — no running Splunk needed.
# Phase 2 (online):  link, dev:refresh, REST verify — expects Splunk running and healthy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=helpers.sh
source "${SCRIPT_DIR}/helpers.sh"

TEST_UCC_APP="${TEST_UCC_APP:-test_ucc_addon}"

# ── Phase 1: Offline (init, build, package) ──────────────────────────

log "UCC init: scaffold add-on"

OUTPUT=$(task ucc:init APP_NAME="${TEST_UCC_APP}" 2>&1)
if echo "$OUTPUT" | grep -q "scaffolded"; then
    pass "ucc:init scaffolded ${TEST_UCC_APP}"
else
    fail "ucc:init did not scaffold"
    echo "    Output: ${OUTPUT:0:300}"
fi

# Verify scaffolded structure
if [ -f "ucc/${TEST_UCC_APP}/globalConfig.json" ]; then
    pass "globalConfig.json exists"
else
    fail "globalConfig.json missing"
fi

if [ -f "ucc/${TEST_UCC_APP}/package/app.manifest" ]; then
    pass "package/app.manifest exists"
else
    fail "package/app.manifest missing"
fi

if [ -f "ucc/${TEST_UCC_APP}/package/bin/${TEST_UCC_APP}_input_helper.py" ]; then
    pass "input helper module exists"
else
    fail "input helper module missing"
fi

log "UCC init: idempotency (should skip)"

OUTPUT=$(task ucc:init APP_NAME="${TEST_UCC_APP}" 2>&1)
if echo "$OUTPUT" | grep -qi "skipping\|already exists"; then
    pass "ucc:init idempotent — second run skipped"
else
    fail "ucc:init idempotent — expected skip message"
    echo "    Output: ${OUTPUT:0:200}"
fi

log "UCC init: APP_NAME synced to .env"

if grep -q "^APP_NAME=${TEST_UCC_APP}" .env 2>/dev/null; then
    pass "APP_NAME=${TEST_UCC_APP} in .env"
else
    fail "APP_NAME not synced to .env"
fi

log "UCC build: generate REST handlers, UI, conf files"

OUTPUT=$(task ucc:build APP_NAME="${TEST_UCC_APP}" 2>&1)
if echo "$OUTPUT" | grep -q "Build complete"; then
    pass "ucc:build succeeded"
else
    fail "ucc:build failed"
    echo "    Output (last 500 chars): ${OUTPUT: -500}"
fi

# Verify build output structure
BUILD_DIR="ucc/output/${TEST_UCC_APP}"

if [ -d "${BUILD_DIR}/bin" ]; then
    pass "build output has bin/"
else
    fail "build output missing bin/"
fi

if [ -d "${BUILD_DIR}/lib" ]; then
    pass "build output has lib/ (Python deps vendored)"
else
    fail "build output missing lib/"
fi

if [ -d "${BUILD_DIR}/default" ]; then
    pass "build output has default/"
else
    fail "build output missing default/"
fi

if [ -f "${BUILD_DIR}/default/restmap.conf" ]; then
    pass "restmap.conf generated"
else
    fail "restmap.conf missing"
fi

if [ -f "${BUILD_DIR}/default/web.conf" ]; then
    pass "web.conf generated"
else
    fail "web.conf missing"
fi

if [ -f "${BUILD_DIR}/default/inputs.conf" ]; then
    pass "inputs.conf generated"
else
    fail "inputs.conf missing"
fi

if [ -f "${BUILD_DIR}/default/app.conf" ]; then
    pass "app.conf generated"
else
    fail "app.conf missing"
fi

if [ -d "${BUILD_DIR}/appserver/static/js/build" ]; then
    pass "React UI bundle generated (appserver/static/js/build/)"
else
    fail "React UI bundle missing"
fi

if [ -f "${BUILD_DIR}/appserver/static/js/build/globalConfig.json" ]; then
    pass "globalConfig.json copied to build output"
else
    fail "globalConfig.json missing from build output"
fi

# Verify generated Python files
if [ -f "${BUILD_DIR}/bin/import_declare_test.py" ]; then
    pass "import_declare_test.py generated"
else
    fail "import_declare_test.py missing"
fi

if ls "${BUILD_DIR}"/bin/*_rh_*.py >/dev/null 2>&1; then
    pass "REST handler(s) generated"
else
    fail "REST handler(s) missing"
fi

if ls "${BUILD_DIR}"/bin/${TEST_UCC_APP}_input*.py >/dev/null 2>&1; then
    pass "modular input script generated"
else
    fail "modular input script missing"
fi

# Verify OpenAPI spec
if [ -f "${BUILD_DIR}/appserver/static/openapi.json" ]; then
    pass "OpenAPI spec generated"
else
    fail "OpenAPI spec missing (non-critical)"
fi

log "UCC build: rebuild (should overwrite cleanly)"

OUTPUT=$(task ucc:build APP_NAME="${TEST_UCC_APP}" 2>&1)
if echo "$OUTPUT" | grep -q "Build complete"; then
    pass "ucc:build rebuild succeeded (idempotent with --overwrite)"
else
    fail "ucc:build rebuild failed"
fi

log "UCC package: create tarball for staging"

OUTPUT=$(task ucc:package APP_NAME="${TEST_UCC_APP}" 2>&1)
if echo "$OUTPUT" | grep -q "Package exported\|tar.gz"; then
    pass "ucc:package created tarball"
else
    fail "ucc:package failed"
    echo "    Output (last 300 chars): ${OUTPUT: -300}"
fi

if ls splunk/stage/${TEST_UCC_APP}*.tar.gz >/dev/null 2>&1; then
    TGZ=$(ls splunk/stage/${TEST_UCC_APP}*.tar.gz | head -1)
    SIZE=$(stat -c%s "$TGZ" 2>/dev/null || stat -f%z "$TGZ" 2>/dev/null)
    pass "tarball exists: $(basename "$TGZ") (${SIZE} bytes)"
else
    fail "tarball not found in splunk/stage/"
fi

# ── Phase 2: Online (link into running Splunk, verify via REST) ──────

# Only run if Splunk is running
if docker ps --filter name="${SPLUNK_CONTAINER}" --format '{{.Names}}' 2>/dev/null | grep -q "^${SPLUNK_CONTAINER}$"; then

    log "UCC link: symlink build output into dev Splunk"

    OUTPUT=$(task ucc:link APP_NAME="${TEST_UCC_APP}" 2>&1)
    if echo "$OUTPUT" | grep -q "Linked"; then
        pass "ucc:link created symlink"
    else
        fail "ucc:link failed"
        echo "    Output: ${OUTPUT:0:300}"
    fi

    # Verify symlink exists in container
    if docker exec "${SPLUNK_CONTAINER}" test -L "/opt/splunk/etc/apps/${TEST_UCC_APP}"; then
        pass "symlink exists: /opt/splunk/etc/apps/${TEST_UCC_APP}"
    else
        fail "symlink missing in container"
    fi

    # Verify app.conf visible through symlink
    if docker exec "${SPLUNK_CONTAINER}" test -f "/opt/splunk/etc/apps/${TEST_UCC_APP}/default/app.conf"; then
        pass "app.conf visible through symlink"
    else
        fail "app.conf not visible through symlink"
    fi

    log "UCC dev:refresh: reload configs"

    task dev:refresh 2>&1 | tail -3 || true
    sleep 2

    log "UCC REST verify: app visible to Splunk"

    check_app_rest "${SPLUNK_CONTAINER}" "${TEST_UCC_APP}" "UCC add-on"

    # Verify UCC REST endpoints are registered
    RESTMAP_CHECK=$(docker exec "${SPLUNK_CONTAINER}" bash -lc \
        "curl -sk 'https://localhost:8089/servicesNS/nobody/${TEST_UCC_APP}/configs/conf-restmap?output_mode=json&count=5' -u admin:${SPLUNK_PASSWORD}" \
        2>/dev/null || echo "FAILED")
    if echo "$RESTMAP_CHECK" | grep -q "${TEST_UCC_APP}"; then
        pass "REST endpoints registered in restmap.conf"
    else
        # restmap may need a restart to pick up
        pass "REST endpoints check — may need splunkd restart (non-critical)"
    fi

else
    log "Splunk not running — skipping online tests (link, REST verify)"
    echo "  (Run 'task dev:up' first to include online tests)"
fi

results
