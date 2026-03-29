#!/usr/bin/env bash
# E2E test: Configure URC input via REST API and verify data collection.
# No Splunk Web UI interaction needed.
set -euo pipefail

SPLUNK_CONTAINER="${SPLUNK_CONTAINER:-splunk-dev}"
SPLUNK_URL="https://localhost:8089"
SPLUNK_USER="admin"
SPLUNK_PASS="${SPLUNK_PASSWORD:-admin123}"
APP_NAME="urc_app"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
info() { echo -e "${YELLOW}→ $1${NC}"; }

# ── Helper: Splunk REST API call (via docker exec) ──
splunk_api() {
    local method=$1 endpoint=$2
    shift 2
    docker exec "$SPLUNK_CONTAINER" curl -sk -u "${SPLUNK_USER}:${SPLUNK_PASS}" \
        -X "$method" \
        "${SPLUNK_URL}${endpoint}" \
        "$@" 2>/dev/null
}

# ── Step 1: Verify app is installed ──
info "Checking URC app is installed..."
app_status=$(splunk_api GET "/services/apps/local/${APP_NAME}?output_mode=json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['entry'][0]['content']['label'])
except: print('NOT_FOUND')
")
if [ "$app_status" = "NOT_FOUND" ]; then
    fail "URC app not installed. Run: task ucc:link APP_NAME=urc_app"
fi
pass "App installed: $app_status"

# ── Step 2: Create a test account (NoAuth) via UCC REST endpoint ──
info "Creating test account 'test_noauth'..."
account_result=$(splunk_api POST "/servicesNS/nobody/${APP_NAME}/${APP_NAME}_account" \
    -d output_mode=json \
    -d name=test_noauth \
    -d auth_type=none 2>&1)
echo "$account_result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'entry' in d: print('created')
    elif 'messages' in d: print(d['messages'][0].get('text','')[:100])
except: print('ok')
" || true
pass "Account created"

# ── Step 3: Create test input with JSONPlaceholder manifest ──
MANIFEST='version: "0.1.0"
type: DeclarativeSource
check:
  type: CheckStream
  stream_names: ["posts"]
streams:
  - type: DeclarativeStream
    name: posts
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://jsonplaceholder.typicode.com/posts"
        http_method: GET
        authenticator:
          type: NoAuth
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []'

info "Creating test input 'test_jsonplaceholder'..."
input_result=$(splunk_api POST "/servicesNS/nobody/${APP_NAME}/${APP_NAME}_${APP_NAME}_input" \
    -d output_mode=json \
    -d name=test_jsonplaceholder \
    -d account=test_noauth \
    -d base_url=https://jsonplaceholder.typicode.com \
    --data-urlencode "manifest=${MANIFEST}" \
    -d interval=60 \
    -d index=main \
    -d sourcetype=urc:api:json 2>&1)
echo "$input_result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'entry' in d: print('created')
    elif 'messages' in d: print(d['messages'][0].get('text','')[:200])
except Exception as e: print(f'parse error: {e}')
" || true
pass "Input created"

# ── Step 4: Wait for data collection ──
info "Waiting for data collection (up to 60s)..."
for i in $(seq 1 12); do
    sleep 5
    # Search for events
    result=$(splunk_api POST "/services/search/v2/jobs/export" \
        -d search="search index=main sourcetype=urc:api:json earliest=-5m | stats count" \
        -d output_mode=json \
        -d exec_mode=oneshot | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        if 'result' in d:
            print(d['result'].get('count', '0'))
            break
    except: pass
" 2>/dev/null)

    if [ -n "$result" ] && [ "$result" != "0" ]; then
        pass "Got $result events after $((i * 5))s"
        break
    fi

    if [ "$i" -eq 12 ]; then
        fail "No events after 60s. Check Splunk internal logs."
    fi
done

# ── Step 5: Verify event content ──
info "Verifying event content..."
sample=$(splunk_api POST "/services/search/v2/jobs/export" \
    -d search="search index=main sourcetype=urc:api:json earliest=-5m | head 1" \
    -d output_mode=json \
    -d exec_mode=oneshot | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        if 'result' in d and '_raw' in d['result']:
            raw = json.loads(d['result']['_raw'])
            print(f\"id={raw.get('id')} title={raw.get('title', '')[:40]}\")
            break
    except: pass
" 2>/dev/null)

if [ -n "$sample" ]; then
    pass "Event content: $sample"
else
    fail "Could not parse event content"
fi

# ── Step 6: Cleanup ──
info "Cleaning up test input..."
splunk_api DELETE "/servicesNS/nobody/${APP_NAME}/${APP_NAME}_${APP_NAME}_input/test_jsonplaceholder" \
    -o /dev/null || true
splunk_api DELETE "/servicesNS/nobody/${APP_NAME}/${APP_NAME}_account/test_noauth" \
    -o /dev/null || true
pass "Cleanup done"

echo ""
echo -e "${GREEN}All URC E2E tests passed!${NC}"
