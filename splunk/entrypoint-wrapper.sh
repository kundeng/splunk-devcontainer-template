#!/bin/bash
# entrypoint-wrapper.sh — Wraps the official Splunk entrypoint to skip
# Ansible re-provisioning on subsequent starts. On first run (or when
# FORCE_PROVISION=true), delegates to the stock entrypoint. On later
# starts, starts splunkd directly (~15s vs 60-120s).
#
# Marker file: /opt/splunk/var/.provisioned (persists in splunk-var volume)

set -e

MARKER_FILE="/opt/splunk/var/.provisioned"
STOCK_ENTRYPOINT="/sbin/entrypoint.sh"
SPLUNK_HOME="${SPLUNK_HOME:-/opt/splunk}"
SPLUNK_USER="${SPLUNK_USER:-splunk}"

# ── Graceful shutdown ────────────────────────────────────────────────
teardown() {
    echo "entrypoint-wrapper: caught signal, stopping splunkd..."
    if [ "$(whoami)" != "${SPLUNK_USER}" ]; then
        sudo -u "${SPLUNK_USER}" "${SPLUNK_HOME}/bin/splunk" stop 2>/dev/null || true
    else
        "${SPLUNK_HOME}/bin/splunk" stop 2>/dev/null || true
    fi
    exit 0
}
trap teardown SIGINT SIGTERM

# ── Decide: provision or skip ────────────────────────────────────────
should_skip_provision() {
    # Only skip for start/start-service commands (the default CMD)
    case "${1:-start-service}" in
        start|start-service) ;;
        *) return 1 ;;  # non-start commands always go to stock entrypoint
    esac

    # Force provision if requested
    if [ "${FORCE_PROVISION}" = "true" ]; then
        echo "entrypoint-wrapper: FORCE_PROVISION=true — running full provisioning"
        return 1
    fi

    # Skip only if marker exists
    if [ -f "${MARKER_FILE}" ]; then
        return 0
    fi

    return 1
}

# ── Skip-provision path: start splunkd directly ──────────────────────
start_without_provision() {
    echo "entrypoint-wrapper: marker found — skipping Ansible provisioning"
    echo "entrypoint-wrapper: starting splunkd directly..."

    # Start splunkd (run as splunk user if we're not already)
    if [ "$(whoami)" != "${SPLUNK_USER}" ]; then
        sudo -u "${SPLUNK_USER}" "${SPLUNK_HOME}/bin/splunk" start --accept-license --answer-yes --no-prompt
    else
        "${SPLUNK_HOME}/bin/splunk" start --accept-license --answer-yes --no-prompt
    fi

    echo "entrypoint-wrapper: splunkd started, tailing stderr log"

    # Tail the log (same behavior as stock entrypoint's watch_for_failure)
    TAIL_FILE="${SPLUNK_TAIL_FILE:-${SPLUNK_HOME}/var/log/splunk/splunkd_stderr.log}"
    if [ "$(whoami)" != "${SPLUNK_USER}" ]; then
        sudo -u "${SPLUNK_USER}" tail -n 0 -F "${TAIL_FILE}" &
    else
        tail -n 0 -F "${TAIL_FILE}" &
    fi
    wait
}

# ── Full provision path: delegate to stock entrypoint + write marker ─
start_with_provision() {
    echo "entrypoint-wrapper: no marker (or forced) — delegating to stock entrypoint"

    # Run the stock entrypoint in background so we can write the marker after
    # The stock entrypoint handles everything: setup, Ansible, watch_for_failure
    "${STOCK_ENTRYPOINT}" "$@" &
    STOCK_PID=$!

    # Wait for Splunk to become healthy (poll the state file)
    STATE_FILE="${CONTAINER_ARTIFACT_DIR:-/opt/container_artifact}/splunk-container.state"
    TIMEOUT=300
    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if [ -f "${STATE_FILE}" ]; then
            STATE=$(cat "${STATE_FILE}" 2>/dev/null || echo "")
            if [ "${STATE}" = "started" ]; then
                echo "entrypoint-wrapper: provisioning complete — writing marker"
                touch "${MARKER_FILE}"
                break
            fi
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "entrypoint-wrapper: WARNING — timed out waiting for provisioning (${TIMEOUT}s)"
        echo "entrypoint-wrapper: marker NOT written; next start will re-provision"
    fi

    # Wait for the stock entrypoint process (it tails logs indefinitely)
    wait $STOCK_PID
}

# ── Main ─────────────────────────────────────────────────────────────
CMD="${1:-start-service}"

if should_skip_provision "$CMD"; then
    start_without_provision
else
    start_with_provision "$@"
fi
