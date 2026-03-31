import import_declare_test  # noqa: F401  # must be first — sets up sys.path
import itertools
import json
import logging
import traceback

from splunklib.persistent_server_connection_application import (
    PersistentServerConnectionApplication,
)

from urc.engine import collect
from urc.manifest import process_manifest

try:
    from splunk.clilib.bundle_paths import make_splunkhome_path  # noqa: F401
except ImportError:
    pass

ADDON_NAME = "urc_app"
MAX_TEST_RECORDS = 5

logger = logging.getLogger(f"{ADDON_NAME}.test_connection")


def _get_account_config(session_key, account_name):
    """Read account credentials via ConfManager (same approach as input_helper)."""
    from solnlib import conf_manager

    cfm = conf_manager.ConfManager(
        session_key,
        ADDON_NAME,
        realm=f"__REST_CREDENTIAL__#{ADDON_NAME}#configs/conf-{ADDON_NAME}_account",
    )
    account_conf = cfm.get_conf(f"{ADDON_NAME}_account")
    return dict(account_conf.get(account_name))


def _build_config_dict(account_config, base_url):
    """Merge account credentials with the supplied base_url into a config dict."""
    return {
        "base_url": base_url or account_config.get("base_url", ""),
        "auth_type": account_config.get("auth_type", "none"),
        "api_key": account_config.get("api_key", ""),
        "api_key_header": account_config.get("api_key_header", "X-API-Key"),
        "bearer_token": account_config.get("bearer_token", ""),
        "username": account_config.get("username", ""),
        "password": account_config.get("password", ""),
        "client_id": account_config.get("client_id", ""),
        "client_secret": account_config.get("client_secret", ""),
        "token_url": account_config.get("token_url", ""),
    }


def _json_response(status_code, body):
    """Return a dict formatted for Splunk's REST handler response."""
    return {
        "status": status_code,
        "headers": {"Content-Type": "application/json"},
        "payload": json.dumps(body, ensure_ascii=False, default=str),
    }


def _parse_payload(raw_payload):
    """Parse the request payload.

    Splunk sends POST payloads as either URL-encoded form data or raw JSON,
    depending on the caller.  Try JSON first, then fall back to
    ``urllib.parse.parse_qs``.
    """
    if not raw_payload:
        return {}

    # Try JSON first
    try:
        parsed = json.loads(raw_payload)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to URL-encoded form data
    from urllib.parse import parse_qs, unquote_plus

    qs = parse_qs(raw_payload, keep_blank_values=True)
    # parse_qs returns lists; flatten single-value keys
    return {k: v[0] if len(v) == 1 else v for k, v in qs.items()}


class TestConnectionHandler(PersistentServerConnectionApplication):
    """Custom REST endpoint to test an API connection by running a dry-run
    collection with a given manifest, returning up to MAX_TEST_RECORDS sample
    records.

    Endpoint: POST /services/urc_app/test_connection
    Payload (JSON or form-encoded):
        manifest      — YAML manifest string
        account_name  — name of the saved UCC account to read credentials from
        base_url      — (optional) override the base URL from the account
    """

    def __init__(self, command_line, command_arg):
        super().__init__(command_line, command_arg)

    # ------------------------------------------------------------------
    # POST  /services/urc_app/test_connection
    # ------------------------------------------------------------------
    def handlePOST(self, request_info):  # noqa: N802 — Splunk naming convention
        try:
            session_key = request_info.get("session", {}).get("authtoken", "")
            if not session_key:
                session_key = request_info.get("system_authtoken", "")

            payload = _parse_payload(request_info.get("payload", ""))

            manifest_yaml = payload.get("manifest", "")
            account_name = payload.get("account_name", "")
            base_url = payload.get("base_url", "")

            # ---- Validate required fields --------------------------------
            if not manifest_yaml:
                return _json_response(400, {
                    "status": "error",
                    "message": "Missing required field: manifest",
                })

            if not account_name:
                return _json_response(400, {
                    "status": "error",
                    "message": "Missing required field: account_name",
                })

            # ---- Read account credentials --------------------------------
            try:
                account_config = _get_account_config(session_key, account_name)
            except Exception as exc:
                logger.error("Failed to read account '%s': %s", account_name, exc)
                return _json_response(400, {
                    "status": "error",
                    "message": f"Unable to read account '{account_name}': {exc}",
                })

            config_dict = _build_config_dict(account_config, base_url)

            # ---- Validate manifest ---------------------------------------
            try:
                process_manifest(manifest_yaml)
            except Exception as exc:
                return _json_response(400, {
                    "status": "error",
                    "message": f"Manifest validation failed: {exc}",
                })

            # ---- Dry-run collection (first page, max N records) ----------
            records = []
            try:
                for stream_name, record, _state in itertools.islice(
                    collect(manifest_yaml, config_dict, checkpoint=None),
                    MAX_TEST_RECORDS,
                ):
                    if record:
                        records.append({
                            "stream": stream_name,
                            "data": record,
                        })
            except Exception as exc:
                logger.error("Collection dry-run failed: %s", exc)
                return _json_response(502, {
                    "status": "error",
                    "message": f"Collection failed: {exc}",
                })

            return _json_response(200, {
                "status": "success",
                "records": records,
                "record_count": len(records),
            })

        except Exception as exc:
            logger.error(
                "Unhandled error in test_connection: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            return _json_response(500, {
                "status": "error",
                "message": f"Internal error: {exc}",
            })

    # ------------------------------------------------------------------
    # Reject other methods gracefully
    # ------------------------------------------------------------------
    def handleGET(self, request_info):  # noqa: N802
        return _json_response(405, {
            "status": "error",
            "message": "Use POST to test a connection.",
        })

    def handleDELETE(self, request_info):  # noqa: N802
        return _json_response(405, {
            "status": "error",
            "message": "DELETE is not supported on this endpoint.",
        })
