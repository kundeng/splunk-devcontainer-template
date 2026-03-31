import json
import logging
import time

import import_declare_test
from solnlib import conf_manager, log
from solnlib.modular_input import checkpointer
from splunklib import modularinput as smi

from urc.cdk_bridge import collect, check_connection

ADDON_NAME = "urc_app"


def logger_for_input(input_name: str) -> logging.Logger:
    return log.Logs().get_logger(f"{ADDON_NAME.lower()}_{input_name}")


def get_account_config(session_key: str, account_name: str) -> dict:
    """Read all account fields (decrypted) from UCC account config."""
    cfm = conf_manager.ConfManager(
        session_key,
        ADDON_NAME,
        realm=f"__REST_CREDENTIAL__#{ADDON_NAME}#configs/conf-{ADDON_NAME}_account",
    )
    account_conf = cfm.get_conf(f"{ADDON_NAME}_account")
    return dict(account_conf.get(account_name))


def build_config_dict(account_config: dict, input_item: dict) -> dict:
    """Merge account credentials + input fields into the config dict
    that manifest interpolation sees as {{ config['api_key'] }}, etc.
    """
    return {
        "base_url": input_item.get("base_url", ""),
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


class CheckpointManager:
    """Bridge between URC engine state and Splunk KV Store."""

    def __init__(self, session_key: str, input_name: str):
        self._ckpt = checkpointer.KVStoreCheckpointer(
            collection_name="urc_checkpoints",
            session_key=session_key,
            app=ADDON_NAME,
        )
        self._key = input_name

    def load(self) -> dict:
        """Load last checkpoint as {stream_name: state_dict}."""
        try:
            data = self._ckpt.get(self._key)
            if data and "state" in data:
                return data["state"]
        except Exception:
            pass
        return {}

    def save(self, stream_name: str, state: dict) -> None:
        """Save stream state to KV Store."""
        try:
            current = self.load()
            if not isinstance(current, dict):
                current = {}
            current[stream_name] = state
            self._ckpt.update(
                self._key,
                {
                    "state": current,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as e:
            logging.getLogger(ADDON_NAME).warning(f"Checkpoint save failed: {e}")


def validate_input(definition: smi.ValidationDefinition):
    """Validate input configuration before saving (called by UCC on save)."""
    manifest_yaml = definition.parameters.get("manifest", "")
    if not manifest_yaml:
        raise ValueError("Manifest YAML is required")
    try:
        from urc.cdk_bridge import create_source
        create_source(manifest_yaml)
    except Exception as e:
        raise ValueError(f"Invalid manifest: {e}")


def stream_events(inputs: smi.InputDefinition, event_writer: smi.EventWriter):
    for input_name, input_item in inputs.inputs.items():
        normalized_input_name = input_name.split("/")[-1]
        logger = logger_for_input(normalized_input_name)
        try:
            session_key = inputs.metadata["session_key"]

            # Configure log level
            log_level = conf_manager.get_log_level(
                logger=logger,
                session_key=session_key,
                app_name=ADDON_NAME,
                conf_name=f"{ADDON_NAME}_settings",
            )
            logger.setLevel(log_level)
            log.modular_input_start(logger, normalized_input_name)

            # 1. Read config
            account_config = get_account_config(session_key, input_item.get("account"))
            config_dict = build_config_dict(account_config, input_item)

            # 2. Get manifest
            manifest_yaml = input_item.get("manifest", "")
            if not manifest_yaml:
                logger.error(f"No manifest configured for input {normalized_input_name}")
                continue

            # 3. Load checkpoint
            ckpt = CheckpointManager(session_key, normalized_input_name)
            checkpoint = ckpt.load()

            # 5. Collect records using URC engine
            sourcetype = input_item.get("sourcetype", "urc:api:json")
            index = input_item.get("index", "main")
            record_count = 0

            for stream_name, record, state in collect(manifest_yaml, config_dict, checkpoint):
                if record:
                    event_writer.write_event(
                        smi.Event(
                            data=json.dumps(record, ensure_ascii=False, default=str),
                            index=index,
                            sourcetype=sourcetype,
                            source=f"urc:{normalized_input_name}:{stream_name}",
                        )
                    )
                    record_count += 1

                if state:
                    ckpt.save(stream_name, state)

            log.events_ingested(
                logger,
                input_name,
                sourcetype,
                record_count,
                index,
                account=input_item.get("account"),
            )
            log.modular_input_end(logger, normalized_input_name)

        except Exception as e:
            log.log_exception(
                logger,
                e,
                "urc_collection_error",
                msg_before=f"Exception raised while collecting data for {normalized_input_name}: ",
            )
