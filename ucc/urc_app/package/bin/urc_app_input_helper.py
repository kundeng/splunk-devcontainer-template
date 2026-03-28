import json
import logging
import time

import import_declare_test
import yaml
from solnlib import conf_manager, log
from solnlib.modular_input import checkpointer
from splunklib import modularinput as smi

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
    """Merge account credentials + input fields into CDK config dict.

    The returned dict is what Airbyte manifest interpolation sees as
    {{ config['base_url'] }}, {{ config['api_key'] }}, etc.
    """
    return {
        # From input
        "base_url": input_item.get("base_url", ""),
        # From account
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


def build_catalog(manifest: dict) -> "ConfiguredAirbyteCatalog":
    """Build a ConfiguredAirbyteCatalog from manifest stream definitions."""
    from airbyte_protocol_dataclasses.models import (
        AirbyteStream,
        ConfiguredAirbyteCatalog,
        ConfiguredAirbyteStream,
        DestinationSyncMode,
        SyncMode,
    )

    streams = []
    for stream_def in manifest.get("streams", []):
        has_incremental = "incremental_sync" in stream_def
        streams.append(
            ConfiguredAirbyteStream(
                stream=AirbyteStream(
                    name=stream_def.get("name", "default"),
                    json_schema={},
                    supported_sync_modes=[
                        SyncMode.incremental if has_incremental else SyncMode.full_refresh
                    ],
                ),
                sync_mode=SyncMode.incremental if has_incremental else SyncMode.full_refresh,
                destination_sync_mode=DestinationSyncMode.append,
            )
        )
    return ConfiguredAirbyteCatalog(streams=streams)


class CheckpointManager:
    """Bridge between Airbyte state messages and Splunk KV Store."""

    def __init__(self, session_key: str, input_name: str):
        self._ckpt = checkpointer.KVStoreCheckpointer(
            collection_name="urc_checkpoints",
            session_key=session_key,
            app=ADDON_NAME,
        )
        self._key = input_name

    def load(self):
        """Load last state as list of AirbyteStateMessage dicts."""
        try:
            data = self._ckpt.get(self._key)
            if data and "state" in data:
                return data["state"]
        except Exception:
            pass
        return []

    def save(self, state_message):
        """Save state message to KV Store."""
        try:
            state_data = state_message
            if hasattr(state_message, "dict"):
                state_data = state_message.dict()
            elif hasattr(state_message, "__dict__"):
                state_data = state_message.__dict__
            self._ckpt.update(
                self._key,
                {
                    "state": [state_data],
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as e:
            # Don't crash on checkpoint failure — log and continue
            logging.getLogger(ADDON_NAME).warning(f"Checkpoint save failed: {e}")


def validate_input(definition: smi.ValidationDefinition):
    """Validate input configuration before saving."""
    manifest_yaml = definition.parameters.get("manifest", "")
    if manifest_yaml:
        try:
            yaml.safe_load(manifest_yaml)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid manifest YAML: {e}")


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

            # 2. Parse manifest
            manifest_yaml = input_item.get("manifest", "")
            if not manifest_yaml:
                logger.error(f"No manifest configured for input {normalized_input_name}")
                continue
            try:
                manifest = yaml.safe_load(manifest_yaml)
            except yaml.YAMLError as e:
                logger.error(f"Invalid manifest YAML for {normalized_input_name}: {e}")
                continue

            # 3. Load checkpoint
            ckpt = CheckpointManager(session_key, normalized_input_name)
            state = ckpt.load()

            # 4. Build catalog from manifest
            catalog = build_catalog(manifest)

            # 5. Create and run declarative source
            from airbyte_cdk.sources.declarative.concurrent_declarative_source import (
                ConcurrentDeclarativeSource,
            )

            source = ConcurrentDeclarativeSource(
                source_config=manifest,
                config=config_dict,
            )

            sourcetype = input_item.get("sourcetype", "urc:api:json")
            index = input_item.get("index", "main")
            record_count = 0

            for message in source.read(logger, config_dict, catalog, state or None):
                msg_type = getattr(message, "type", None)
                msg_type_str = str(msg_type) if msg_type else ""

                if "RECORD" in msg_type_str:
                    record = message.record
                    data = record.data if hasattr(record, "data") else record
                    event_writer.write_event(
                        smi.Event(
                            data=json.dumps(data, ensure_ascii=False, default=str),
                            index=index,
                            sourcetype=sourcetype,
                            source=f"urc:{normalized_input_name}",
                        )
                    )
                    record_count += 1

                elif "STATE" in msg_type_str:
                    ckpt.save(message.state)

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
