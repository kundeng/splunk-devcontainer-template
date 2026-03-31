"""Bridge between the Airbyte CDK 6.x declarative runtime and URC/Splunk.

Uses ConcurrentDeclarativeSource.read() which yields AirbyteMessages.
Extracts RECORD messages as plain dicts, STATE messages for checkpointing.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
from airbyte_cdk.models import AirbyteMessage, ConfiguredAirbyteCatalog, SyncMode, Type
from airbyte_cdk.sources.declarative.concurrent_declarative_source import ConcurrentDeclarativeSource

logger = logging.getLogger("urc.cdk_bridge")


def create_source(
    manifest_yaml: str,
    config: Optional[dict] = None,
) -> ConcurrentDeclarativeSource:
    """Parse a manifest YAML string and create a CDK declarative source."""
    manifest_dict = yaml.safe_load(manifest_yaml)
    if not isinstance(manifest_dict, dict):
        raise ValueError(f"Manifest must be a YAML mapping, got {type(manifest_dict).__name__}")
    return ConcurrentDeclarativeSource(
        catalog=None,
        config=config or {},
        state=None,
        source_config=manifest_dict,
    )


def _build_catalog(source: ConcurrentDeclarativeSource, config: dict) -> ConfiguredAirbyteCatalog:
    """Build a ConfiguredAirbyteCatalog from the source's streams."""
    from airbyte_cdk.models import (
        AirbyteStream,
        ConfiguredAirbyteStream,
        DestinationSyncMode,
    )

    streams = source.streams(config)
    configured_streams = []
    for stream in streams:
        airbyte_stream = stream.as_airbyte_stream()
        configured_streams.append(
            ConfiguredAirbyteStream(
                stream=airbyte_stream,
                sync_mode=SyncMode.full_refresh,
                destination_sync_mode=DestinationSyncMode.overwrite,
            )
        )
    return ConfiguredAirbyteCatalog(streams=configured_streams)


def collect(
    manifest_yaml: str,
    config: dict,
    checkpoint: Optional[dict] = None,
) -> Iterator[Tuple[str, dict, Optional[dict]]]:
    """Execute a declarative manifest and yield records.

    Drop-in replacement for the old urc.engine.collect() interface.

    Args:
        manifest_yaml: Raw YAML manifest string.
        config: User config dict (account creds + input fields).
        checkpoint: Optional last checkpoint state {stream_name: state_dict}.

    Yields:
        Tuples of (stream_name, record_dict, updated_state_or_None).
    """
    source = create_source(manifest_yaml, config)
    catalog = _build_catalog(source, config)

    # Convert checkpoint to Airbyte state format
    state = None
    if checkpoint:
        from airbyte_cdk.models import AirbyteStateMessage, AirbyteStreamState, StreamDescriptor
        state = [
            AirbyteStateMessage(
                stream=AirbyteStreamState(
                    stream_descriptor=StreamDescriptor(name=stream_name),
                    stream_state=stream_state,
                ),
            )
            for stream_name, stream_state in checkpoint.items()
        ]

    for message in source.read(logger, config, catalog, state):
        if not isinstance(message, AirbyteMessage):
            continue

        if message.type == Type.RECORD and message.record:
            stream_name = message.record.stream
            record = message.record.data or {}
            yield (stream_name, dict(record), None)

        elif message.type == Type.STATE and message.state:
            # Extract stream state for checkpointing
            if hasattr(message.state, 'stream') and message.state.stream:
                stream_name = message.state.stream.stream_descriptor.name
                stream_state = message.state.stream.stream_state or {}
                # AirbyteStateBlob may not be a plain dict — convert safely
                if hasattr(stream_state, '__dict__'):
                    stream_state = {k: v for k, v in stream_state.__dict__.items() if not k.startswith('_')}
                yield (stream_name, {}, dict(stream_state) if isinstance(stream_state, dict) else {})


def check_connection(manifest_yaml: str, config: dict) -> Tuple[bool, Optional[str]]:
    """Test the connection using the CDK's check mechanism."""
    source = create_source(manifest_yaml, config)
    try:
        return source.check_connection(logger, config)
    except Exception as e:
        return (False, str(e))
