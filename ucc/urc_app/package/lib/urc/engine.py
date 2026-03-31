# Collection engine — processes a manifest and yields records.

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from urc import registry
from urc.manifest import process_manifest
from urc.interpolation import eval_string

# Import components to trigger registration
import urc.components  # noqa: F401

logger = logging.getLogger(__name__)


def collect(
    manifest_yaml: str,
    config: dict,
    checkpoint: Optional[dict] = None,
) -> Iterator[Tuple[str, dict, Optional[dict]]]:
    """Execute a declarative manifest and yield records.

    Args:
        manifest_yaml: Raw YAML manifest string.
        config: User config dict (account creds + input fields).
        checkpoint: Optional last checkpoint state {stream_name: state_dict}.

    Yields:
        Tuples of (stream_name, record_dict, updated_state_or_None).
    """
    # Process manifest: parse → resolve refs → propagate types
    manifest = process_manifest(manifest_yaml)

    streams = manifest.get("streams", [])
    if not streams:
        logger.warning("Manifest has no streams defined")
        return

    for stream_def in streams:
        stream_name = stream_def.get("name", "default")
        logger.info(f"Starting collection for stream '{stream_name}'")

        try:
            yield from _collect_stream(stream_def, config, stream_name, checkpoint)
        except Exception as e:
            logger.error(f"Error collecting stream '{stream_name}': {e}")
            raise


def _collect_stream(
    stream_def: dict,
    config: dict,
    stream_name: str,
    checkpoint: Optional[dict] = None,
) -> Iterator[Tuple[str, dict, Optional[dict]]]:
    """Collect records from a single stream definition."""
    # Build retriever
    retriever_def = stream_def.get("retriever", {})
    if not retriever_def.get("type"):
        retriever_def["type"] = "SimpleRetriever"
    retriever = registry.create(retriever_def, config)

    # Build incremental cursor if present
    incremental_def = stream_def.get("incremental_sync")
    cursor = None
    if incremental_def:
        cursor = _build_cursor(incremental_def, config, stream_name, checkpoint)

    # Determine stream slices (time windows for incremental, or single empty slice)
    if cursor:
        slices = cursor.get_slices()
    else:
        slices = [{}]  # Single full-refresh slice

    # Build partition router if defined on the retriever
    partition_router_def = retriever_def.get("partition_router")
    partitions: List[dict] = [{}]  # default: single empty partition
    if partition_router_def and partition_router_def.get("type"):
        router = registry.create(partition_router_def, config)
        # SubstreamPartitionRouter.get_partitions accepts checkpoint;
        # ListPartitionRouter.get_partitions does not — use try/except to
        # handle both signatures simply.
        try:
            partitions = list(router.get_partitions(config, checkpoint=checkpoint))
        except TypeError:
            partitions = list(router.get_partitions(config))
        if not partitions:
            partitions = [{}]

    # Build transformation pipeline if defined
    transformations = []
    for t_def in stream_def.get("transformations", []):
        transformations.append(registry.create(t_def, config))

    record_count = 0
    for partition in partitions:
        for stream_slice in slices:
            merged_slice = {**stream_slice, **partition}
            for record in retriever.read_records(config, stream_slice=merged_slice):
                # Apply transformations in order
                for t in transformations:
                    record = t.transform(
                        record, config,
                        stream_partition=merged_slice,
                        stream_name=stream_name,
                    )

                record_count += 1

                # Update cursor state if incremental
                state = None
                if cursor:
                    cursor.observe(record)
                    state = cursor.get_state()

                yield (stream_name, record, state)

    # Emit final state after all records
    if cursor:
        final_state = cursor.get_state()
        if final_state:
            yield (stream_name, {}, final_state)

    logger.info(f"Stream '{stream_name}': collected {record_count} records")


def _build_cursor(
    incremental_def: dict,
    config: dict,
    stream_name: str,
    checkpoint: Optional[dict] = None,
) -> Any:
    """Build an incremental cursor from the manifest definition."""
    cursor_type = incremental_def.get("type", "")

    if cursor_type == "DatetimeBasedCursor":
        return DatetimeBasedCursor(incremental_def, config, stream_name, checkpoint)
    elif cursor_type == "IncrementingCountCursor":
        return IncrementingCountCursor(incremental_def, config, stream_name, checkpoint)
    else:
        logger.warning(f"Unknown cursor type '{cursor_type}', falling back to full refresh")
        return None


class DatetimeBasedCursor:
    """Tracks incremental sync state based on a datetime cursor field."""

    def __init__(self, definition: dict, config: dict, stream_name: str,
                 checkpoint: Optional[dict] = None):
        self._cursor_field = eval_string(definition.get("cursor_field", ""), config)
        self._start_datetime = eval_string(definition.get("start_datetime", ""), config)
        self._datetime_format = definition.get("datetime_format", "%Y-%m-%dT%H:%M:%SZ")
        self._config = config

        # Load from checkpoint or start from beginning
        self._cursor_value = None
        if checkpoint and stream_name in checkpoint:
            self._cursor_value = checkpoint[stream_name].get("cursor_value")
        if not self._cursor_value:
            self._cursor_value = self._start_datetime

    def get_slices(self) -> list:
        """Return stream slices. For now, single slice with start/end times."""
        return [{"start_time": self._cursor_value}]

    def observe(self, record: dict) -> None:
        """Update cursor based on a record's cursor field value."""
        value = record.get(self._cursor_field)
        if value is not None:
            if self._cursor_value is None or str(value) > str(self._cursor_value):
                self._cursor_value = str(value)

    def get_state(self) -> dict:
        """Return the current cursor state."""
        return {
            "cursor_field": self._cursor_field,
            "cursor_value": self._cursor_value,
        }


class IncrementingCountCursor:
    """Tracks incremental sync state based on an incrementing integer."""

    def __init__(self, definition: dict, config: dict, stream_name: str,
                 checkpoint: Optional[dict] = None):
        self._cursor_field = eval_string(definition.get("cursor_field", ""), config)
        self._start_value = eval_string(definition.get("start_value", 0), config)
        self._config = config

        self._cursor_value = self._start_value
        if checkpoint and stream_name in checkpoint:
            self._cursor_value = checkpoint[stream_name].get("cursor_value", self._start_value)

    def get_slices(self) -> list:
        return [{"start_value": self._cursor_value}]

    def observe(self, record: dict) -> None:
        value = record.get(self._cursor_field)
        if value is not None:
            try:
                int_value = int(value)
                if int_value > int(self._cursor_value or 0):
                    self._cursor_value = int_value
            except (ValueError, TypeError):
                pass

    def get_state(self) -> dict:
        return {
            "cursor_field": self._cursor_field,
            "cursor_value": self._cursor_value,
        }
