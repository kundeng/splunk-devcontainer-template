# Collection engine — processes a manifest and yields records.

import datetime
import hashlib
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from urc import registry
from urc.manifest import process_manifest
from urc.interpolation import eval_string
from urc.structured_logger import emit
from urc.recovery import SkipRecord, SkipComponent
from urc.components.event_timestamp import CursorBasedTimestamp, FetchTimestamp

# Import components to trigger registration
import urc.components  # noqa: F401

# Register instrumentation hooks on first import
from urc.instrumentation import register_hook
from urc.structured_logger import before_hook, after_hook, error_hook
from urc.recovery import recovery_error_hook

register_hook("before", before_hook)
register_hook("after", after_hook)
register_hook("error", error_hook)
register_hook("error", recovery_error_hook)


# ── Partition key helpers ──


def _partition_key(partition: dict) -> str:
    """Deterministic hash of a partition dict for use as state key.

    Produces a short hex string so the KV Store state dict stays
    readable in Splunk's kvstore inspector.
    """
    if not partition:
        return "__global__"
    canonical = json.dumps(partition, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


# ── Public API ──


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
    manifest = process_manifest(manifest_yaml)

    streams = manifest.get("streams", [])
    if not streams:
        emit(action="warning", message="manifest has no streams defined")
        return

    for stream_def in streams:
        stream_name = stream_def.get("name", "default")
        emit(action="stream_start", stream=stream_name)

        try:
            yield from _collect_stream(stream_def, config, stream_name, checkpoint)
        except SkipComponent:
            emit(action="stream_skip", stream=stream_name)
        except Exception as e:
            emit(action="stream_error", stream=stream_name,
                 error_type=type(e).__name__, error=str(e)[:200])
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

    # Build incremental cursor definition (shared across partitions)
    incremental_def = stream_def.get("incremental_sync")

    # Build partition router if defined on the retriever
    partition_router_def = retriever_def.get("partition_router")
    partitions: List[dict] = [{}]  # default: single empty partition
    if partition_router_def and partition_router_def.get("type"):
        router = registry.create(partition_router_def, config)
        try:
            partitions = list(router.get_partitions(config, checkpoint=checkpoint))
        except TypeError:
            partitions = list(router.get_partitions(config))
        if not partitions:
            partitions = [{}]

    # Build transformation pipeline
    transformations = []
    for t_def in stream_def.get("transformations", []):
        transformations.append(registry.create(t_def, config))

    # Build event timestamp resolver
    ts_def = stream_def.get("event_timestamp")
    if ts_def and ts_def.get("type"):
        ts_resolver = registry.create(ts_def, config)
    elif incremental_def and incremental_def.get("type") == "DatetimeBasedCursor":
        ts_resolver = CursorBasedTimestamp({}, config)
    else:
        ts_resolver = FetchTimestamp({}, config)

    # Per-partition state manager
    state_mgr = PerPartitionStateManager(stream_name, checkpoint)

    start_time = time.monotonic()
    record_count = 0
    skip_count = 0
    for partition in partitions:
        pk = _partition_key(partition)

        # Build a per-partition cursor from the incremental definition
        cursor = None
        if incremental_def:
            cursor = _build_cursor(
                incremental_def, config, stream_name,
                partition_state=state_mgr.get_partition_state(pk),
            )

        # Determine stream slices (time windows for incremental, or single empty slice)
        slices = cursor.get_slices() if cursor else [{}]

        for stream_slice in slices:
            merged_slice = {**stream_slice, **partition}
            for record in retriever.read_records(config, stream_slice=merged_slice):
                try:
                    # Apply transformations
                    for t in transformations:
                        record = t.transform(
                            record, config,
                            stream_partition=merged_slice,
                            stream_name=stream_name,
                        )

                    # Resolve Splunk _time
                    record["_time"] = ts_resolver.resolve(record, config, stream_def)
                except SkipRecord:
                    skip_count += 1
                    continue

                record_count += 1

                # Update cursor state
                state = None
                if cursor:
                    cursor.observe(record)
                    partition_cursor_state = cursor.get_state()
                    state_mgr.update_partition(pk, partition, partition_cursor_state)
                    state = state_mgr.get_full_state()

                yield (stream_name, record, state)

    # Emit final state after all records
    final_state = state_mgr.get_full_state()
    if final_state:
        yield (stream_name, {}, final_state)

    elapsed = time.monotonic() - start_time
    emit(action="stream_complete", stream=stream_name,
         records=record_count, skipped=skip_count,
         partitions=len(partitions), elapsed=f"{elapsed:.1f}")


# ── Per-partition state manager ──


class PerPartitionStateManager:
    """Manages incremental state per partition within a stream.

    State structure in checkpoint:
        {
            "stream_name": {
                "use_global_cursor": false,
                "partitions": {
                    "<partition_key>": {
                        "partition": {"repo_name": "foo"},
                        "cursor": {"cursor_field": "updated_at", "cursor_value": "..."}
                    },
                    ...
                },
                "global_cursor": {"cursor_field": "updated_at", "cursor_value": "..."}
            }
        }

    For non-partitioned streams, a single "__global__" partition is used,
    making this backward-compatible with the old flat state format.
    """

    def __init__(self, stream_name: str, checkpoint: Optional[dict] = None):
        self._stream_name = stream_name
        self._partitions: Dict[str, dict] = {}
        self._global_cursor: Optional[dict] = None

        # Load from checkpoint
        if checkpoint and stream_name in checkpoint:
            stream_state = checkpoint[stream_name]
            if isinstance(stream_state, dict):
                if "partitions" in stream_state:
                    # New per-partition format
                    for pk, pdata in stream_state["partitions"].items():
                        self._partitions[pk] = pdata
                    self._global_cursor = stream_state.get("global_cursor")
                else:
                    # Legacy flat format: treat as global partition
                    self._partitions["__global__"] = {
                        "partition": {},
                        "cursor": stream_state,
                    }
                    self._global_cursor = stream_state

    def get_partition_state(self, partition_key: str) -> Optional[dict]:
        """Get the cursor state for a specific partition."""
        pdata = self._partitions.get(partition_key)
        if pdata:
            return pdata.get("cursor")
        return None

    def update_partition(
        self, partition_key: str, partition: dict, cursor_state: dict
    ) -> None:
        """Update state for a specific partition and refresh global cursor."""
        self._partitions[partition_key] = {
            "partition": partition,
            "cursor": cursor_state,
        }
        # Update global cursor (tracks overall max across all partitions)
        self._update_global_cursor(cursor_state)

    def _update_global_cursor(self, cursor_state: dict) -> None:
        """Track the maximum cursor value across all partitions."""
        if not cursor_state:
            return
        new_value = cursor_state.get("cursor_value")
        if new_value is None:
            return

        if self._global_cursor is None:
            self._global_cursor = dict(cursor_state)
            return

        old_value = self._global_cursor.get("cursor_value")
        if old_value is None or str(new_value) > str(old_value):
            self._global_cursor = dict(cursor_state)

    def get_full_state(self) -> Optional[dict]:
        """Return the full state dict for this stream."""
        if not self._partitions:
            return None

        # For single __global__ partition, emit flat format for backward compat
        if list(self._partitions.keys()) == ["__global__"]:
            return self._partitions["__global__"].get("cursor")

        return {
            "partitions": dict(self._partitions),
            "global_cursor": self._global_cursor,
        }


# ── Cursor builders ──


def _build_cursor(
    incremental_def: dict,
    config: dict,
    stream_name: str,
    partition_state: Optional[dict] = None,
    # Legacy compat — ignored if partition_state is given
    checkpoint: Optional[dict] = None,
) -> Any:
    """Build an incremental cursor from the manifest definition."""
    cursor_type = incremental_def.get("type", "")

    # Resolve partition state from legacy checkpoint if needed
    if partition_state is None and checkpoint and stream_name in checkpoint:
        partition_state = checkpoint[stream_name]

    if cursor_type == "DatetimeBasedCursor":
        return DatetimeBasedCursor(incremental_def, config, stream_name, partition_state)
    elif cursor_type == "IncrementingCountCursor":
        return IncrementingCountCursor(incremental_def, config, stream_name, partition_state)
    else:
        emit(action="warning", message=f"unknown cursor type '{cursor_type}', falling back to full refresh")
        return None


# ── Cursor implementations ──


class DatetimeBasedCursor:
    """Tracks incremental sync state based on a datetime cursor field.

    Features:
    - Per-partition state (via partition_state arg)
    - Lookback windows (subtract duration from cursor before querying)
    - MinMaxDatetime clamping (clamp start/end to configured bounds)
    - Resumable full refresh (checkpoint even in full-refresh mode)
    """

    def __init__(
        self,
        definition: dict,
        config: dict,
        stream_name: str,
        partition_state: Optional[dict] = None,
    ):
        self._cursor_field = eval_string(definition.get("cursor_field", ""), config)
        self._start_datetime = eval_string(definition.get("start_datetime", ""), config)
        self._end_datetime = eval_string(definition.get("end_datetime", ""), config)
        self._datetime_format = definition.get("datetime_format", "%Y-%m-%dT%H:%M:%SZ")
        self._cursor_granularity = definition.get("cursor_granularity")
        self._config = config

        # Lookback window: subtract this duration from cursor before querying
        self._lookback_window = definition.get("lookback_window")

        # Step: split time range into windows of this size
        self._step = definition.get("step")

        # Load from partition state or start from beginning
        self._cursor_value = None
        if partition_state:
            self._cursor_value = partition_state.get("cursor_value")
        if not self._cursor_value:
            self._cursor_value = self._start_datetime

    def get_slices(self) -> list:
        """Return stream slices for incremental sync.

        If step is configured, splits the time range into windows.
        If lookback_window is configured, adjusts the start time backward.
        """
        effective_start = self._cursor_value or self._start_datetime

        # Apply lookback window
        if self._lookback_window and effective_start:
            effective_start = self._apply_lookback(effective_start)

        # Determine end time
        end_time = self._end_datetime
        if not end_time:
            end_time = datetime.datetime.now(
                datetime.timezone.utc
            ).strftime(self._datetime_format)

        # If step is configured, split into windows
        if self._step and effective_start:
            return self._generate_time_windows(effective_start, end_time)

        return [{"start_time": effective_start, "end_time": end_time}]

    def _apply_lookback(self, start_value: str) -> str:
        """Subtract lookback window from the start cursor value."""
        try:
            delta = _parse_duration(self._lookback_window)
            if delta:
                dt = datetime.datetime.strptime(start_value, self._datetime_format)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                adjusted = dt - delta
                return adjusted.strftime(self._datetime_format)
        except (ValueError, TypeError):
            pass  # silently fall back to original start value
        return start_value

    def _generate_time_windows(self, start: str, end: str) -> list:
        """Split a time range into windows of size `step`."""
        try:
            step_delta = _parse_duration(self._step)
            if not step_delta:
                return [{"start_time": start, "end_time": end}]

            start_dt = datetime.datetime.strptime(start, self._datetime_format)
            end_dt = datetime.datetime.strptime(end, self._datetime_format)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)

            windows = []
            current = start_dt
            while current < end_dt:
                window_end = min(current + step_delta, end_dt)
                windows.append({
                    "start_time": current.strftime(self._datetime_format),
                    "end_time": window_end.strftime(self._datetime_format),
                })
                current = window_end

                # Safety: max 10000 windows
                if len(windows) >= 10000:
                    emit(action="warning", message="time window safety limit reached (10000)")
                    break

            return windows if windows else [{"start_time": start, "end_time": end}]
        except (ValueError, TypeError) as e:
            pass  # silently fall back to single window
            return [{"start_time": start, "end_time": end}]

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

    def __init__(
        self,
        definition: dict,
        config: dict,
        stream_name: str,
        partition_state: Optional[dict] = None,
    ):
        self._cursor_field = eval_string(definition.get("cursor_field", ""), config)
        self._start_value = eval_string(definition.get("start_value", 0), config)
        self._config = config

        self._cursor_value = self._start_value
        if partition_state:
            self._cursor_value = partition_state.get(
                "cursor_value", self._start_value
            )

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


# ── Duration parser ──


def _parse_duration(duration_str: str) -> Optional[datetime.timedelta]:
    """Parse an ISO 8601 duration string (e.g. 'P1D', 'PT1H', 'P30D').

    Falls back to isodate if available, otherwise handles common patterns.
    """
    if not duration_str:
        return None

    try:
        import isodate
        return isodate.parse_duration(duration_str)
    except ImportError:
        pass
    except Exception:
        pass

    # Simple fallback parser for common patterns
    import re
    match = re.match(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        duration_str,
    )
    if not match:
        return None

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
