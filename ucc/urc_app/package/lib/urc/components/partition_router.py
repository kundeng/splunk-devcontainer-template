# Partition routers — generate stream slices from parent streams or static lists.

import itertools
from typing import Any, Dict, Iterator, List, Optional

from urc.interpolation import eval_string
from urc.registry import component
from urc.structured_logger import emit


@component("SubstreamPartitionRouter")
class SubstreamPartitionRouter:
    """Iterate over parent stream records to create partitions for a child stream.

    Each parent_stream_config describes a parent stream whose records are
    collected; for every record the ``parent_key`` value is extracted and
    yielded as ``{partition_field: value}`` so the child retriever can
    reference it via ``{{ stream_partition['partition_field'] }}``.
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._parent_stream_configs: List[dict] = definition.get("parent_stream_configs", [])
        self._config = config

    def get_partitions(
        self,
        config: dict,
        checkpoint: Optional[dict] = None,
    ) -> Iterator[dict]:
        """Yield one partition dict per parent record.

        Args:
            config: User config dict.
            checkpoint: Optional checkpoint state (passed through to parent
                stream collection).
        """
        # Late import to avoid circular dependency (engine imports components).
        from urc.engine import _collect_stream

        for psc in self._parent_stream_configs:
            parent_stream_def = psc.get("stream", {})
            parent_key: str = psc.get("parent_key", "")
            partition_field: str = psc.get("partition_field", "")

            if not parent_key or not partition_field:
                continue

            parent_stream_name = parent_stream_def.get("name", "parent")
            emit(action="parent_collect", component="SubstreamPartitionRouter",
                 parent=parent_stream_name, partition_field=partition_field)
            seen_values: set = set()
            for _name, record, _state in _collect_stream(
                parent_stream_def, config, parent_stream_name, checkpoint
            ):
                # Skip the empty trailing-state record emitted by _collect_stream
                if not record:
                    continue

                value = record.get(parent_key)
                if value is None:
                    continue

                # Deduplicate partition values
                hashable = str(value)
                if hashable in seen_values:
                    continue
                seen_values.add(hashable)

                yield {partition_field: value}


@component("ListPartitionRouter")
class ListPartitionRouter:
    """Iterate over a static (or config-derived) list of values.

    ``values`` can be a plain list of strings **or** a Jinja2 expression
    that evaluates to a list (e.g. ``"{{ config['regions'] }}"``).
    Each value is yielded as ``{cursor_field: value}``.
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._cursor_field: str = definition.get("cursor_field", "")
        self._values_raw: Any = definition.get("values", [])
        self._config = config

    def get_partitions(self, config: dict) -> Iterator[dict]:
        """Yield one partition dict per value in the list.

        Args:
            config: User config dict (used when ``values`` is a template).
        """
        values = self._values_raw

        # If values is a Jinja2 expression string, evaluate it.
        if isinstance(values, str):
            evaluated = eval_string(values, config)
            if isinstance(evaluated, str):
                # eval_string returns the original string on error; attempt
                # to parse as a Python literal (list).
                import ast
                try:
                    evaluated = ast.literal_eval(evaluated)
                except (ValueError, SyntaxError):
                    return
            values = evaluated

        if not isinstance(values, (list, tuple)):
            return

        for v in values:
            yield {self._cursor_field: v}


_CARTESIAN_LIMIT = 100_000


@component("CartesianProductStreamSlicer")
class CartesianProductStreamSlicer:
    """Produce the Cartesian product of multiple partition routers.

    Each sub-router's partitions are collected into a list, then
    ``itertools.product`` is used to yield every combination with the
    partition dicts merged together.
    """

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        self._sub_routers = [
            create_component(slicer_def, config)
            for slicer_def in definition.get("stream_slicers", [])
        ]
        self._config = config

    def get_partitions(self, config: dict) -> Iterator[dict]:
        if not self._sub_routers:
            return

        # Collect all partition lists from each sub-router
        partition_lists: List[List[dict]] = []
        total = 1
        for router in self._sub_routers:
            partitions = list(router.get_partitions(config))
            if not partitions:
                return  # empty dimension -> no combinations
            total *= len(partitions)
            if total > _CARTESIAN_LIMIT:
                raise ValueError(
                    f"CartesianProductStreamSlicer: combination count "
                    f"({total}) exceeds safety limit of {_CARTESIAN_LIMIT}"
                )
            partition_lists.append(partitions)

        for combo in itertools.product(*partition_lists):
            merged: dict = {}
            for part in combo:
                merged.update(part)
            yield merged
