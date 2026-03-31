# Partition routers — generate stream slices from parent streams or static lists.

import logging
from typing import Any, Dict, Iterator, List, Optional

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


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
                logger.warning(
                    "SubstreamPartitionRouter: skipping config with missing "
                    "parent_key or partition_field"
                )
                continue

            parent_stream_name = parent_stream_def.get("name", "parent")

            logger.debug(
                "SubstreamPartitionRouter: collecting parent stream '%s'",
                parent_stream_name,
            )

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
                    logger.warning(
                        "ListPartitionRouter: could not evaluate values "
                        "expression '%s' to a list",
                        values,
                    )
                    return
            values = evaluated

        if not isinstance(values, (list, tuple)):
            logger.warning(
                "ListPartitionRouter: values is not a list (got %s)",
                type(values).__name__,
            )
            return

        for v in values:
            yield {self._cursor_field: v}
