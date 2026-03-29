# Retriever — orchestrates requester + paginator + record selector.

import logging
from typing import Any, Dict, Iterator, List, Optional

import requests

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


@component("SimpleRetriever")
class SimpleRetriever:
    """Fetches records by making paginated HTTP requests."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        # Requester
        requester_def = definition.get("requester", {})
        self._requester = create_component(requester_def, config)

        # Record selector
        selector_def = definition.get("record_selector", {})
        self._selector = create_component(selector_def, config)

        # Paginator
        paginator_def = definition.get("paginator")
        if paginator_def and paginator_def.get("type", "NoPagination") != "NoPagination":
            self._paginator = create_component(paginator_def, config)
        else:
            from urc.components.pagination import NoPagination
            self._paginator = NoPagination({}, config)

        self._config = config

    def read_records(
        self,
        config: dict,
        stream_slice: Optional[dict] = None,
    ) -> Iterator[dict]:
        """Fetch all records across all pages."""
        extra_params = {}
        page_count = 0

        while True:
            # Make request
            response = self._requester.send(
                config,
                extra_params=extra_params if extra_params else None,
                stream_slice=stream_slice,
            )

            # Parse response
            try:
                response_body = response.json()
            except Exception:
                logger.warning(f"Non-JSON response: {response.text[:200]}")
                break

            # Extract records
            records = self._selector.select(response_body, config)
            yield from records

            # Check for next page
            next_params = self._paginator.next_page(
                response, extra_params, records, page_count
            )
            if next_params is None:
                break

            extra_params = next_params
            page_count += 1

            if page_count > 10000:
                logger.error("Pagination safety limit reached (10000 pages)")
                break
