# AsyncRetriever — bulk job lifecycle: create → poll → download.
#
# Many enterprise APIs (Salesforce Bulk, Splunk REST, ServiceNow export,
# Azure async operations) use this pattern:
#   1. POST to create a job → get a job ID
#   2. Poll a status endpoint until job completes
#   3. GET the results from a download endpoint
#
# The manifest schema uses AsyncRetriever with:
#   - creation_requester: HTTP request to create the job
#   - polling_requester: HTTP request to check job status
#   - download_retriever: SimpleRetriever for fetching results
#   - status_mapping: maps API status strings to internal states
#   - urls_extractor: extracts download URLs from the job response

import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from urc.interpolation import eval_string, eval_dict
from urc.registry import component, create as create_component
from urc.structured_logger import emit


# ── Job status enum ──

class JobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


# ── Status mapping ──


class StatusMapping:
    """Maps API-specific status strings to internal JobStatus values."""

    def __init__(self, definition: dict):
        self._pending = set(definition.get("pending", ["pending", "queued"]))
        self._running = set(definition.get("running", ["running", "processing", "in_progress"]))
        self._completed = set(definition.get("completed", ["completed", "done", "finished"]))
        self._failed = set(definition.get("failed", ["failed", "error", "aborted"]))

    def resolve(self, status_value: str) -> str:
        """Map an API status string to an internal JobStatus."""
        normalized = str(status_value).lower().strip()
        if normalized in self._completed:
            return JobStatus.COMPLETED
        if normalized in self._failed:
            return JobStatus.FAILED
        if normalized in self._running:
            return JobStatus.RUNNING
        if normalized in self._pending:
            return JobStatus.PENDING
        # Unknown status — treat as running to keep polling
        return JobStatus.RUNNING


# ── URL extractor ──


class UrlsExtractor:
    """Extracts download URLs from a job creation or polling response."""

    def __init__(self, definition: dict, config: dict):
        self._field_path = definition.get("field_path", [])
        self._url_template = definition.get("url", "")
        self._config = config

    def extract(self, response_body: Any, config: dict, **context) -> List[str]:
        """Extract download URLs from the response.

        Supports:
        - field_path: dpath-style extraction from response body
        - url: Jinja2 template with response context
        """
        urls = []

        if self._field_path:
            value = response_body
            for key in self._field_path:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, (list, tuple)) and isinstance(key, int):
                    value = value[key]
                else:
                    value = None
                    break

            if isinstance(value, list):
                urls.extend(str(v) for v in value if v)
            elif value:
                urls.append(str(value))

        elif self._url_template:
            url = eval_string(
                self._url_template, config,
                response=response_body, **context,
            )
            if url and url != self._url_template:
                urls.append(str(url))

        return urls


# ── AsyncRetriever ──


@component("AsyncRetriever")
class AsyncRetriever:
    """Manages async bulk job lifecycle: create → poll → download.

    Manifest schema:
        type: AsyncRetriever
        creation_requester:
            type: HttpRequester
            url: "https://api.example.com/jobs"
            http_method: POST
            request_body_json:
                operation: query
                query: "SELECT Id, Name FROM Account"
        polling_requester:
            type: HttpRequester
            url: "https://api.example.com/jobs/{{ stream_partition['job_id'] }}"
            http_method: GET
        download_retriever:
            type: SimpleRetriever
            requester:
                type: HttpRequester
                url: "{{ stream_partition['download_url'] }}"
                http_method: GET
            record_selector:
                type: RecordSelector
                extractor:
                    type: DpathExtractor
                    field_path: [records]
        status_mapping:
            completed: [completed, done]
            failed: [failed, error]
            running: [running, processing]
            pending: [pending, queued]
        status_extractor:
            field_path: [status]
        urls_extractor:
            field_path: [downloadUrl]
        job_id_extractor:
            field_path: [id]
    """

    # Polling configuration defaults
    DEFAULT_POLL_INTERVAL = 5.0
    DEFAULT_MAX_POLL_TIME = 3600  # 1 hour
    MAX_POLL_INTERVAL = 300  # 5 minutes

    def __init__(self, definition: dict, config: dict, **kwargs):
        # Creation requester — used to POST and create the bulk job
        creation_def = definition.get("creation_requester", {})
        if creation_def:
            self._creation_requester = create_component(creation_def, config)
        else:
            raise ValueError("AsyncRetriever requires a creation_requester")

        # Polling requester — used to check job status
        polling_def = definition.get("polling_requester", {})
        if polling_def:
            self._polling_requester = create_component(polling_def, config)
        else:
            raise ValueError("AsyncRetriever requires a polling_requester")

        # Download retriever — used to fetch results after job completes
        download_def = definition.get("download_retriever", {})
        if download_def:
            if not download_def.get("type"):
                download_def["type"] = "SimpleRetriever"
            self._download_retriever = create_component(download_def, config)
        else:
            self._download_retriever = None

        # Record selector (used if no download_retriever, reading from poll response)
        selector_def = definition.get("record_selector")
        if selector_def:
            self._record_selector = create_component(selector_def, config)
        else:
            self._record_selector = None

        # Status mapping
        status_def = definition.get("status_mapping", {})
        self._status_mapping = StatusMapping(status_def)

        # Status extractor — where to find status in poll response
        status_extractor_def = definition.get("status_extractor", {})
        self._status_field_path = status_extractor_def.get("field_path", ["status"])

        # Job ID extractor — where to find job ID in creation response
        job_id_def = definition.get("job_id_extractor", {})
        self._job_id_field_path = job_id_def.get("field_path", ["id"])

        # URLs extractor — where to find download URLs
        urls_def = definition.get("urls_extractor", {})
        self._urls_extractor = UrlsExtractor(urls_def, config) if urls_def else None

        # Polling configuration
        self._poll_interval = float(definition.get("poll_interval_seconds", self.DEFAULT_POLL_INTERVAL))
        self._max_poll_time = float(definition.get("max_poll_time_seconds", self.DEFAULT_MAX_POLL_TIME))
        self._poll_backoff_factor = float(definition.get("poll_backoff_factor", 1.5))

        # Decoder for creation/polling responses
        decoder_def = definition.get("decoder")
        if decoder_def:
            self._decoder = create_component(decoder_def, config)
        else:
            from urc.components.decoders import JsonDecoder
            self._decoder = JsonDecoder({}, config)

        self._config = config

    def read_records(
        self,
        config: dict,
        stream_slice: Optional[dict] = None,
    ) -> Iterator[dict]:
        """Execute the full async job lifecycle and yield records."""
        # Step 1: Create the job
        job_id, creation_body = self._create_job(config, stream_slice)
        if not job_id:
            emit(action="job_create_failed", component="AsyncRetriever")
            return

        emit(action="job_created", component="AsyncRetriever", job_id=job_id)

        # Step 2: Poll until completion
        job_context = {**(stream_slice or {}), "job_id": job_id}
        final_body = self._poll_job(config, job_context)
        if final_body is None:
            emit(action="job_failed", component="AsyncRetriever", job_id=job_id)
            return

        emit(action="job_completed", component="AsyncRetriever", job_id=job_id)

        # Step 3: Download results
        yield from self._download_results(config, job_context, final_body)

    def _create_job(
        self, config: dict, stream_slice: Optional[dict]
    ) -> tuple:
        """Create the bulk job and return (job_id, response_body)."""
        response = self._creation_requester.send(
            config, stream_slice=stream_slice,
        )
        body = self._decode_response(response)
        job_id = self._extract_field(body, self._job_id_field_path)
        return str(job_id) if job_id is not None else None, body

    def _poll_job(
        self, config: dict, job_context: dict,
    ) -> Optional[dict]:
        """Poll the job status until completion, failure, or timeout."""
        start_time = time.monotonic()
        interval = self._poll_interval
        poll_count = 0

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > self._max_poll_time:
                return None

            # Wait before polling (skip first iteration)
            if poll_count > 0:
                time.sleep(interval)
                # Exponential backoff on poll interval, capped
                interval = min(interval * self._poll_backoff_factor, self.MAX_POLL_INTERVAL)

            try:
                response = self._polling_requester.send(
                    config, stream_slice=job_context,
                )
                body = self._decode_response(response)
                status_value = self._extract_field(body, self._status_field_path)

                if status_value is None:
                    poll_count += 1
                    continue

                status = self._status_mapping.resolve(str(status_value))

                emit(action="job_poll", component="AsyncRetriever",
                     job_id=job_context.get("job_id"),
                     status=status_value, resolved=status,
                     poll=poll_count + 1, elapsed=f"{elapsed:.0f}")

                if status == JobStatus.COMPLETED:
                    return body
                elif status == JobStatus.FAILED:
                    return None

            except Exception:
                pass

            poll_count += 1

            # Safety limit
            if poll_count > 10000:
                return None

    def _download_results(
        self, config: dict, job_context: dict, final_body: dict,
    ) -> Iterator[dict]:
        """Download results from completed job."""
        # Try to extract download URLs
        download_urls = []
        if self._urls_extractor:
            download_urls = self._urls_extractor.extract(
                final_body, config, stream_partition=job_context,
            )

        if self._download_retriever:
            if download_urls:
                # Download from each URL
                for url in download_urls:
                    dl_context = {**job_context, "download_url": url}
                    yield from self._download_retriever.read_records(
                        config, stream_slice=dl_context,
                    )
            else:
                # Use the download retriever with job context
                yield from self._download_retriever.read_records(
                    config, stream_slice=job_context,
                )
        elif self._record_selector:
            # Extract records directly from the polling response
            records = self._record_selector.select(final_body, config)
            yield from records
        else:
            # Last resort: yield the final body as a single record
            if isinstance(final_body, dict):
                yield final_body
            elif isinstance(final_body, list):
                yield from final_body

    def _decode_response(self, response: requests.Response) -> Any:
        """Decode an HTTP response using the configured decoder."""
        try:
            decoded = self._decoder.decode(response)
            if isinstance(decoded, list) and len(decoded) == 1:
                return decoded[0]
            return decoded
        except Exception:
            # Fall back to JSON
            try:
                return response.json()
            except Exception:
                return {"raw": response.text}

    @staticmethod
    def _extract_field(body: Any, field_path: list) -> Any:
        """Extract a value from a nested dict/list using a field path."""
        value = body
        for key in field_path:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, (list, tuple)):
                try:
                    value = value[int(key)]
                except (IndexError, ValueError, TypeError):
                    return None
            else:
                return None
        return value
