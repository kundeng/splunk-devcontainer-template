"""Integration tests: async retriever bulk job lifecycle."""

import json
from unittest.mock import patch, MagicMock

import responses

from urc.components.async_retriever import AsyncRetriever


BASE_URL = "https://mock-api.test"


def _build_async_retriever(config, **overrides):
    """Build an AsyncRetriever from a definition dict."""
    definition = {
        "type": "AsyncRetriever",
        "creation_requester": {
            "type": "HttpRequester",
            "url": f"{BASE_URL}/jobs",
            "http_method": "POST",
        },
        "polling_requester": {
            "type": "HttpRequester",
            "url": f"{BASE_URL}/jobs/{{{{ stream_partition['job_id'] }}}}/status",
            "http_method": "GET",
        },
        "download_retriever": {
            "type": "SimpleRetriever",
            "requester": {
                "type": "HttpRequester",
                "url": f"{BASE_URL}/jobs/{{{{ stream_partition['job_id'] }}}}/results",
                "http_method": "GET",
            },
            "record_selector": {
                "type": "RecordSelector",
                "extractor": {
                    "type": "DpathExtractor",
                    "field_path": [],
                },
            },
        },
        "status_mapping": {
            "completed": ["complete", "completed", "done"],
            "failed": ["failed", "error"],
            "running": ["processing", "running"],
            "pending": ["pending", "accepted", "queued"],
        },
        "status_extractor": {"field_path": ["status"]},
        "job_id_extractor": {"field_path": ["job_id"]},
        "poll_interval_seconds": 0.01,
        "max_poll_time_seconds": 5,
    }
    definition.update(overrides)
    return AsyncRetriever(definition, config)


# ── Happy path ──


@patch("urc.components.async_retriever.time")
def test_happy_path_create_poll_download(mock_time, activate_responses):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()

    rsps = activate_responses

    # Create job
    rsps.add(rsps.POST, f"{BASE_URL}/jobs",
             json={"job_id": "j42", "status": "accepted"}, status=202)

    # Poll: pending → processing → complete
    poll_responses = [
        {"job_id": "j42", "status": "pending"},
        {"job_id": "j42", "status": "processing"},
        {"job_id": "j42", "status": "complete"},
    ]
    for i, body in enumerate(poll_responses):
        rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/status",
                 json=body, status=200)

    # Download results
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/j42/results",
             json=[{"id": 1, "data": "result1"}, {"id": 2, "data": "result2"}],
             status=200)

    config = {"base_url": BASE_URL}
    retriever = _build_async_retriever(config)
    records = list(retriever.read_records(config))

    assert len(records) == 2
    assert records[0]["id"] == 1
    assert records[1]["data"] == "result2"


# ── Failed job ──


@patch("urc.components.async_retriever.time")
def test_failed_job_no_records(mock_time, activate_responses):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()

    rsps = activate_responses

    rsps.add(rsps.POST, f"{BASE_URL}/jobs",
             json={"job_id": "jfail", "status": "accepted"}, status=202)

    rsps.add(rsps.GET, f"{BASE_URL}/jobs/jfail/status",
             json={"job_id": "jfail", "status": "failed",
                    "error": {"message": "query syntax error"}},
             status=200)

    config = {"base_url": BASE_URL}
    retriever = _build_async_retriever(config)
    records = list(retriever.read_records(config))

    assert len(records) == 0


# ── Timeout ──


@patch("urc.components.async_retriever.time")
def test_timeout_no_records(mock_time, activate_responses):
    # Simulate time advancing past max_poll_time
    call_count = {"n": 0}

    def monotonic_side_effect():
        call_count["n"] += 1
        return call_count["n"] * 2.0  # 2s per call, timeout at 5s

    mock_time.monotonic.side_effect = monotonic_side_effect
    mock_time.sleep = MagicMock()

    rsps = activate_responses

    rsps.add(rsps.POST, f"{BASE_URL}/jobs",
             json={"job_id": "jtimeout", "status": "accepted"}, status=202)

    # Always return processing — never completes
    rsps.add(rsps.GET, f"{BASE_URL}/jobs/jtimeout/status",
             json={"job_id": "jtimeout", "status": "processing"},
             status=200)

    config = {"base_url": BASE_URL}
    retriever = _build_async_retriever(config,
                                        max_poll_time_seconds=5)
    records = list(retriever.read_records(config))

    assert len(records) == 0
    # Verify sleep was called (polling happened)
    assert mock_time.sleep.call_count > 0


# ── URL extraction ──


@patch("urc.components.async_retriever.time")
def test_download_url_extraction(mock_time, activate_responses):
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep = MagicMock()

    rsps = activate_responses

    rsps.add(rsps.POST, f"{BASE_URL}/jobs",
             json={"job_id": "jurl", "status": "accepted"}, status=202)

    rsps.add(rsps.GET, f"{BASE_URL}/jobs/jurl/status",
             json={"job_id": "jurl", "status": "complete",
                    "downloadUrl": f"{BASE_URL}/downloads/jurl"},
             status=200)

    rsps.add(rsps.GET, f"{BASE_URL}/downloads/jurl",
             json=[{"id": 99, "data": "from_download_url"}],
             status=200)

    config = {"base_url": BASE_URL}
    retriever = _build_async_retriever(
        config,
        urls_extractor={"field_path": ["downloadUrl"]},
        download_retriever={
            "type": "SimpleRetriever",
            "requester": {
                "type": "HttpRequester",
                "url": "{{ stream_partition['download_url'] }}",
                "http_method": "GET",
            },
            "record_selector": {
                "type": "RecordSelector",
                "extractor": {"type": "DpathExtractor", "field_path": []},
            },
        },
    )

    records = list(retriever.read_records(config))
    assert len(records) == 1
    assert records[0]["data"] == "from_download_url"
