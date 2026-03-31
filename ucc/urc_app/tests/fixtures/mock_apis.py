"""Functions that register `responses` callbacks for common API patterns."""

import json


def register_simple_api(rsps, base_url, path, records):
    """Register a single-page API endpoint returning a list of records."""
    rsps.add(
        rsps.GET,
        f"{base_url}{path}",
        json=records,
        status=200,
    )


def register_paginated_api(rsps, base_url, total_records, page_size):
    """Register offset-based pagination that returns records in pages.

    Returns {"data": [...], "total": N} for each page, driven by
    offset/limit query params.
    """
    all_records = [{"id": i, "name": f"record_{i}"} for i in range(total_records)]

    def paginated_callback(request):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(request.url)
        qs = parse_qs(parsed.query)
        offset = int(qs.get("offset", ["0"])[0])
        limit = int(qs.get("limit", [str(page_size)])[0])
        page = all_records[offset:offset + limit]
        body = {"data": page, "total": total_records}
        return (200, {}, json.dumps(body))

    rsps.add_callback(
        rsps.GET,
        f"{base_url}/paginated",
        callback=paginated_callback,
        content_type="application/json",
    )


def register_rate_limited_api(rsps, base_url, fail_count):
    """Return 429 N times, then 200 with a success body."""
    call_count = {"n": 0}

    def rate_limit_callback(request):
        call_count["n"] += 1
        if call_count["n"] <= fail_count:
            return (429, {"Retry-After": "1"}, json.dumps({"error": "rate limited"}))
        return (200, {}, json.dumps({"data": [{"id": 1}]}))

    rsps.add_callback(
        rsps.GET,
        f"{base_url}/rate-limited",
        callback=rate_limit_callback,
        content_type="application/json",
    )


def register_bulk_job_api(rsps, base_url, poll_count):
    """Register a create/poll/download cycle for bulk jobs.

    - POST /jobs -> 202 with job_id
    - GET /jobs/<id>/status -> "processing" N times, then "complete"
    - GET /jobs/<id>/results -> records
    """
    poll_state = {"n": 0}

    rsps.add(
        rsps.POST,
        f"{base_url}/jobs",
        json={"job_id": "job-123", "status": "accepted"},
        status=202,
    )

    def poll_callback(request):
        poll_state["n"] += 1
        if poll_state["n"] <= poll_count:
            return (200, {}, json.dumps({"job_id": "job-123", "status": "processing"}))
        return (200, {}, json.dumps({"job_id": "job-123", "status": "complete"}))

    rsps.add_callback(
        rsps.GET,
        f"{base_url}/jobs/job-123/status",
        callback=poll_callback,
        content_type="application/json",
    )

    rsps.add(
        rsps.GET,
        f"{base_url}/jobs/job-123/results",
        json=[{"id": 1, "value": "bulk_result"}],
        status=200,
    )


def register_csv_api(rsps, base_url, csv_text):
    """Register a CSV endpoint."""
    rsps.add(
        rsps.GET,
        f"{base_url}/csv",
        body=csv_text,
        status=200,
        content_type="text/csv",
    )


def register_error_sequence(rsps, base_url, status_codes):
    """Return the given status codes in order, then 200 forever after."""
    call_state = {"n": 0}

    def error_callback(request):
        idx = call_state["n"]
        call_state["n"] += 1
        if idx < len(status_codes):
            code = status_codes[idx]
            return (code, {}, json.dumps({"error": f"status {code}"}))
        return (200, {}, json.dumps({"data": "ok"}))

    rsps.add_callback(
        rsps.GET,
        f"{base_url}/errors",
        callback=error_callback,
        content_type="application/json",
    )
