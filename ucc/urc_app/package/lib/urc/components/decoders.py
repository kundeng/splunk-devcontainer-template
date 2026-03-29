# Response decoders — parse HTTP responses into lists of record dicts.

import csv
import gzip
import io
import json
import logging
from typing import List

import requests

from urc.registry import component

try:
    import xmltodict
except ImportError:
    xmltodict = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@component("JsonDecoder")
class JsonDecoder:
    """Parse response body as JSON."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def decode(self, response: requests.Response) -> List[dict]:
        body = response.json()
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            return [body]
        return [{"record": body}]


@component("JsonlDecoder")
class JsonlDecoder:
    """Parse newline-delimited JSON (one JSON object per line)."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def decode(self, response: requests.Response) -> List[dict]:
        records: List[dict] = []
        for line in response.text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line: %s", line[:200])
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"record": obj})
        return records


@component("CsvDecoder")
class CsvDecoder:
    """Parse CSV response into a list of dicts (one per row)."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._delimiter = definition.get("delimiter", ",")
        self._encoding = definition.get("encoding", "utf-8")

    def decode(self, response: requests.Response) -> List[dict]:
        text = response.text
        reader = csv.DictReader(
            io.StringIO(text),
            delimiter=self._delimiter,
        )
        return [dict(row) for row in reader]


@component("XmlDecoder")
class XmlDecoder:
    """Parse XML response using xmltodict."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        if xmltodict is None:
            raise ImportError(
                "xmltodict is required for XmlDecoder but is not installed"
            )

    def decode(self, response: requests.Response) -> List[dict]:
        parsed = xmltodict.parse(response.text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return [{"record": parsed}]


@component("GzipDecoder")
class GzipDecoder:
    """Decompress a gzip response, then delegate to an inner decoder."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        inner_def = definition.get("decoder", {"type": "JsonDecoder"})
        self._inner = create_component(inner_def, config)

    def decode(self, response: requests.Response) -> List[dict]:
        decompressed = gzip.decompress(response.content)
        # Build a lightweight stand-in response so the inner decoder can
        # use .text and .json() as usual.
        inner_resp = requests.models.Response()
        inner_resp.status_code = response.status_code
        inner_resp.headers.update(response.headers)
        inner_resp._content = decompressed  # type: ignore[attr-defined]
        inner_resp.encoding = response.encoding or "utf-8"
        return self._inner.decode(inner_resp)


@component("IterableDecoder")
class IterableDecoder:
    """Parse newline-separated strings, wrapping each in a dict."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        pass

    def decode(self, response: requests.Response) -> List[dict]:
        records: List[dict] = []
        for line in response.text.split("\n"):
            if line:
                records.append({"record": line})
        return records
