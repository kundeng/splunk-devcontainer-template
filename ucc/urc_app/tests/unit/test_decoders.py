"""Tests for urc.components.decoders — response decoders."""

import json

import requests

from urc.components.decoders import JsonDecoder, JsonlDecoder, CsvDecoder


def _make_response(body, content_type="application/json", status=200):
    """Build a mock requests.Response with the given body content."""
    resp = requests.models.Response()
    resp.status_code = status
    if isinstance(body, (dict, list)):
        resp._content = json.dumps(body).encode()
    elif isinstance(body, str):
        resp._content = body.encode()
    elif isinstance(body, bytes):
        resp._content = body
    resp.headers["Content-Type"] = content_type
    resp.encoding = "utf-8"
    return resp


# ── JsonDecoder ──


def test_json_decoder_array():
    decoder = JsonDecoder({}, {})
    resp = _make_response([{"id": 1}, {"id": 2}])
    result = decoder.decode(resp)
    assert result == [{"id": 1}, {"id": 2}]


def test_json_decoder_object():
    decoder = JsonDecoder({}, {})
    resp = _make_response({"key": "value"})
    result = decoder.decode(resp)
    assert result == [{"key": "value"}]


def test_json_decoder_empty_object():
    decoder = JsonDecoder({}, {})
    resp = _make_response({})
    result = decoder.decode(resp)
    assert result == [{}]


def test_json_decoder_scalar():
    decoder = JsonDecoder({}, {})
    resp = _make_response('"hello"')
    result = decoder.decode(resp)
    assert result == [{"record": "hello"}]


# ── JsonlDecoder ──


def test_jsonl_decoder_multiple_lines():
    decoder = JsonlDecoder({}, {})
    body = '{"id": 1}\n{"id": 2}\n{"id": 3}\n'
    resp = _make_response(body, content_type="application/x-ndjson")
    result = decoder.decode(resp)
    assert len(result) == 3
    assert result[0] == {"id": 1}


def test_jsonl_decoder_skips_empty_lines():
    decoder = JsonlDecoder({}, {})
    body = '{"id": 1}\n\n{"id": 2}\n'
    resp = _make_response(body)
    result = decoder.decode(resp)
    assert len(result) == 2


def test_jsonl_decoder_skips_invalid_lines():
    decoder = JsonlDecoder({}, {})
    body = '{"id": 1}\nnot json\n{"id": 2}\n'
    resp = _make_response(body)
    result = decoder.decode(resp)
    assert len(result) == 2


# ── CsvDecoder ──


def test_csv_decoder_basic():
    decoder = CsvDecoder({}, {})
    csv_text = "id,name\n1,Alice\n2,Bob\n"
    resp = _make_response(csv_text, content_type="text/csv")
    result = decoder.decode(resp)
    assert len(result) == 2
    assert result[0] == {"id": "1", "name": "Alice"}
    assert result[1] == {"id": "2", "name": "Bob"}


def test_csv_decoder_custom_delimiter():
    decoder = CsvDecoder({"delimiter": ";"}, {})
    csv_text = "id;name\n1;Alice\n"
    resp = _make_response(csv_text, content_type="text/csv")
    result = decoder.decode(resp)
    assert result == [{"id": "1", "name": "Alice"}]


def test_csv_decoder_empty():
    decoder = CsvDecoder({}, {})
    csv_text = "id,name\n"
    resp = _make_response(csv_text, content_type="text/csv")
    result = decoder.decode(resp)
    assert result == []


# Placeholder: ZipfileDecoder not yet implemented
# TODO: Add test_zipfile_decoder_* when ZipfileDecoder is added
