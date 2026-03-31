"""Tests for urc.manifest — YAML parsing, ref resolution, type propagation."""

from urc.manifest import (
    parse_manifest,
    resolve_refs,
    propagate_types_and_params,
    process_manifest,
)
import pytest


# ── parse_manifest ──


def test_parse_simple_manifest():
    yaml_str = "type: DeclarativeSource\nstreams: []"
    result = parse_manifest(yaml_str)
    assert result["type"] == "DeclarativeSource"
    assert result["streams"] == []


def test_parse_invalid_manifest():
    with pytest.raises(ValueError, match="YAML mapping"):
        parse_manifest("- just\n- a\n- list")


# ── resolve_refs ──


def test_resolve_no_refs():
    manifest = {"type": "DeclarativeSource", "streams": []}
    result = resolve_refs(manifest)
    assert result == {"type": "DeclarativeSource", "streams": []}


def test_resolve_one_level_ref():
    manifest = {
        "definitions": {
            "my_auth": {"type": "BearerAuthenticator", "api_token": "tok"},
        },
        "streams": [
            {
                "type": "DeclarativeStream",
                "retriever": {
                    "authenticator": {"$ref": "#/definitions/my_auth"},
                },
            }
        ],
    }
    result = resolve_refs(manifest)
    auth = result["streams"][0]["retriever"]["authenticator"]
    assert auth["type"] == "BearerAuthenticator"
    assert auth["api_token"] == "tok"


def test_resolve_nested_ref():
    manifest = {
        "definitions": {
            "base_requester": {"type": "HttpRequester", "url": "https://api.test"},
            "my_retriever": {
                "type": "SimpleRetriever",
                "requester": {"$ref": "#/definitions/base_requester"},
            },
        },
        "streams": [
            {
                "retriever": {"$ref": "#/definitions/my_retriever"},
            }
        ],
    }
    result = resolve_refs(manifest)
    requester = result["streams"][0]["retriever"]["requester"]
    assert requester["type"] == "HttpRequester"
    assert requester["url"] == "https://api.test"


def test_resolve_ref_with_override():
    manifest = {
        "definitions": {
            "base_auth": {"type": "BearerAuthenticator", "api_token": "default"},
        },
        "streams": [
            {
                "auth": {
                    "$ref": "#/definitions/base_auth",
                    "api_token": "overridden",
                },
            }
        ],
    }
    result = resolve_refs(manifest)
    # Component values take precedence over referenced values
    assert result["streams"][0]["auth"]["api_token"] == "overridden"
    assert result["streams"][0]["auth"]["type"] == "BearerAuthenticator"


def test_resolve_missing_ref():
    manifest = {
        "streams": [{"auth": {"$ref": "#/definitions/nonexistent"}}],
    }
    with pytest.raises(ValueError, match="not found"):
        resolve_refs(manifest)


def test_resolve_circular_ref():
    manifest = {
        "definitions": {
            "a": {"$ref": "#/definitions/b"},
            "b": {"$ref": "#/definitions/a"},
        },
        "value": {"$ref": "#/definitions/a"},
    }
    with pytest.raises(ValueError, match="Circular"):
        resolve_refs(manifest)


# ── propagate_types_and_params ──


def test_propagate_default_types():
    manifest = {
        "type": "DeclarativeSource",
        "streams": [
            {
                "type": "DeclarativeStream",
                "retriever": {
                    # No type specified — should get SimpleRetriever
                },
            }
        ],
    }
    result = propagate_types_and_params(manifest)
    assert result["streams"][0]["retriever"]["type"] == "SimpleRetriever"


def test_propagate_parameters():
    manifest = {
        "type": "DeclarativeSource",
        "$parameters": {"base_url": "https://api.test"},
        "streams": [
            {
                "type": "DeclarativeStream",
                "name": "test",
            }
        ],
    }
    result = propagate_types_and_params(manifest)
    # Parameters should propagate to children
    assert result["streams"][0]["$parameters"]["base_url"] == "https://api.test"


def test_propagate_child_params_override_parent():
    manifest = {
        "type": "DeclarativeSource",
        "$parameters": {"base_url": "https://parent.test"},
        "streams": [
            {
                "type": "DeclarativeStream",
                "$parameters": {"base_url": "https://child.test"},
            }
        ],
    }
    result = propagate_types_and_params(manifest)
    assert result["streams"][0]["$parameters"]["base_url"] == "https://child.test"


# ── process_manifest (full pipeline) ──


def test_process_manifest_full_pipeline():
    yaml_str = """
type: DeclarativeSource
definitions:
  my_auth:
    type: BearerAuthenticator
    api_token: "tok"
streams:
  - type: DeclarativeStream
    name: test_stream
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://api.test"
        authenticator:
          $ref: "#/definitions/my_auth"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
"""
    result = process_manifest(yaml_str)
    auth = result["streams"][0]["retriever"]["requester"]["authenticator"]
    assert auth["type"] == "BearerAuthenticator"
    assert auth["api_token"] == "tok"
