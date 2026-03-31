"""Tests for urc.interpolation — eval_string and eval_dict."""

import datetime
from unittest.mock import patch

from urc.interpolation import eval_string, eval_dict


# ── eval_string basics ──


def test_simple_config_access(mock_config):
    result = eval_string("{{ config['api_key'] }}", mock_config)
    assert result == "test-key-123"


def test_nested_config_access():
    config = {"nested": {"key": "deep_value"}}
    result = eval_string("{{ config['nested']['key'] }}", config)
    assert result == "deep_value"


def test_non_template_passthrough(mock_config):
    result = eval_string("plain string", mock_config)
    assert result == "plain string"


def test_non_string_passthrough(mock_config):
    result = eval_string(42, mock_config)
    assert result == 42


def test_undefined_var_returns_original(mock_config):
    template = "{{ config['missing'] }}"
    result = eval_string(template, mock_config)
    # On error (StrictUndefined), returns original template
    assert result == template


def test_literal_coercion_int(mock_config):
    result = eval_string("{{ 42 }}", mock_config)
    assert result == 42
    assert isinstance(result, int)


def test_literal_coercion_bool_true(mock_config):
    result = eval_string("{{ true }}", mock_config)
    assert result is True


def test_literal_coercion_bool_false(mock_config):
    result = eval_string("{{ false }}", mock_config)
    assert result is False


def test_literal_coercion_none(mock_config):
    result = eval_string("{{ none }}", mock_config)
    assert result is None


def test_literal_coercion_float(mock_config):
    result = eval_string("{{ 3.14 }}", mock_config)
    assert result == 3.14
    assert isinstance(result, float)


# ── Filters ──


def test_base64encode_filter(mock_config):
    result = eval_string("{{ 'hello' | base64encode }}", mock_config)
    import base64
    assert result == base64.b64encode(b"hello").decode()


def test_hash_md5_filter(mock_config):
    result = eval_string("{{ 'hello' | hash('md5') }}", mock_config)
    import hashlib
    expected = hashlib.md5(b"hello").hexdigest()
    assert result == expected


def test_regex_search_filter(mock_config):
    result = eval_string(
        "{{ 'order-12345-suffix' | regex_search('order-(\\d+)') }}",
        mock_config,
    )
    # regex_search returns "12345" which _literal_eval coerces to int
    assert result == 12345


def test_regex_replace_filter(mock_config):
    result = eval_string(
        "{{ 'hello world' | regex_replace('world', 'pytest') }}",
        mock_config,
    )
    assert result == "hello pytest"


# ── Macros ──


def test_now_utc_returns_datetime(mock_config):
    result = eval_string("{{ now_utc() }}", mock_config)
    # Result is rendered to string and then returned as-is (not a literal)
    assert isinstance(result, str)
    # Should contain a date-like substring
    assert "20" in result  # year prefix


def test_timestamp_converts_iso_to_epoch(mock_config):
    result = eval_string("{{ timestamp('2024-01-01T00:00:00+00:00') }}", mock_config)
    assert isinstance(result, int)
    assert result == 1704067200


def test_day_delta_returns_string(mock_config):
    result = eval_string("{{ day_delta(0) }}", mock_config)
    assert isinstance(result, str)
    # Should be an ISO-ish timestamp string
    assert "T" in result


def test_max_macro(mock_config):
    result = eval_string("{{ max(1, 5, 3) }}", mock_config)
    assert result == 5


# ── eval_dict ──


def test_eval_dict_nested_structure(mock_config):
    obj = {
        "url": "https://api.test/{{ config['api_key'] }}",
        "headers": {"Auth": "Bearer {{ config['api_key'] }}"},
        "tags": ["{{ config['username'] }}", "static"],
        "count": 42,
    }
    result = eval_dict(obj, mock_config)
    assert result["url"] == "https://api.test/test-key-123"
    assert result["headers"]["Auth"] == "Bearer test-key-123"
    assert result["tags"] == ["testuser", "static"]
    assert result["count"] == 42


def test_eval_dict_passthrough_non_template():
    result = eval_dict({"key": "plain"}, {})
    assert result == {"key": "plain"}


def test_eval_dict_with_list():
    config = {"x": "val"}
    result = eval_dict(["{{ config['x'] }}", "literal"], config)
    assert result == ["val", "literal"]


# ── Context variables ──


def test_stream_partition_context(mock_config):
    result = eval_string(
        "{{ stream_partition['repo'] }}",
        mock_config,
        stream_partition={"repo": "my-repo"},
    )
    assert result == "my-repo"


def test_multiple_context_vars(mock_config):
    result = eval_string(
        "{{ config['api_key'] }}-{{ stream_partition['id'] }}",
        mock_config,
        stream_partition={"id": "42"},
    )
    assert result == "test-key-123-42"
