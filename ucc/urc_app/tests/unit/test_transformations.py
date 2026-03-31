"""Tests for urc.components.transformations — record transformations."""

import pytest

from urc.components.transformations import (
    AddFields,
    RemoveFields,
    KeysToLower,
    KeysToSnakeCase,
    FlattenFields,
)


# ── AddFields ──


def test_add_fields_simple_value():
    t = AddFields(
        {"fields": [{"path": ["source"], "value": "api"}]},
        {},
    )
    record = {"id": 1}
    result = t.transform(record, {})
    assert result["source"] == "api"
    assert result["id"] == 1


def test_add_fields_jinja_template():
    config = {"env": "prod"}
    t = AddFields(
        {"fields": [{"path": ["env"], "value": "{{ config['env'] }}"}]},
        config,
    )
    result = t.transform({"id": 1}, config)
    assert result["env"] == "prod"


def test_add_fields_nested_path():
    t = AddFields(
        {"fields": [{"path": ["meta", "tag"], "value": "test"}]},
        {},
    )
    result = t.transform({"id": 1}, {})
    assert result["meta"]["tag"] == "test"


def test_add_fields_with_condition_true():
    config = {}
    t = AddFields(
        {
            "fields": [{"path": ["flagged"], "value": "yes"}],
            "condition": "{{ record['status'] == 'active' }}",
        },
        config,
    )
    result = t.transform({"status": "active"}, config)
    assert result["flagged"] == "yes"


def test_add_fields_with_condition_false():
    config = {}
    t = AddFields(
        {
            "fields": [{"path": ["flagged"], "value": "yes"}],
            "condition": "{{ record['status'] == 'active' }}",
        },
        config,
    )
    result = t.transform({"status": "inactive"}, config)
    assert "flagged" not in result


# ── RemoveFields ──


def test_remove_fields_simple():
    t = RemoveFields({"field_pointers": [["secret"]]}, {})
    result = t.transform({"id": 1, "secret": "hidden"}, {})
    assert "secret" not in result
    assert result["id"] == 1


def test_remove_fields_nested():
    t = RemoveFields({"field_pointers": [["meta", "internal"]]}, {})
    record = {"id": 1, "meta": {"internal": "x", "public": "y"}}
    result = t.transform(record, {})
    assert "internal" not in result["meta"]
    assert result["meta"]["public"] == "y"


def test_remove_fields_missing_path_no_error():
    t = RemoveFields({"field_pointers": [["nonexistent"]]}, {})
    record = {"id": 1}
    result = t.transform(record, {})
    assert result == {"id": 1}


def test_remove_fields_missing_nested_no_error():
    t = RemoveFields({"field_pointers": [["a", "b", "c"]]}, {})
    record = {"id": 1}
    result = t.transform(record, {})
    assert result == {"id": 1}


# ── KeysToLower ──


def test_keys_to_lower_basic():
    t = KeysToLower({}, {})
    result = t.transform({"Name": "Alice", "AGE": 30}, {})
    assert result == {"name": "Alice", "age": 30}


def test_keys_to_lower_nested():
    t = KeysToLower({}, {})
    result = t.transform({"Outer": {"InnerKey": "val"}}, {})
    assert result == {"outer": {"innerkey": "val"}}


def test_keys_to_lower_already_lowercase():
    t = KeysToLower({}, {})
    record = {"name": "Alice", "age": 30}
    result = t.transform(record, {})
    assert result == {"name": "Alice", "age": 30}


# ── KeysToSnakeCase ──


def test_keys_to_snake_camel_case():
    t = KeysToSnakeCase({}, {})
    result = t.transform({"firstName": "Alice", "lastName": "Smith"}, {})
    assert result == {"first_name": "Alice", "last_name": "Smith"}


def test_keys_to_snake_pascal_case():
    t = KeysToSnakeCase({}, {})
    result = t.transform({"FirstName": "Alice"}, {})
    assert result == {"first_name": "Alice"}


def test_keys_to_snake_nested():
    t = KeysToSnakeCase({}, {})
    result = t.transform(
        {"userData": {"firstName": "Alice", "contactInfo": {"phoneNumber": "555"}}},
        {},
    )
    assert result == {
        "user_data": {
            "first_name": "Alice",
            "contact_info": {"phone_number": "555"},
        }
    }


def test_keys_to_snake_acronyms():
    t = KeysToSnakeCase({}, {})
    result = t.transform({"HTTPSEnabled": True, "xmlParser": "lxml"}, {})
    assert "https_enabled" in result or "h_t_t_p_s_enabled" in result
    assert result.get("xml_parser") == "lxml"


# ── FlattenFields ──


def test_flatten_nested_objects():
    t = FlattenFields({}, {})
    result = t.transform({"a": {"b": 1, "c": {"d": 2}}}, {})
    assert result == {"a.b": 1, "a.c.d": 2}


def test_flatten_with_lists():
    t = FlattenFields({"flatten_lists": True}, {})
    result = t.transform({"items": [10, 20]}, {})
    assert result == {"items.0": 10, "items.1": 20}


def test_flatten_custom_separator():
    t = FlattenFields({"separator": "_"}, {})
    result = t.transform({"a": {"b": 1}}, {})
    assert result == {"a_b": 1}


def test_flatten_preserves_scalars():
    t = FlattenFields({}, {})
    result = t.transform({"x": 1, "y": "hello"}, {})
    assert result == {"x": 1, "y": "hello"}


# ── DpathFlattenFields ──


def test_dpath_flatten_fields_basic():
    from urc.components.transformations import DpathFlattenFields
    t = DpathFlattenFields({"field_path": ["metadata"], "delete_origin": True}, {})
    record = {"id": 1, "metadata": {"source": "api", "version": 2}}
    result = t.transform(record, {})
    assert result["metadata.source"] == "api"
    assert result["metadata.version"] == 2
    assert "metadata" not in result  # deleted origin


def test_dpath_flatten_fields_keep_origin():
    from urc.components.transformations import DpathFlattenFields
    t = DpathFlattenFields({"field_path": ["data"], "delete_origin": False}, {})
    record = {"id": 1, "data": {"x": 10}}
    result = t.transform(record, {})
    assert result["data.x"] == 10
    assert result["data"] == {"x": 10}  # kept


def test_dpath_flatten_fields_missing_path():
    from urc.components.transformations import DpathFlattenFields
    t = DpathFlattenFields({"field_path": ["nonexistent"]}, {})
    record = {"id": 1, "name": "test"}
    result = t.transform(record, {})
    assert result == {"id": 1, "name": "test"}  # unchanged


# ── KeysReplace ──


def test_keys_replace_single():
    from urc.components.transformations import KeysReplace
    t = KeysReplace({"replacements": [{"old": ".", "new": "_"}]}, {})
    record = {"user.name": "Alice", "user.email": "alice@test.com"}
    result = t.transform(record, {})
    assert "user_name" in result
    assert "user_email" in result


def test_keys_replace_multiple():
    from urc.components.transformations import KeysReplace
    t = KeysReplace({"replacements": [{"old": "-", "new": "_"}, {"old": " ", "new": "_"}]}, {})
    record = {"first-name": "Bob", "last name": "Smith"}
    result = t.transform(record, {})
    assert "first_name" in result
    assert "last_name" in result


def test_keys_replace_recursive():
    from urc.components.transformations import KeysReplace
    t = KeysReplace({"replacements": [{"old": ".", "new": "_"}]}, {})
    record = {"outer.key": {"inner.key": "value"}}
    result = t.transform(record, {})
    assert "outer_key" in result
    assert "inner_key" in result["outer_key"]


# ── SchemaNormalization ──


def test_schema_normalization_basic():
    from urc.components.transformations import SchemaNormalization
    t = SchemaNormalization({}, {})
    record = {"UserName": "Alice", "E-Mail": "alice@test.com", "Phone Number": "555"}
    result = t.transform(record, {})
    assert "username" in result
    assert "e_mail" in result
    assert "phone_number" in result


def test_schema_normalization_special_chars():
    from urc.components.transformations import SchemaNormalization
    t = SchemaNormalization({}, {})
    record = {"__leading__": 1, "trailing___": 2, "multi...dots": 3}
    result = t.transform(record, {})
    # Should strip leading/trailing underscores, collapse multiples
    for key in result:
        assert not key.startswith("_")
        assert not key.endswith("_")
        assert "__" not in key


def test_schema_normalization_recursive():
    from urc.components.transformations import SchemaNormalization
    t = SchemaNormalization({}, {})
    record = {"OuterKey": {"InnerKey": "value"}}
    result = t.transform(record, {})
    assert "outerkey" in result
    assert "innerkey" in result["outerkey"]
