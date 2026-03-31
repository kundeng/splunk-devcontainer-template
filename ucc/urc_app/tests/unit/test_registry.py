"""Tests for urc.registry — component registration and creation."""

import pytest

from urc.registry import REGISTRY, component, create


# ── @component decorator ──


def test_component_decorator_registers():
    @component("TestComponent_12345")
    class MyTestComp:
        def __init__(self, definition, config, **kwargs):
            self.definition = definition

    assert "TestComponent_12345" in REGISTRY
    assert REGISTRY["TestComponent_12345"] is MyTestComp

    # Clean up
    del REGISTRY["TestComponent_12345"]


def test_component_decorator_preserves_class():
    @component("PreserveTest_67890")
    class Preserved:
        pass

    assert Preserved.__name__ == "Preserved"
    del REGISTRY["PreserveTest_67890"]


# ── create() ──


def test_create_instantiates_correct_class():
    # Use an existing registered component
    obj = create({"type": "NoAuth"}, {})
    from urc.components.auth import NoAuth
    assert isinstance(obj, NoAuth)


def test_create_passes_definition_and_config():
    @component("_TestCreate_99")
    class Capture:
        def __init__(self, definition, config, **kwargs):
            self.definition = definition
            self.config = config

    definition = {"type": "_TestCreate_99", "param": "value"}
    config = {"key": "val"}
    obj = create(definition, config)
    assert obj.definition == definition
    assert obj.config == config

    del REGISTRY["_TestCreate_99"]


def test_create_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown component type"):
        create({"type": "CompletelyFakeType"}, {})


def test_create_unknown_type_lists_registered():
    with pytest.raises(ValueError) as exc_info:
        create({"type": "FakeType123"}, {})
    # Error message should list registered types
    assert "Registered types" in str(exc_info.value)


def test_create_missing_type_raises():
    with pytest.raises(ValueError, match="missing 'type' key"):
        create({"param": "value"}, {})


# ── Verify known components are registered ──


def test_known_components_registered():
    """Verify that importing urc.components populates the registry."""
    expected = [
        "NoAuth",
        "BearerAuthenticator",
        "BasicHttpAuthenticator",
        "ApiKeyAuthenticator",
        "OAuthAuthenticator",
        "NoPagination",
        "OffsetIncrement",
        "PageIncrement",
        "CursorPagination",
        "DefaultPaginator",
        "JsonDecoder",
        "DpathExtractor",
        "RecordSelector",
        "SimpleRetriever",
        "HttpRequester",
        "DefaultErrorHandler",
        "AddFields",
        "RemoveFields",
        "KeysToLower",
        "KeysToSnakeCase",
        "FlattenFields",
    ]
    for name in expected:
        assert name in REGISTRY, f"{name} not found in REGISTRY"
