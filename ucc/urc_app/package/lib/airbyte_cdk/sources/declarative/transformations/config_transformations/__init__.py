#
# Copyright (c) 2025 Airbyte, Inc., all rights reserved.
#


def __getattr__(name):
    _imports = {
        "ConfigAddFields": ".add_fields",
        "ConfigRemapField": ".remap_field",
        "ConfigRemoveFields": ".remove_fields",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
