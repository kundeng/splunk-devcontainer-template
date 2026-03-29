#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#
# RecordTransformation is depended upon by every class in this module (since it's the abc everything implements). For this reason,
# the order of imports matters i.e: this file must fully import RecordTransformation before importing anything which depends on RecordTransformation
# Otherwise there will be a circular dependency (load order will be init.py --> RemoveFields (which tries to import RecordTransformation) -->
# init.py --> circular dep error, since loading this file causes it to try to import itself down the line.
# so we add the split directive below to tell isort to sort imports while keeping RecordTransformation as the first import
# isort: split


def __getattr__(name):
    _imports = {
        "AddFields": ".add_fields",
        "RecordTransformation": ".transformation",
        "RemoveFields": ".remove_fields",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
