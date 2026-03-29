#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#


def __getattr__(name):
    _imports = {
        "AirbyteTracedException": ".traced_exception",
        "PrintBuffer": ".print_buffer",
        "is_cloud_environment": ".is_cloud_environment",
    }
    if name == "SchemaInferrer":
        try:
            from .schema_inferrer import SchemaInferrer
            return SchemaInferrer
        except ImportError:
            return None
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AirbyteTracedException", "SchemaInferrer", "is_cloud_environment", "PrintBuffer"]
