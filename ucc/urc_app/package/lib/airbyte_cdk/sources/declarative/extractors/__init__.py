#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#


def __getattr__(name):
    _imports = {
        "DpathExtractor": ".dpath_extractor",
        "HttpSelector": ".http_selector",
        "RecordFilter": ".record_filter",
        "RecordSelector": ".record_selector",
        "ResponseToFileExtractor": ".response_to_file_extractor",
        "TypeTransformer": ".type_transformer",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TypeTransformer",
    "HttpSelector",
    "DpathExtractor",
    "RecordFilter",
    "RecordSelector",
    "ResponseToFileExtractor",
]
