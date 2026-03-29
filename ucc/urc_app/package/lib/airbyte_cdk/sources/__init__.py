#
# Copyright (c) 2021 Airbyte, Inc., all rights reserved.
#

import dpath.options

# Lazy imports to avoid circular dependency chain
# (types.py is in this package and importing it triggers __init__.py)


def __getattr__(name):
    if name == "AbstractSource":
        from .abstract_source import AbstractSource
        return AbstractSource
    elif name == "Source":
        from .source import Source
        return Source
    elif name == "BaseConfig":
        from .config import BaseConfig
        return BaseConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# As part of the CDK sources, we do not control what the APIs return and it is possible that a key is empty.
dpath.options.ALLOW_EMPTY_STRING_KEYS = True

__all__ = [
    "AbstractSource",
    "BaseConfig",
    "Source",
]
