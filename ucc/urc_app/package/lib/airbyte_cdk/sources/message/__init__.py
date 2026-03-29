#
# Copyright (c) 2021 Airbyte, Inc., all rights reserved.
#


def __getattr__(name):
    _imports = {
        "InMemoryMessageRepository": ".repository",
        "LogAppenderMessageRepositoryDecorator": ".repository",
        "LogMessage": ".repository",
        "MessageRepository": ".repository",
        "NoopMessageRepository": ".repository",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "InMemoryMessageRepository",
    "LogAppenderMessageRepositoryDecorator",
    "LogMessage",
    "MessageRepository",
    "NoopMessageRepository",
]
