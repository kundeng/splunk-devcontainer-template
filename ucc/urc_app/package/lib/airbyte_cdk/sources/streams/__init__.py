#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

# Lazy imports to avoid circular dependency chain
def __getattr__(name):
    if name in ("NO_CURSOR_STATE_KEY", "CheckpointMixin", "IncrementalMixin", "Stream"):
        from .core import NO_CURSOR_STATE_KEY, CheckpointMixin, IncrementalMixin, Stream
        _map = {
            "NO_CURSOR_STATE_KEY": NO_CURSOR_STATE_KEY,
            "CheckpointMixin": CheckpointMixin,
            "IncrementalMixin": IncrementalMixin,
            "Stream": Stream,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["NO_CURSOR_STATE_KEY", "IncrementalMixin", "CheckpointMixin", "Stream"]
