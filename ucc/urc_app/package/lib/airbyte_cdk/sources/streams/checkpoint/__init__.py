# Copyright (c) 2024 Airbyte, Inc., all rights reserved.


def __getattr__(name):
    _imports = {
        "Cursor": ".cursor",
        "ResumableFullRefreshCursor": ".resumable_full_refresh_cursor",
        "CheckpointMode": ".checkpoint_reader",
        "CheckpointReader": ".checkpoint_reader",
        "CursorBasedCheckpointReader": ".checkpoint_reader",
        "FullRefreshCheckpointReader": ".checkpoint_reader",
        "IncrementalCheckpointReader": ".checkpoint_reader",
        "LegacyCursorBasedCheckpointReader": ".checkpoint_reader",
        "ResumableFullRefreshCheckpointReader": ".checkpoint_reader",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CheckpointMode",
    "CheckpointReader",
    "Cursor",
    "CursorBasedCheckpointReader",
    "FullRefreshCheckpointReader",
    "IncrementalCheckpointReader",
    "LegacyCursorBasedCheckpointReader",
    "ResumableFullRefreshCheckpointReader",
    "ResumableFullRefreshCursor",
]
