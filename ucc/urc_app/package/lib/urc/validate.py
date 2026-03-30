# Manifest validation — validate processed manifest dicts against Pydantic models.
# Catches config errors with clear field-path messages BEFORE any HTTP requests.

import logging
from typing import Any, Dict, List, Optional

from urc.models_generated import DeclarativeStream

logger = logging.getLogger(__name__)


class ManifestValidationError(Exception):
    """Raised when a manifest fails validation."""

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


class ValidatedManifest:
    """Result of manifest validation — holds validated streams."""

    def __init__(self, streams: List[DeclarativeStream]):
        self.streams = streams


def validate_manifest(manifest_dict: dict) -> ValidatedManifest:
    """Validate a processed manifest dict against the Pydantic schema.

    Validates each stream individually using DeclarativeStream.parse_obj(),
    avoiding the top-level anyOf union issue where pydantic v1 can't
    discriminate between ConditionalStreams and DeclarativeStream.

    Args:
        manifest_dict: Processed manifest (after resolve_refs + propagate_types).

    Returns:
        ValidatedManifest with validated stream instances.

    Raises:
        ManifestValidationError: If any stream is invalid, with field-path details.
    """
    streams_data = manifest_dict.get("streams", [])
    if not streams_data:
        raise ManifestValidationError("Manifest has no streams defined")

    validated_streams = []
    all_errors = []

    for i, stream_dict in enumerate(streams_data):
        stream_name = stream_dict.get("name", f"stream[{i}]")
        try:
            validated = DeclarativeStream.parse_obj(stream_dict)
            validated_streams.append(validated)
        except Exception as e:
            errors = []
            if hasattr(e, "errors"):
                try:
                    errors = e.errors()
                except Exception:
                    pass

            if errors:
                for err in errors[:10]:
                    loc = " -> ".join(str(x) for x in err.get("loc", []))
                    msg = err.get("msg", "unknown error")
                    all_errors.append(f"  stream '{stream_name}' -> {loc}: {msg}")
            else:
                all_errors.append(f"  stream '{stream_name}': {e}")

    if all_errors:
        detail_str = "\n".join(all_errors)
        raise ManifestValidationError(
            f"Manifest validation failed with {len(all_errors)} error(s):\n{detail_str}",
            errors=all_errors,
        )

    return ValidatedManifest(streams=validated_streams)


def validate_stream(stream_dict: dict) -> DeclarativeStream:
    """Validate a single stream definition."""
    try:
        return DeclarativeStream.parse_obj(stream_dict)
    except Exception as e:
        raise ManifestValidationError(f"Stream validation failed: {e}") from e
