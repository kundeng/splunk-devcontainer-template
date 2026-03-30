# Manifest validation — validate processed manifest dicts against Pydantic models.
# Catches config errors with clear field-path messages BEFORE any HTTP requests.

import logging
from typing import Any, Dict, List, Optional

from urc.models_generated import DeclarativeSource1 as SourceModel

logger = logging.getLogger(__name__)


class ManifestValidationError(Exception):
    """Raised when a manifest fails validation."""

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


def validate_manifest(manifest_dict: dict) -> SourceModel:
    """Validate a processed manifest dict against the Pydantic schema.

    Args:
        manifest_dict: Processed manifest (after resolve_refs + propagate_types).

    Returns:
        Validated SourceModel instance.

    Raises:
        ManifestValidationError: If the manifest is invalid, with field-path details.
    """
    try:
        source = SourceModel.parse_obj(manifest_dict)
        return source
    except Exception as e:
        error_str = str(e)
        errors = []
        if hasattr(e, "errors"):
            try:
                errors = e.errors()
            except Exception:
                pass

        if errors:
            details = []
            for err in errors[:10]:
                loc = " -> ".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "unknown error")
                details.append(f"  {loc}: {msg}")
            detail_str = "\n".join(details)
            raise ManifestValidationError(
                f"Manifest validation failed with {len(errors)} error(s):\n{detail_str}",
                errors=errors,
            ) from e
        else:
            raise ManifestValidationError(f"Manifest validation failed: {error_str}") from e


def validate_stream(stream_dict: dict) -> Any:
    """Validate a single stream definition."""
    from urc.models_generated import DeclarativeStream

    try:
        return DeclarativeStream.parse_obj(stream_dict)
    except Exception as e:
        raise ManifestValidationError(f"Stream validation failed: {e}") from e
