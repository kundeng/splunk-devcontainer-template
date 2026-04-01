# Manifest validation — validate processed manifest dicts against Pydantic models.
# Catches config errors with clear field-path messages BEFORE any HTTP requests.

from typing import Any, Dict, List, Optional

from urc.models import DeclarativeSource1 as SourceModel


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
    from urc.models import DeclarativeStream

    try:
        return DeclarativeStream.parse_obj(stream_dict)
    except Exception as e:
        raise ManifestValidationError(f"Stream validation failed: {e}") from e


# ── Capability detection ──

# Structural types that appear at the top level — not actual runtime components.
_STRUCTURAL_TYPES = {"DeclarativeSource", "DeclarativeStream", "CompositeErrorHandler"}

# JWT algorithm families that require the cryptography package.
_JWT_RSA_EC_ALGORITHMS = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}

# OAuth2 fields that indicate PKCE usage.
_PKCE_FIELDS = {"code_challenge_method", "pkce_enabled"}


def validate_manifest_capabilities(manifest_yaml: str) -> List[Dict[str, Any]]:
    """Validate a manifest for unsupported component types and capability warnings.

    Parses the manifest via process_manifest(), then recursively walks the
    result looking for ``type`` fields.  Each type is checked against the
    component REGISTRY and against known capability gates (JWT with RSA/EC
    algorithms, OAuth2 with PKCE).

    Args:
        manifest_yaml: Raw YAML string of the manifest.

    Returns:
        List of issue dicts, each with keys: path, type, severity, message.
        An empty list means no issues were found.
    """
    from urc.manifest import process_manifest
    import urc.components  # noqa: F401  — ensure @component decorators run
    from urc.registry import REGISTRY

    processed = process_manifest(manifest_yaml)
    issues: List[Dict[str, Any]] = []
    _walk(processed, "", issues, REGISTRY)
    return issues


def _walk(
    node: Any,
    path: str,
    issues: List[Dict[str, Any]],
    registry: Dict[str, Any],
) -> None:
    """Recursively walk a manifest dict, collecting capability issues."""
    if isinstance(node, dict):
        type_name = node.get("type")
        if type_name and type_name not in _STRUCTURAL_TYPES:
            # Unknown type check
            if type_name not in registry:
                issues.append({
                    "path": path or "$",
                    "type": type_name,
                    "severity": "error",
                    "message": f"Unknown component type '{type_name}' — not in registry",
                })

            # JWT with RSA/EC algorithm → needs cryptography
            if type_name == "JwtAuthenticator":
                algorithm = node.get("algorithm", "")
                if algorithm in _JWT_RSA_EC_ALGORITHMS:
                    issues.append({
                        "path": path or "$",
                        "type": type_name,
                        "severity": "warning",
                        "message": (
                            f"JwtAuthenticator with {algorithm} requires the "
                            "'cryptography' package (not bundled by default)"
                        ),
                    })

            # OAuth2 with PKCE fields
            if type_name == "OAuthAuthenticator":
                pkce_fields_present = _PKCE_FIELDS & set(node.keys())
                if pkce_fields_present:
                    issues.append({
                        "path": path or "$",
                        "type": type_name,
                        "severity": "warning",
                        "message": (
                            "OAuthAuthenticator uses PKCE "
                            f"({', '.join(sorted(pkce_fields_present))})"
                        ),
                    })

        # Recurse into child keys
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            _walk(value, child_path, issues, registry)

    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", issues, registry)
