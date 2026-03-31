# Manifest processing — parse YAML, resolve refs, propagate types/parameters.
# Implements Airbyte steps 1-3 without CDK dependencies.

import copy
import re
from typing import Any, Dict, Mapping, Optional, Set

import yaml


# ── Step 1: YAML parsing ──

def parse_manifest(manifest_yaml: str) -> dict:
    """Parse a YAML manifest string into a dict."""
    manifest = yaml.safe_load(manifest_yaml)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a YAML mapping, got {type(manifest).__name__}")
    return manifest


# ── Step 2: Reference resolution ──

_REF_TAG = "$ref"


def resolve_refs(manifest: dict) -> dict:
    """Resolve all $ref references in the manifest.

    References use JSON pointer syntax: "#/definitions/my_auth"
    """
    return _resolve_node(manifest, manifest, set())


def _resolve_node(node: Any, root: Mapping[str, Any], visited: Set[str]) -> Any:
    if isinstance(node, dict):
        evaluated = {
            k: _resolve_node(v, root, visited)
            for k, v in node.items()
            if k != _REF_TAG
        }
        if _REF_TAG in node:
            ref_value = _resolve_node(node[_REF_TAG], root, visited)
            if isinstance(ref_value, dict):
                # Component values take precedence over referenced values
                return {**ref_value, **evaluated}
            return ref_value
        return evaluated
    elif isinstance(node, list):
        return [_resolve_node(v, root, visited) for v in node]
    elif isinstance(node, str) and node.startswith("#/"):
        if node in visited:
            raise ValueError(f"Circular reference detected: {node}")
        visited.add(node)
        result = _resolve_node(_lookup_ref(node, root), root, visited)
        visited.discard(node)
        return result
    return node


def _lookup_ref(ref: str, root: Mapping[str, Any]) -> Any:
    """Follow a JSON pointer like '#/definitions/my_auth'."""
    path = ref.lstrip("#/").split("/")
    current = root
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise ValueError(f"Reference '{ref}' not found at key '{key}'")
    return current


# ── Step 3: Type and parameter propagation ──

# Default type mappings — when a field doesn't specify a type,
# infer it from the parent field name.
DEFAULT_TYPES: Dict[str, str] = {
    "DeclarativeSource.streams": "DeclarativeStream",
    "DeclarativeStream.retriever": "SimpleRetriever",
    "SimpleRetriever.requester": "HttpRequester",
    "SimpleRetriever.record_selector": "RecordSelector",
    "SimpleRetriever.paginator": "NoPagination",
    "RecordSelector.extractor": "DpathExtractor",
    "HttpRequester.error_handler": "DefaultErrorHandler",
    "DefaultPaginator.pagination_strategy": "OffsetIncrement",
}

_PARAMETERS_KEY = "$parameters"


def propagate_types_and_params(
    manifest: dict,
    parent_field: str = "",
    parent_params: Optional[dict] = None,
) -> dict:
    """Recursively propagate types and $parameters through the manifest tree."""
    if parent_params is None:
        parent_params = {}

    result = dict(copy.deepcopy(manifest))

    # Auto-insert missing type from DEFAULT_TYPES
    if "type" not in result and parent_field in DEFAULT_TYPES:
        result["type"] = DEFAULT_TYPES[parent_field]

    # Merge parameters: child params override parent params
    current_params = dict(copy.deepcopy(parent_params))
    component_params = result.pop(_PARAMETERS_KEY, {})
    current_params.update(component_params)

    # Store merged parameters back if non-empty
    if current_params:
        result[_PARAMETERS_KEY] = current_params

    # Recurse into nested dicts
    component_type = result.get("type", "")
    for key, value in list(result.items()):
        if key == _PARAMETERS_KEY:
            continue
        field_id = f"{component_type}.{key}" if component_type else key
        if isinstance(value, dict):
            result[key] = propagate_types_and_params(value, field_id, current_params)
        elif isinstance(value, list):
            result[key] = [
                propagate_types_and_params(item, field_id, current_params)
                if isinstance(item, dict) else item
                for item in value
            ]

    return result


# ── Combined pipeline ──

def process_manifest(manifest_yaml: str) -> dict:
    """Full manifest processing pipeline: parse → resolve refs → propagate types."""
    raw = parse_manifest(manifest_yaml)
    resolved = resolve_refs(raw)
    propagated = propagate_types_and_params(resolved)
    return propagated
