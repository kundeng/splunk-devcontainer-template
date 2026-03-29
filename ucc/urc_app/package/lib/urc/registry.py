# Component registry — maps manifest type names to runtime classes.
# Components self-register via the @component decorator.

from typing import Any, Dict, Type

REGISTRY: Dict[str, Type[Any]] = {}


def component(type_name: str):
    """Decorator that registers a runtime component class by its manifest type name.

    Usage:
        @component("ApiKeyAuthenticator")
        class ApiKeyAuth:
            ...
    """
    def decorator(cls):
        REGISTRY[type_name] = cls
        return cls
    return decorator


def create(component_def: dict, config: dict, **kwargs) -> Any:
    """Create a runtime component from a resolved manifest definition.

    Args:
        component_def: Resolved manifest dict with a 'type' key.
        config: User config dict (for interpolation).
        **kwargs: Additional context passed to the component constructor.

    Returns:
        Instantiated runtime component.
    """
    type_name = component_def.get("type")
    if not type_name:
        raise ValueError(f"Component definition missing 'type' key: {component_def}")

    cls = REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown component type '{type_name}'. "
            f"Registered types: {sorted(REGISTRY.keys())}"
        )

    return cls(component_def, config, **kwargs)
