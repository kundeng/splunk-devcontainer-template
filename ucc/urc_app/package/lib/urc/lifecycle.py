# Lifecycle hooks — before-hooks for modular input lifecycle events.
#
# Registered via instrumentation.register_hook("before", hook).
# Each hook inspects context['config'] for lifecycle flags and raises
# SkipComponent to cleanly stop collection.

from urc.recovery import SkipComponent


def check_disabled(component_type: str, method_name: str, context: dict) -> None:
    """Skip collection if input was disabled mid-run.

    The modular input handler sets config['_disabled'] = True when
    Splunk signals the input should stop.
    """
    config = context.get('config', {})
    if isinstance(config, dict) and config.get('_disabled'):
        raise SkipComponent("input disabled")


def check_run_once(component_type: str, method_name: str, context: dict) -> None:
    """Skip if input is in run-once mode and already completed.

    The modular input handler sets config['_run_once'] = True for
    one-shot inputs, and config['_last_run_completed'] = True after
    the first successful collection.
    """
    config = context.get('config', {})
    if isinstance(config, dict):
        if config.get('_run_once') and config.get('_last_run_completed'):
            raise SkipComponent("run-once already completed")


def check_config_change(component_type: str, method_name: str, context: dict) -> None:
    """Flag for component rebuild if config hash changed.

    The modular input handler compares config hashes between cycles
    and sets config['_config_changed'] = True if they differ.
    The engine checks config['_needs_rebuild'] after this hook runs.
    """
    config = context.get('config', {})
    if isinstance(config, dict) and config.get('_config_changed'):
        config['_needs_rebuild'] = True
