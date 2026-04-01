"""Tests for urc.recovery — centralized error recovery policy."""

import pytest

from urc.recovery import (
    SkipRecord,
    SkipComponent,
    RECOVERY_POLICY,
    recovery_error_hook,
)


# ── Policy lookup ──


def test_policy_skip_for_csv_decoder():
    assert RECOVERY_POLICY[("CsvDecoder", "decode")] == "skip"


def test_policy_raise_for_http_requester():
    assert RECOVERY_POLICY[("HttpRequester", "send")] == "raise"


def test_policy_skip_for_transforms():
    assert RECOVERY_POLICY[("AddFields", "transform")] == "skip"
    assert RECOVERY_POLICY[("KeysReplace", "transform")] == "skip"
    assert RECOVERY_POLICY[("SchemaNormalization", "transform")] == "skip"


def test_unknown_defaults_to_raise():
    # Not in RECOVERY_POLICY → should default to "raise"
    key = ("UnknownComponent", "unknown_method")
    assert key not in RECOVERY_POLICY


# ── recovery_error_hook ──


def test_recovery_hook_raises_skip_record_for_skip_policy():
    exc = ValueError("bad data")
    with pytest.raises(SkipRecord):
        recovery_error_hook("CsvDecoder", "decode", exc, 0.01, {})


def test_recovery_hook_does_nothing_for_raise_policy():
    exc = ConnectionError("timeout")
    # Should NOT raise SkipRecord — just returns (caller re-raises the original)
    recovery_error_hook("HttpRequester", "send", exc, 0.5, {})


def test_recovery_hook_does_nothing_for_unknown():
    exc = RuntimeError("unexpected")
    # Unknown component → default "raise" → does nothing
    recovery_error_hook("UnknownThing", "method", exc, 0.0, {})


# ── Exception types ──


def test_skip_record_is_exception():
    assert issubclass(SkipRecord, Exception)


def test_skip_component_is_exception():
    assert issubclass(SkipComponent, Exception)


def test_skip_record_preserves_message():
    exc = SkipRecord("bad row 42")
    assert str(exc) == "bad row 42"


# ── Lifecycle hooks ──


def test_check_disabled_raises_skip_component():
    from urc.lifecycle import check_disabled

    with pytest.raises(SkipComponent, match="disabled"):
        check_disabled("Comp", "method", {"config": {"_disabled": True}})


def test_check_disabled_no_op_when_not_disabled():
    from urc.lifecycle import check_disabled
    # Should not raise
    check_disabled("Comp", "method", {"config": {}})
    check_disabled("Comp", "method", {"config": {"_disabled": False}})
    check_disabled("Comp", "method", {})


def test_check_run_once_raises_when_completed():
    from urc.lifecycle import check_run_once

    with pytest.raises(SkipComponent, match="run-once"):
        check_run_once("Comp", "method", {
            "config": {"_run_once": True, "_last_run_completed": True}
        })


def test_check_run_once_no_op_when_not_completed():
    from urc.lifecycle import check_run_once
    check_run_once("Comp", "method", {
        "config": {"_run_once": True, "_last_run_completed": False}
    })


def test_check_config_change_sets_flag():
    from urc.lifecycle import check_config_change
    config = {"_config_changed": True}
    check_config_change("Comp", "method", {"config": config})
    assert config.get("_needs_rebuild") is True


def test_check_config_change_no_flag_when_unchanged():
    from urc.lifecycle import check_config_change
    config = {}
    check_config_change("Comp", "method", {"config": config})
    assert "_needs_rebuild" not in config
