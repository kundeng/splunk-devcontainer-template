"""Tests for urc.instrumentation — AOP method wrapping."""

from unittest.mock import MagicMock

from urc.instrumentation import (
    instrument,
    register_hook,
    clear_hooks,
    set_enabled,
    _bind_context,
)
from urc.recovery import SkipRecord, SkipComponent


class DummyComponent:
    """Test component with public and private methods."""

    def __init__(self):
        self.call_log = []

    def process(self, data, config=None):
        self.call_log.append(("process", data))
        return {"result": data}

    def transform(self, record, config=None, **context):
        return record

    def _private(self):
        return "private"

    @property
    def name(self):
        return "dummy"


def setup_function():
    """Clear hooks before each test."""
    clear_hooks()
    set_enabled(True)


# ── instrument() wraps public methods ──


def test_instrument_wraps_public_methods():
    comp = DummyComponent()
    instrument(comp, "DummyComponent")

    # Method still works
    result = comp.process("hello")
    assert result == {"result": "hello"}
    assert comp.call_log == [("process", "hello")]


def test_instrument_skips_private_methods():
    comp = DummyComponent()
    original_private = comp._private
    instrument(comp, "DummyComponent")

    # _private should not be wrapped
    assert comp._private() == "private"


def test_isinstance_still_works():
    comp = DummyComponent()
    instrument(comp, "DummyComponent")

    # Not a proxy — isinstance works
    assert isinstance(comp, DummyComponent)


def test_instrument_disabled():
    set_enabled(False)
    comp = DummyComponent()

    hook = MagicMock()
    register_hook("after", hook)

    instrument(comp, "DummyComponent")
    comp.process("test")

    # Hook should NOT fire when disabled
    hook.assert_not_called()


# ── Hooks fire in correct order ──


def test_before_hook_fires():
    calls = []
    register_hook("before", lambda ct, mn, ctx: calls.append(("before", ct, mn)))

    comp = DummyComponent()
    instrument(comp, "TestComp")
    comp.process("data")

    assert len(calls) == 1
    assert calls[0] == ("before", "TestComp", "process")


def test_after_hook_fires_on_success():
    calls = []

    def after(ct, mn, result, elapsed, ctx):
        calls.append(("after", ct, mn, result))

    register_hook("after", after)

    comp = DummyComponent()
    instrument(comp, "TestComp")
    comp.process("data")

    assert len(calls) == 1
    assert calls[0][0] == "after"
    assert calls[0][3] == {"result": "data"}


def test_error_hook_fires_on_exception():
    calls = []

    def error(ct, mn, exc, elapsed, ctx):
        calls.append(("error", ct, mn, type(exc).__name__))

    register_hook("error", error)

    class FailComp:
        def fail(self):
            raise ValueError("boom")

    comp = FailComp()
    instrument(comp, "FailComp")

    try:
        comp.fail()
    except ValueError:
        pass

    assert len(calls) == 1
    assert calls[0] == ("error", "FailComp", "fail", "ValueError")


def test_after_hook_gets_elapsed():
    elapsed_values = []

    def after(ct, mn, result, elapsed, ctx):
        elapsed_values.append(elapsed)

    register_hook("after", after)

    comp = DummyComponent()
    instrument(comp, "TestComp")
    comp.process("data")

    assert len(elapsed_values) == 1
    assert elapsed_values[0] >= 0  # non-negative elapsed time


# ── SkipRecord/SkipComponent pass through ──


def test_skip_record_passes_through():
    """SkipRecord should not trigger error hooks."""
    error_calls = []
    register_hook("error", lambda *a: error_calls.append(a))

    class SkippingComp:
        def transform(self, record):
            raise SkipRecord("bad record")

    comp = SkippingComp()
    instrument(comp, "SkippingComp")

    try:
        comp.transform({})
    except SkipRecord:
        pass
    else:
        assert False, "SkipRecord should have been raised"

    # Error hooks should NOT have fired
    assert len(error_calls) == 0


def test_skip_component_passes_through():
    """SkipComponent should not trigger error hooks."""
    error_calls = []
    register_hook("error", lambda *a: error_calls.append(a))

    class DisabledComp:
        def process(self):
            raise SkipComponent("disabled")

    comp = DisabledComp()
    instrument(comp, "DisabledComp")

    try:
        comp.process()
    except SkipComponent:
        pass
    else:
        assert False, "SkipComponent should have been raised"

    assert len(error_calls) == 0


# ── _bind_context extracts safe params ──


def test_bind_context_extracts_safe_keys():
    from inspect import signature

    def method(self, config, stream_slice=None, extra="ignored"):
        pass

    sig = signature(method)
    ctx = _bind_context(sig, (None, {"key": "val"}, {"start": "2024"}), {})

    assert "config" in ctx
    assert "stream_slice" in ctx
    assert "extra" not in ctx  # not in safe keys


def test_bind_context_handles_bad_args():
    from inspect import signature

    def method(self, x, y):
        pass

    sig = signature(method)
    # Wrong number of args — should return empty dict, not crash
    ctx = _bind_context(sig, (), {})
    assert ctx == {}


# ── Multiple hooks fire in order ──


def test_multiple_before_hooks_fire_in_order():
    calls = []
    register_hook("before", lambda ct, mn, ctx: calls.append("first"))
    register_hook("before", lambda ct, mn, ctx: calls.append("second"))

    comp = DummyComponent()
    instrument(comp, "TestComp")
    comp.process("data")

    assert calls == ["first", "second"]
