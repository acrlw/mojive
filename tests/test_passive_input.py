"""Passive action ownership, bounded requests and explicit acknowledgements."""

from queue import Queue
from types import SimpleNamespace

import pytest

from mojive import PassiveAction, PassiveEvent
from mojive.app.passive_input import PassiveInput
from mojive.ui.viewport_widgets import ToolHint, ViewportChromeRegistry


@pytest.fixture
def bridge(monkeypatch):
    from imgui_bundle import imgui

    monkeypatch.setattr(imgui, "is_any_item_active", lambda: False)
    app = SimpleNamespace(
        viewport_chrome=ViewportChromeRegistry(), localizer=SimpleNamespace(text=lambda text: text)
    )
    messages = []
    viewer = SimpleNamespace(
        app=app,
        session=SimpleNamespace(report_message=lambda *args, **kwargs: messages.append(args)),
    )
    bridge = PassiveInput(viewer, Queue(maxsize=64))
    bridge.configure((PassiveAction("pause", "space", "Pause policy", "toggle"),))
    return bridge


def test_requests_are_bounded_coalesced_and_acknowledged(bridge):
    bridge.request("pause")
    bridge.request("pause")
    event = bridge.events.get_nowait()
    assert event == PassiveEvent(1, "pause")
    assert bridge.events.empty()
    assert not bridge.viewer.app.external_playback_actions
    with pytest.raises(RuntimeError, match="Acknowledge"):
        bridge.configure(())
    bridge.acknowledge(event, error="Policy rejected pause")
    assert bridge.viewer.app.external_playback_actions == {"toggle"}
    with pytest.raises(ValueError, match="acknowledged"):
        bridge.acknowledge(event)
    bridge.request("pause")
    assert bridge.events.get_nowait().sequence == 2


@pytest.mark.parametrize(
    "blocked,focused,modifier,active",
    (
        (True, True, False, False),
        (False, False, False, False),
        (False, True, True, False),
        (False, True, False, True),
    ),
)
def test_passive_keys_respect_ui_and_window_ownership(
    bridge, monkeypatch, blocked, focused, modifier, active
):
    from imgui_bundle import imgui

    monkeypatch.setattr(imgui, "is_any_item_active", lambda: active)
    context = SimpleNamespace(
        blocked=blocked,
        viewport_focused=focused,
        key_down=lambda key: modifier,
        key_pressed=lambda key: True,
    )
    assert bridge(context) is None
    assert bridge.events.empty()


def test_key_claim_and_toolbar_share_the_same_request(bridge):
    claim = bridge(
        SimpleNamespace(
            blocked=False,
            viewport_focused=True,
            key_down=lambda key: False,
            key_pressed=lambda key: key == "space",
        )
    )
    assert claim.claims_key("space") and not claim.claims_key("r")
    assert bridge.viewer.app.viewport_chrome.dispatch("playback", "toggle")
    assert bridge.events.qsize() == 1


def test_invalid_bindings_preserve_existing_actions(bridge):
    actions = bridge.actions
    for invalid in (
        (PassiveAction("a", "8"), PassiveAction("b", "8")),
        (PassiveAction("a", "not_a_key"),),
        tuple(PassiveAction(f"a{i}", "8") for i in range(65)),
    ):
        with pytest.raises(ValueError):
            bridge.configure(invalid)
        assert bridge.actions == actions


@pytest.mark.parametrize("prefix", ("", "left_", "right_", "mod_"))
@pytest.mark.parametrize("modifier", ("shift", "ctrl", "alt", "super"))
def test_modifier_only_actions_fail_at_construction(prefix, modifier):
    with pytest.raises(ValueError, match="non-modifier"):
        PassiveAction("unreachable", prefix + modifier)


def test_custom_scene_hints_replace_defaults_without_rebinding_keys(bridge):
    registry = bridge.viewer.app.viewport_chrome.tool_hints
    unrelated = ToolHint("text", label="Inspection")
    registry.add("inspection", unrelated, surface="scene")
    actions = (PassiveAction("increase", "up_arrow"), PassiveAction("decrease", "down_arrow"))
    hints = (ToolHint("keys", label="Speed", keys=(("Up", "+"), ("Down", "−"))),)
    bridge.configure(actions, hints=hints, hint_surface="scene")
    assert not registry.resolve()
    assert len(registry.resolve(surface="scene")) == 2
    assert registry.resolve(surface="scene")[-1].keys == hints[0].keys
    assert bridge.keys == {"up_arrow", "down_arrow"}
    bridge.request("increase")
    event = bridge.events.get_nowait()
    bridge.acknowledge(event)

    for kwargs in ({"hint_surface": "unknown"}, {"hints": (object(),)}):
        with pytest.raises(ValueError):
            bridge.configure(actions, **kwargs)
        assert bridge.hint_surface == "scene"
        assert len(registry.resolve(surface="scene")) == 2
        assert not registry.resolve()

    bridge.configure(actions)
    assert len(registry.resolve()) == 2
    assert [hint.label for hint in registry.resolve(surface="scene")] == ["Inspection"]
    bridge.configure(actions, hints=())
    assert not registry.resolve()
    assert bridge.claim.claims_key("up_arrow")


def test_hint_update_does_not_change_pending_actions_or_transport(bridge):
    bridge.request("pause")
    pending = dict(bridge.pending)
    controls = tuple(bridge.viewer.app.viewport_chrome.playback_controls)
    actions = bridge.actions
    hints = (ToolHint("key", "Space", "Waiting"),)
    bridge.configure_hints(hints, surface="scene")
    registry = bridge.viewer.app.viewport_chrome.tool_hints
    assert not registry.resolve()
    assert registry.resolve(surface="scene")[0].label == "Waiting"
    assert bridge.pending == pending
    assert bridge.actions is actions
    assert tuple(bridge.viewer.app.viewport_chrome.playback_controls) == controls
    with pytest.raises(ValueError):
        bridge.configure_hints((object(),))
    assert registry.resolve(surface="scene")[0].label == "Waiting"
    bridge.acknowledge(next(iter(pending.values())))
    assert bridge.viewer.app.external_playback_actions == {"toggle"}
