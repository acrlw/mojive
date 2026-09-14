"""Passive action ownership, bounded requests and explicit acknowledgements."""

from queue import Queue
from types import SimpleNamespace

import pytest

from mojive import PassiveAction, PassiveEvent
from mojive.app.passive_input import PassiveInput
from mojive.ui.viewport_widgets import ViewportChromeRegistry


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
