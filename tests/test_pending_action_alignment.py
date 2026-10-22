"""Pending-edit actions stay right aligned when captions wrap above them."""

from types import SimpleNamespace

import pytest

from mojive.ui.workspace.viewport import pending_model_edits
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.25, 2), indirect=True)


@pytest.mark.parametrize("width,height", ((850, 700), (350, 550), (360, 240)))
def test_pending_actions_share_the_content_right_edge(appearance, width, height):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    app = SimpleNamespace(
        model_edits=SimpleNamespace(applying=False, error="", clear=lambda: None),
        gizmo=SimpleNamespace(using=False, _reset_model_placement=lambda: None),
        _apply_model_edits_requested=False,
    )
    workspace = SimpleNamespace(ui=ui, s=ui.s, app=app)

    def draw():
        runtime.observed.clear()
        pending_model_edits(workspace, 20 * ui.s, 20 * ui.s, width * ui.s, height * ui.s, [])

    for _ in range(3):
        frame(draw)
    parent = next(row for row in runtime.observed if row["key"] == "pending-model-edits")
    left, top, content_width, _content_height = parent["content"]
    apply = ui.hits["apply-model-edits"]
    discard = ui.hits["discard-model-edits"]
    assert discard[0] + discard[2] == pytest.approx(left + content_width, abs=0.05)
    assert apply[0] >= left
    assert apply[1] == pytest.approx(discard[1], abs=0.05)
    if width < 392 and height >= 300:
        assert apply[1] > top


def test_projection_arrow_has_even_trailing_inset(appearance, monkeypatch):
    from mojive.ui.workspace.viewport_controls import camera_controls

    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(camera=SimpleNamespace(orthographic=False)),
        menus=SimpleNamespace(popup=lambda *args: None),
    )
    slots = []
    original = ui.role_icon

    def role_icon(name, x, y, w=None, h=None, **options):
        if name == "down":
            slots.append((x, y, w, h))
        return original(name, x, y, w, h, **options)

    monkeypatch.setattr(ui, "role_icon", role_icon)

    def draw():
        slots.clear()
        camera_controls(workspace, (20 * ui.s, 20 * ui.s, 900 * ui.s, 600 * ui.s), [])

    for _ in range(3):
        frame(draw)
    assert len(slots) == 1
    button = ui.hits["projection-button"]
    x, _y, width, _height = slots[0]
    assert button[0] + button[2] - x - width == pytest.approx(8 * ui.s, abs=0.05)
