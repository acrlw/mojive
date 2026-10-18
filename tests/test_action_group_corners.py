"""Production rounded action plates and catalogue examples share native offsets."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mojive.design.native.geometry import Rect, part_scope
from mojive.design.native.specimen_groups import ACTION_PADDING, action_layout, actions
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.notices import draw_file_receipt
from mojive.ui.workspace.viewport import pending_model_edits, transport
from mojive.ui.workspace.viewport_controls import camera_controls, toolbox
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance
from tests.test_menu_corner_offsets import _corner_point, _paint_covers

GROUPS = {
    "toolbox": ("viewport-toolbox", "tool-select", "snap"),
    "viewport_toolbar": ("viewport-toolbar", "projection-button", "viewport-layers"),
    "transport": ("viewport-transport", "previous", "playback-options"),
    "pending_edits": ("pending-model-edits", "apply-model-edits", "discard-model-edits"),
    "file_receipt": ("file-receipt", "receipt-open-folder", "receipt-copy-path"),
}


def _workspace(ui):
    session = SimpleNamespace(
        replay_info=None,
        can_step_back=False,
        state_take_recording=False,
        adapter=SimpleNamespace(caps=SimpleNamespace(simulation=True)),
    )
    app = SimpleNamespace(
        camera=SimpleNamespace(orthographic=False),
        tool="select",
        session=session,
        input_bindings=ui.value_context.input_bindings,
        model_edits=SimpleNamespace(applying=False, error=""),
        gizmo=SimpleNamespace(using=False),
        localizer=SimpleNamespace(text=str),
        _viewport_rect=(20 * ui.s, 20 * ui.s, 1000 * ui.s, 500 * ui.s),
    )
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=app,
        time=0,
        playing=False,
        snap=False,
        space="World",
        menus=SimpleNamespace(popup=lambda *args: None),
        activate=lambda name: None,
    )
    app.workspace = workspace
    return workspace


def _paint(part, workspace):
    ui, scale = workspace.ui, workspace.s
    if part == "toolbox":
        toolbox(
            workspace,
            (20 * scale, 20 * scale, 1000 * scale, 500 * scale),
            80 * scale,
            False,
            False,
            [],
        )
    elif part == "viewport_toolbar":
        camera_controls(workspace, (20 * scale, 20 * scale, 1000 * scale, 500 * scale), [])
    elif part == "transport":
        transport(
            workspace,
            20 * scale,
            20 * scale,
            1000 * scale,
            lambda x, y, w, h, *, radius: ui.rect(
                x, y, w * scale, h * scale, (22, 25, 28, 230), radius
            ),
        )
    elif part == "pending_edits":
        pending_model_edits(workspace, 20 * scale, 20 * scale, 1000 * scale, 500 * scale, [])
    else:
        draw_file_receipt(
            workspace.app,
            None,
            SimpleNamespace(text="Saved scene.xml", copy_text=""),
            Path("/tmp/scene.xml"),
        )


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("part", GROUPS)
@pytest.mark.parametrize("outer", (None, 0, 12, 32))
def test_actual_group_action_plates_keep_one_uniform_native_corner_budget(
    appearance, monkeypatch, part, outer
):
    ui, runtime, update, frame = appearance
    update({part: {} if outer is None else {"radius": outer}})
    if outer is not None:
        update({"icon_button": {"radius": 23}, "button": {"radius": 23}})
    runtime.observe_components = True
    workspace = _workspace(ui)
    children = {}
    original = ui.button

    def button(key, *args, **kwargs):
        radius = ui.component_style("icon_button").get("radius", 6) * ui.s
        result = original(key, *args, **kwargs)
        children[key] = radius
        return result

    monkeypatch.setattr(ui, "button", button)
    for _ in range(3):
        runtime.observed.clear()
        frame(lambda: _paint(part, workspace))
    key, first, last = GROUPS[part]
    plate = next(record for record in runtime.observed if record["key"] == key)
    x, y, w, h = plate["hit"]
    padding = plate["padding"]
    assert len(set(padding)) == 1 and padding[0] > 0
    gap = padding[0]
    expected = max(0, plate["radius"] - gap)
    assert plate["childRadius"] == pytest.approx(expected)
    assert children[first] == pytest.approx(expected)
    assert children[last] == pytest.approx(expected)
    a, b, c, d = ui.hits[last]
    assert x + w - a - c == pytest.approx(gap)
    assert b - y == pytest.approx(gap) if part != "toolbox" else y + h - b - d == pytest.approx(gap)
    if outer is not None:
        assert plate["radius"] == pytest.approx(outer * ui.s)
    else:
        assert plate["radius"] == pytest.approx(6 * ui.s + gap)


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("part", ACTION_PADDING)
def test_catalogue_action_groups_paint_the_production_corner_budget(appearance, part, monkeypatch):
    ui, _runtime, update, frame = appearance
    update({part: {"radius": 12}, "icon_button": {"radius": 23}, "button": {"radius": 23}})
    painted = []
    original = ui.rect

    def rectangle(x, y, w, h, ink, r=0):
        painted.append(((x, y, w, h), r * ui.s))
        return original(x, y, w, h, ink, r)

    monkeypatch.setattr(ui, "rect", rectangle)
    owner = SimpleNamespace(values={})
    bounds = Rect(40 * ui.s, 40 * ui.s, 1000 * ui.s, 500 * ui.s)

    def paint():
        with part_scope(part):
            box, _flow = action_layout(ui, part, bounds)
            actions(owner, ui, ui.design.primitives, ui.component_style(part), part, bounds)
            return box

    box = frame(paint)
    assert painted[0] == (pytest.approx(box.frame.tuple()), pytest.approx(12 * ui.s))
    assert len(set(box.padding)) == 1
    assert all(
        radius == pytest.approx(max(0, 12 * ui.s - box.padding[0])) for _, radius in painted[1:]
    )


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("part", tuple(GROUPS)[:-1])
@pytest.mark.parametrize("dimension", ("width", "height"))
def test_actual_authored_small_action_frames_keep_the_parent_curve_in_their_native_mesh(
    appearance, monkeypatch, part, dimension
):
    ui, runtime, update, frame = appearance
    update(
        {
            part: {"radius": 32},
            "icon_button": {dimension: 20, "iconSize": 24 if dimension == "width" else 12},
            "button": {dimension: 20, **({"widthMode": "fixed"} if dimension == "width" else {})},
        }
    )
    spec = dict(runtime.spec, native=dict(runtime.spec["native"], g3=False))
    runtime.update_spec(spec)
    runtime.observe_components = True
    workspace, paints, current = _workspace(ui), {}, []
    button, rectangle = ui.button, ui.rect

    def painted_button(key, *args, **kwargs):
        current.append(key)
        try:
            return button(key, *args, **dict(kwargs, filled=True))
        finally:
            current.pop()

    def painted_rect(x, y, width, height, tint, r=None, **kwargs):
        draw = kwargs.get("draw") or ui.draw
        ink = tint if kwargs.get("color_resolved") else ui.ink(tint)
        result = rectangle(x, y, width, height, tint, r, **kwargs)
        if current:
            paints[current[-1]] = ((x, y, width, height), (r or 0) * ui.s, color(ink), draw)
        return result

    monkeypatch.setattr(ui, "button", painted_button)
    monkeypatch.setattr(ui, "rect", painted_rect)
    for _ in range(3):
        runtime.observed.clear()
        frame(lambda: _paint(part, workspace))
    group, first, last = GROUPS[part]
    plate = next(record for record in runtime.observed if record["key"] == group)
    assert plate["radius"] == pytest.approx(32 * ui.s)
    assert plate["padding"][0] == plate["padding"][2]
    assert plate["padding"][1] == plate["padding"][3]
    inset = max(plate["padding"])
    assert inset > 8 * ui.s
    for key in (first, last):
        bounds, radius, ink, draw = paints[key]
        assert bounds == pytest.approx(ui.hits[key])
        assert radius == pytest.approx(max(0, plate["radius"] - inset))
        assert radius <= min(bounds[2:]) / 2 - 1
        assert bounds[2 if dimension == "width" else 3] == pytest.approx(20 * ui.s)
        x, y, width, height = plate["hit"]
        epsilon = native_epsilon(x + width, y + height)
        if part != "toolbox" or key == first:
            assert bounds[1] - y == pytest.approx(inset, abs=epsilon)
        if part != "toolbox" or key == last:
            assert y + height - bounds[1] - bounds[3] == pytest.approx(inset, abs=epsilon)
        if key == last:
            assert x + width - bounds[0] - bounds[2] == pytest.approx(inset, abs=epsilon)
        diagonal = radius * (1 - 0.5**0.5)
        for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
            assert not _paint_covers(
                draw, _corner_point(bounds, right, bottom, diagonal - 0.8), ink
            )
            assert _paint_covers(draw, _corner_point(bounds, right, bottom, diagonal + 0.8), ink)


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("part", ACTION_PADDING)
def test_catalogue_group_slots_measure_authored_control_heights_before_allocating_the_plate(
    appearance, part
):
    ui, _runtime, update, frame = appearance
    update(
        {
            part: {"radius": 32},
            "icon_button": {"height": 20, "iconSize": 12},
            "button": {"height": 20},
        }
    )
    bounds = Rect(40 * ui.s, 40 * ui.s, 1000 * ui.s, 600 * ui.s)

    def paint():
        with part_scope(part):
            box, layout = action_layout(ui, part, bounds)
            children = actions(
                SimpleNamespace(values={}),
                ui,
                ui.design.primitives,
                ui.component_style(part),
                part,
                bounds,
            )
            return box, layout, children

    box, layout, children = frame(paint)
    assert tuple(rect.height for rect in layout.rectangles) == pytest.approx(
        tuple(rect[3] for rect in children)
    )
    assert min(rect[1] for rect in children) - box.frame.y == pytest.approx(box.padding[1])
    assert box.frame.bottom - max(rect[1] + rect[3] for rect in children) == pytest.approx(
        box.padding[3]
    )


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("layout", ("pending-compact", "pending-stacked", "transport-compact"))
def test_compact_and_stacked_action_consumers_use_their_actual_adjacent_corner_gaps(
    appearance, layout
):
    ui, runtime, update, frame = appearance
    part = "transport" if layout.startswith("transport") else "pending_edits"
    update(
        {
            part: {"radius": 32},
            "icon_button": {"height": 20, "iconSize": 12},
            "button": {"height": 20},
        }
    )
    runtime.observe_components = True
    workspace, scale = _workspace(ui), ui.s

    def draw():
        if part == "transport":
            transport(workspace, 20 * scale, 20 * scale, 330 * scale, lambda *a, **k: None)
        else:
            pending_model_edits(
                workspace,
                20 * scale,
                20 * scale,
                320 * scale,
                (220 if layout.endswith("compact") else 500) * scale,
                [],
            )

    for _ in range(3):
        runtime.observed.clear()
        frame(draw)
    key, first, last = GROUPS[part]
    plate = next(record for record in runtime.observed if record["key"] == key)
    x, y, width, height = plate["hit"]
    inset = max(plate["padding"])
    epsilon = native_epsilon(x + width, y + height)
    for name in (first, last):
        child = ui.hits[name]
        assert child[3] == pytest.approx(20 * scale)
        assert y + height - child[1] - child[3] == pytest.approx(inset, abs=epsilon)
        assert plate["radius"] - inset == pytest.approx(plate["childRadius"])
    if layout == "pending-stacked":
        assert ui.hits[first][0] - x == pytest.approx(inset, abs=epsilon)
    else:
        child = ui.hits[last]
        assert x + width - child[0] - child[2] == pytest.approx(inset, abs=epsilon)
        assert child[1] - y == pytest.approx(inset, abs=epsilon)
        if part == "pending_edits":
            status = ui.hits["model-edit-status"]
            assert status[0] - x == pytest.approx(inset, abs=epsilon)
            assert status[1] - y == pytest.approx(inset, abs=epsilon)
