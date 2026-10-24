"""Camera directions and created primitives retain distinct production symbols."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.base import AdapterCaps
from mojive.types import MeshShape
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.workspace.menu_actions import dispatch
from mojive.ui.workspace.menu_catalog import items
from mojive.ui.workspace.widget_icons import GLYPHS, ICON_STROKE

PRIMITIVES = ("Box", "Sphere", "Cylinder", "Capsule", "Cone", "Ellipsoid", "Plane")
DIRECTIONS = ("front", "back", "left", "right", "top", "bottom", "iso")


def test_viewport_layers_share_the_visible_height_of_camera_and_shading():
    heights = [
        GLYPHS[name]["bounds"][3] - GLYPHS[name]["bounds"][1]
        for name in ("camera", "shading", "layers")
    ]
    assert max(heights) / min(heights) <= 1.05


def test_view_actions_do_not_reuse_object_or_state_snapshot_silhouettes():
    from mojive.ui.workspace.widget_icons import COMPILED_ALIASES

    for names in (
        ("view-iso", "shape-box", "body-frame"),
        ("camera", "capture-image", "snapshot", "view-front", "view-back"),
        ("settings", "general", "mujoco-visuals", "layers"),
        ("collision", "shape-box", "both"),
        ("view-top", "export"),
        ("view-bottom", "upload"),
        ("pose", "joint", "joint-hinge"),
        ("snapshot", "key", "reset", "performance", "capture-image", "save"),
        ("view-front", "body", "view-back"),
        ("sensor", "waveform", "performance", "body", "camera"),
    ):
        assert len({repr(GLYPHS[name]["contours"]) for name in names}) == len(names)
    assert COMPILED_ALIASES["tool-body"] == "body-frame"
    assert COMPILED_ALIASES["key-snapshot"] == "snapshot"


@pytest.fixture
def icon_context():
    context = imgui.create_context()
    yield
    imgui.destroy_context(context)


@pytest.mark.parametrize(
    "names",
    [
        tuple("view-" + n for n in DIRECTIONS),
        tuple("shape-" + n.lower() for n in PRIMITIVES),
        ("camera", "capture-image", "snapshot"),
        ("pose", "joint", "joint-hinge"),
        ("geometry-default",),
        ("sensor",),
    ],
)
def test_direction_and_primitive_symbols_remain_distinct_at_production_sizes(names, icon_context):
    from mojive.ui.workspace.icon_mesh import prepared_icon

    glyphs = [GLYPHS[name] for name in names]
    assert len({repr(glyph["contours"]) for glyph in glyphs}) == len(names)
    for name, glyph in zip(names, glyphs, strict=True):
        inset = {"pose": 0.75, "camera": 1.25, "sensor": 1.25, "joint": 1.75}.get(name, 2)
        assert min(glyph["bounds"]) >= inset
        assert max(glyph["bounds"]) <= 24 - inset
        for contour in glyph["contours"]:
            if contour["stroke"] != "none":
                assert (contour["width"], contour["cap"], contour["join"]) == (
                    ICON_STROKE,
                    "round",
                    "round",
                )
        for size in (14, 16, 18, 24, 56, 112):
            operations = prepared_icon(name, size)
            assert any(
                indices
                for _background, _opacity, groups in operations
                for _fringe, _vertices, indices in groups
            )


@pytest.mark.parametrize("topology", (False, True))
@pytest.mark.parametrize("authoring", (False, True))
def test_add_menu_keeps_creation_symbols_and_capabilities(topology, authoring):
    context = imgui.create_context()
    workspace = SimpleNamespace(
        app=SimpleNamespace(
            entity=None,
            session=SimpleNamespace(
                adapter=SimpleNamespace(
                    caps=AdapterCaps(topology_editing=topology, scene_authoring=authoring)
                )
            ),
        )
    )
    try:
        rows = items(SimpleNamespace(workspace=workspace), "Add")
        shapes = rows[: len(PRIMITIVES)]
        assert tuple(row.label for row in shapes) == PRIMITIVES
        assert tuple(row.icon for row in shapes) == tuple(
            "shape-" + name.lower() for name in PRIMITIVES
        )
        assert all(row.icon in GLYPHS for row in rows if row)
        assert all(row.enabled == (row.label != "Capsule" or topology) for row in shapes)
        assert rows[len(PRIMITIVES)] is None
        assert [(row.label, row.icon, row.enabled) for row in rows[len(PRIMITIVES) + 1 :]] == [
            ("Point Light", "light", authoring),
            ("Camera", "camera", authoring),
            ("Site", "target", topology),
        ]
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_add_menu_routes_each_shape_to_its_existing_creation_operation(primitive):
    calls = []
    app = SimpleNamespace(
        session=SimpleNamespace(
            adapter=SimpleNamespace(caps=SimpleNamespace(topology_editing=True))
        ),
        _add_scene_object=lambda *args, **kwargs: calls.append(("scene", args, kwargs)),
        _add_model_primitive=lambda *args, **kwargs: calls.append(("model", args, kwargs)),
    )
    dispatch(SimpleNamespace(workspace=SimpleNamespace(app=app)), "create:" + primitive)
    if primitive == "Capsule":
        assert calls == [("model", ("capsule", "capsule"), {})]
    elif primitive == "Ellipsoid":
        assert calls == [("scene", (MeshShape.SPHERE, "ellipsoid"), {"size": (0.65, 0.45, 0.35)})]
    else:
        assert calls == [("scene", (MeshShape(primitive.lower()), primitive.lower()), {})]


@pytest.mark.parametrize("label", ("Capsule", "Site", "Point Light", "Camera"))
def test_creation_does_not_write_to_an_adapter_without_required_capability(label):
    app = SimpleNamespace(session=SimpleNamespace(adapter=SimpleNamespace(caps=AdapterCaps())))
    dispatch(SimpleNamespace(workspace=SimpleNamespace(app=app)), "create:" + label)


@pytest.mark.parametrize("orthographic", (False, True))
def test_view_menu_separates_action_icons_from_projection_configuration(orthographic, icon_context):
    from mojive.capture import RecordingPhase

    app = SimpleNamespace(
        entity=None,
        session=SimpleNamespace(adapter=SimpleNamespace(caps=SimpleNamespace())),
        camera=SimpleNamespace(orthographic=orthographic),
        recording=SimpleNamespace(phase=RecordingPhase.IDLE, active=False),
        tool="select",
        gizmos=True,
        input_bindings=DEFAULT_INPUT_BINDINGS,
    )
    menu = SimpleNamespace(workspace=SimpleNamespace(app=app))
    view = {row.label: row for row in items(menu, "View") if row}
    assert not view["Projection"].icon
    assert view["Capture"].icon == "capture-image"
    assert all(not row.icon for row in items(menu, "Capture"))
    projection = {row.label: row for row in items(menu, "Projection") if row}
    assert projection["Orthographic"].checked is orthographic
    assert projection["Perspective"].checked is not orthographic


@pytest.mark.parametrize("scale", (1, 2))
def test_inventory_capture_allocates_a_window_for_the_final_icon_row(monkeypatch, tmp_path, scale):
    from examples.ui_design.native import docking_study

    class WindowAllocated(Exception):
        pass

    configurations = []

    def allocate(config, _args):
        configurations.append(config)
        raise WindowAllocated

    monkeypatch.setattr(docking_study, "create_review_window", allocate)
    with pytest.raises(WindowAllocated):
        docking_study.main(["--icons", "--ui-scale", str(scale), "--output", str(tmp_path)])
    config = configurations[0]
    last_row_y = 88 + (len(GLYPHS) - 1) // 12 * 90
    assert config.height / scale >= last_row_y + 37 + 12 + 24
    assert config.width / scale >= 24 + 11 * 116 + 80 + 24 + 24


def test_workbench_action_samples_use_registered_production_symbols():
    from mojive.design.native.specimen_groups import ACTIONS
    from mojive.ui.workspace.widget_icons import COMPILED_ALIASES, PRODUCTION_GLYPHS

    known = GLYPHS.keys() | COMPILED_ALIASES.keys() | PRODUCTION_GLYPHS.keys()
    for group, actions in ACTIONS.items():
        assert all(icon is None or icon in known for _label, icon in actions), group
    assert dict(ACTIONS["viewport_toolbar"])["透视"] == "perspective"
    assert dict(ACTIONS["timeline_toolbar"])["快照"] == "snapshot"
    assert dict(ACTIONS["bottom_header"])["曲线"] == "waveform"
    assert dict(ACTIONS["bottom_header"])["统计"] == "performance"


def test_sensor_navigation_and_curve_actions_keep_separate_symbols():
    from examples.ui_design.native import docking_study, guide_icons

    from mojive.ui.workspace import constants

    for panel_names, symbols in (
        (constants.PANELS, constants.ICONS),
        (docking_study.PANELS, docking_study.ICONS),
    ):
        panel_icons = dict(zip(panel_names, symbols, strict=True))
        assert panel_icons["Sensors"] == "sensor"
        assert panel_icons["Statistics"] == "performance"
    assert guide_icons.LABELS["sensor"] == "Sensors"
    assert guide_icons.LABELS["waveform"] == "Curves"


@pytest.mark.parametrize("name", ("view-top", "view-bottom"))
@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_vertical_camera_views_use_closed_planes_separated_from_direction_arrows(name, size):
    plane, *arrow = GLYPHS[name]["contours"]
    assert plane["closed"]
    plane_y = [point[1] for point in plane["points"]]
    for contour in arrow:
        arrow_y = [point[1] for point in contour["points"]]
        center_gap = (
            min(plane_y) - max(arrow_y) if name == "view-top" else min(arrow_y) - max(plane_y)
        )
        gap = center_gap - (plane["width"] + contour["width"]) / 2
        assert gap * size / 24 >= 2.5 * size / 24 - 1e-6
