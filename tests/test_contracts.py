from __future__ import annotations

import numpy as np
import pytest


def test_adapter_and_render_extension_types_are_public():
    """A custom physics adapter or tool should not need forge's internal module layout."""
    import forge_viewer as fv

    for name in (
        "ActuatorInfo",
        "ActuatorVisualType",
        "CameraView",
        "ConformanceReport",
        "CameraInfo",
        "DebugDraw",
        "DiagnosticFrame",
        "DiagnosticSource",
        "Environment",
        "FrameNeeds",
        "FrameMode",
        "JointInfo",
        "JointVisualType",
        "KeyframeInfo",
        "LabelMode",
        "MeshUpdate",
        "MuJoCoAdapter",
        "Occlusion",
        "RenderFlag",
        "RemoteSceneAdapter",
        "SceneAdapterBase",
        "SceneAdapter",
        "SceneFrame",
        "SceneModelInfo",
        "SceneSource",
        "SnapshotPublisher",
        "SnapshotWriter",
        "SensorInfo",
        "ToyPhysicsAdapter",
        "VisualGroupInfo",
        "audit_model",
        "make_adapter",
        "visual_coverage",
    ):
        assert name in fv.__all__
        assert getattr(fv, name) is not None


def test_null_backend_can_render_after_an_explicit_update():
    from forge_viewer.adapters.base import SceneFrame
    from forge_viewer.render.backend import NullBackend, RenderBackend

    backend = NullBackend()
    backend.update(SceneFrame())
    backend.resize(32, 24)

    assert isinstance(backend, RenderBackend)
    assert backend.render() is None
    assert backend.target.read_color().shape == (24, 32, 4)
    assert backend.target.read_depth().shape == (24, 32)
    assert backend.target.read_ids().shape == (24, 32)


def test_debug_outputs_are_not_exposed_as_independent_render_flags():
    from forge_viewer.render.backend import DebugView, RenderFlag

    debug_only = {view.value for view in DebugView} - {DebugView.WIREFRAME.value}
    assert debug_only.isdisjoint(flag.value for flag in RenderFlag)


def test_mujoco_visual_audit_covers_every_enum_flag():
    mujoco = pytest.importorskip("mujoco")

    from forge_viewer.mujoco_audit import visual_coverage

    coverage = visual_coverage()
    actual_rnd = {item["feature"] for item in coverage["mjtRndFlag"]}
    actual_vis = {item["feature"] for item in coverage["mjtVisFlag"]}
    expected_rnd = {name for name in dir(mujoco.mjtRndFlag) if name.startswith("mjRND_")}
    expected_vis = {name for name in dir(mujoco.mjtVisFlag) if name.startswith("mjVIS_")}
    assert actual_rnd == expected_rnd
    assert actual_vis == expected_vis
    assert all(
        item["status"] in {"exact", "equivalent", "partial", "deferred"}
        for group in coverage.values()
        for item in group
    )
    assert not any(item["status"] == "partial" for group in coverage.values() for item in group)
    assert [
        item["feature"]
        for group in coverage.values()
        for item in group
        if item["status"] == "deferred"
    ] == ["mjVIS_SDFITER"]


def test_camera_preset_tables_agree_on_which_way_is_up():

    from forge_viewer.ui.camera import PRESETS as CAM
    from forge_viewer.ui.panels.camera import PRESETS as PANEL

    panel = {name: (yaw, pitch) for name, yaw, pitch in PANEL}
    assert set(panel) == set(CAM)
    for name, (yaw, pitch) in panel.items():
        cam_yaw, cam_pitch = CAM[name]
        assert np.sign(pitch) == np.sign(cam_pitch)
        assert abs(pitch - cam_pitch) < 1.0
        assert abs(((yaw - cam_yaw + 180) % 360) - 180) < 1.0


def test_camera_slider_reset_matches_the_default_camera():

    from forge_viewer.ui.camera import OrbitCamera
    from forge_viewer.ui.panels.camera import PARAM_SLIDERS

    initial = {attr: init for attr, _lo, _hi, _fmt, init in PARAM_SLIDERS}
    camera = OrbitCamera()
    assert initial["yaw"] == camera.yaw
    assert initial["pitch"] == camera.pitch


def test_the_two_picking_coordinate_paths_agree():

    from forge_viewer.render.forge.picking import viewport_point_to_target_pixel
    from forge_viewer.types import ViewportImage

    rect = (12.0, 40.0, 800.0, 450.0)
    img = ViewportImage(texture_id=1, width=1600, height=900)
    probes = [
        (12.0, 40.0),
        (811.9, 489.9),
        (412.0, 265.0),
        (12.0, 489.9),
        (700.0, 100.0),
        (5.0, 20.0),
    ]
    for p in probes:
        mine = img.pixel_from_viewport_point(p, rect)
        theirs = viewport_point_to_target_pixel(p, rect, (img.width, img.height))
        assert mine == theirs


def test_picking_flips_y_and_uses_the_target_over_rect_ratio():

    from forge_viewer.types import ViewportImage

    rect = (0.0, 0.0, 400.0, 300.0)
    img = ViewportImage(texture_id=1, width=800, height=600)

    top = img.pixel_from_viewport_point((200.0, 0.0), rect)
    bottom = img.pixel_from_viewport_point((200.0, 299.9), rect)
    assert top is not None and bottom is not None
    assert top[1] == img.height - 1
    assert bottom[1] == 0

    mid = img.pixel_from_viewport_point((100.0, 150.0), rect)
    assert mid is not None and mid[0] == 200

    assert img.pixel_from_viewport_point((-1.0, 10.0), rect) is None
    assert img.pixel_from_viewport_point((10.0, 400.0), rect) is None


def test_instance_layout_matches_the_documented_stride():

    from forge_viewer.render.forge.instances import (
        INSTANCE_ATTRIBUTES,
        INSTANCE_BYTES,
        INSTANCE_WORDS,
    )
    from forge_viewer.render.scene import INSTANCE_FLOATS, INSTANCE_STRIDE

    assert INSTANCE_FLOATS == 32, "transform 16 + color 4 + material 4 + tex_coef 4 + cube_coef 4"
    assert INSTANCE_WORDS == 33
    assert INSTANCE_BYTES == 132
    assert INSTANCE_STRIDE == INSTANCE_BYTES

    cursor = 0
    for _name, _fmt, nbytes, _comps, off, _t in INSTANCE_ATTRIBUTES:
        assert off == cursor
        cursor += nbytes
    assert cursor == INSTANCE_BYTES


def test_object_id_attribute_is_an_integer_type():

    from forge_viewer.render.forge import gl_native as G
    from forge_viewer.render.forge.instances import INSTANCE_ATTRIBUTES

    ids = [a for a in INSTANCE_ATTRIBUTES if a[0] == "in_object_id"]
    assert len(ids) == 1
    _name, fmt, nbytes, comps, _off, gl_type = ids[0]
    assert gl_type == G.GL_UNSIGNED_INT
    assert fmt == "1u"
    assert (nbytes, comps) == (4, 1)


def test_shadow_clip_survives_the_whole_pathway():

    from forge_viewer.adapters.base import SceneSource
    from forge_viewer.render.forge.cascades import DEFAULT_SHADOW_CLIP, cascade_radii
    from forge_viewer.render.scene import RenderScene

    assert hasattr(SceneSource(), "shadow_clip")
    assert hasattr(RenderScene(), "shadow_clip")
    assert SceneSource().shadow_clip == DEFAULT_SHADOW_CLIP == 1.0

    r1 = cascade_radii(10.0, 1.0)
    r2 = cascade_radii(10.0, 1.5)
    assert np.allclose(r1, [10.0 / 9.0, 10.0 / 3.0, 10.0])
    assert np.allclose(r2, r1 * 1.5)


def test_pass_order_is_the_one_the_spec_pins():

    from forge_viewer.render.forge.registry import PASS_ORDER

    assert PASS_ORDER == (
        "shadow",
        "reflect",
        "opaque",
        "id",
        "skybox",
        "tendon",
        "transparent",
        "outline",
        "debug",
        "gizmo",
        "present",
    )


def test_registering_an_unknown_pass_name_is_refused():

    from forge_viewer.render.forge.registry import register_pass

    with pytest.raises(ValueError, match="Unknown pass"):
        register_pass("bloom", lambda: None)


def test_render_flags_cover_the_reference_renderer_feature_switches():

    from forge_viewer.render.backend import DebugView, RenderFlag

    names = {f.value for f in RenderFlag}
    for required in (
        "shadow",
        "wireframe",
        "reflection",
        "additive",
        "skybox",
        "fog",
        "haze",
        "cull_face",
        "texture",
        "joint",
        "actuator",
        "activation",
        "camera",
        "light",
        "contactpoint",
        "contactforce",
        "contactsplit",
        "autoconnect",
        "transparent",
        "com",
        "convexhull",
    ):
        assert required in names
    assert {"segment", "idcolor"}.issubset(view.value for view in DebugView)


def test_debug_view_combo_lists_every_enum_member():

    import ast
    import inspect
    import textwrap

    from forge_viewer.ui.panels import settings as settings_panel

    src = textwrap.dedent(inspect.getsource(settings_panel.SettingsPanel._debug_view))
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    assert loops
    iterated = {ast.unparse(n.iter) for n in loops}
    assert "DebugView" in iterated


def test_settings_panel_reads_the_debug_view_from_the_backend():

    from forge_viewer.render.backend import DebugView, NullBackend
    from forge_viewer.ui.panels.settings import SettingsPanel

    panel = SettingsPanel()
    backend = NullBackend()
    backend._view = DebugView.NORMAL
    assert panel.current_view(backend) is DebugView.NORMAL

    class NoGetter:
        pass

    assert panel.current_view(NoGetter()) is DebugView.SHADED
