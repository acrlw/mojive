from __future__ import annotations

import numpy as np
import pytest


def test_adapter_and_render_extension_types_are_public():
    """A custom physics adapter or tool should not need opengl's internal module layout."""
    import mojive as fv

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
        "RenderProduct",
        "RenderRequest",
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
        "schema_coverage",
        "visual_coverage",
    ):
        assert name in fv.__all__
        assert getattr(fv, name) is not None


def test_null_backend_can_render_after_an_explicit_update():
    from mojive.adapters.base import SceneFrame
    from mojive.render.backend import NullBackend, RenderBackend

    backend = NullBackend()
    backend.update(SceneFrame())
    backend.resize(32, 24)

    assert isinstance(backend, RenderBackend)
    assert backend.render() is None
    assert backend.target.read_color().shape == (24, 32, 4)
    rgb = np.empty((24, 32, 3), np.uint8)
    assert backend.target.read_rgb(out=rgb) is rgb
    assert backend.target.read_depth().shape == (24, 32)
    assert backend.target.read_ids().shape == (24, 32)


def test_render_requests_describe_products_without_prescribing_passes():
    from mojive.render.backend import RenderProduct, RenderRequest

    viewport = RenderRequest.viewport()
    assert viewport.needs(RenderProduct.COLOR)
    assert viewport.needs(RenderProduct.OBJECT_ID)
    assert not viewport.needs(RenderProduct.METRIC_DEPTH)
    assert RenderRequest.color().products == RenderProduct.COLOR
    assert RenderRequest.metric_depth().products == RenderProduct.METRIC_DEPTH
    assert RenderRequest.segmentation().products == RenderProduct.SEGMENTATION
    assert RenderRequest(int(RenderProduct.COLOR)).products is RenderProduct.COLOR
    with pytest.raises(ValueError, match="at least one product"):
        RenderRequest(RenderProduct(0))


def test_opengl_render_plan_prunes_unrequested_products():
    from mojive.render.backend import DebugView, RenderProduct, RenderRequest
    from mojive.render.opengl.backend import compile_render_plan

    viewport = compile_render_plan(None)
    assert "opaque" in viewport.passes
    assert "id" in viewport.passes
    assert "present" in viewport.passes

    color = compile_render_plan(RenderRequest.color())
    assert "opaque" in color.passes
    assert "id" not in color.passes
    assert "present" in color.passes

    debug_ids = compile_render_plan(RenderRequest.color(), DebugView.SEGMENT)
    assert "id" in debug_ids.passes

    depth = compile_render_plan(RenderRequest.metric_depth())
    assert depth.passes == ("export",)
    assert not depth.returns_color

    segmentation = compile_render_plan(RenderRequest.segmentation())
    assert segmentation.passes == ("export",)
    assert not segmentation.returns_color

    combined = compile_render_plan(
        RenderRequest(RenderProduct.COLOR | RenderProduct.METRIC_DEPTH | RenderProduct.SEGMENTATION)
    )
    assert "opaque" in combined.passes
    assert "export" in combined.passes
    assert "id" not in combined.passes


def test_wgpu_render_plan_separates_scene_and_export_workloads():
    from mojive.render.backend import DebugView, RenderProduct, RenderRequest
    from mojive.render.webgpu.backend import compile_render_plan

    viewport = compile_render_plan(None)
    assert viewport.color and viewport.export_identity and viewport.export
    assert not viewport.export_depth

    color = compile_render_plan(RenderRequest.color())
    assert color.color and not color.export

    debug_ids = compile_render_plan(RenderRequest.color(), DebugView.IDCOLOR)
    assert debug_ids.color and debug_ids.export_identity

    depth = compile_render_plan(RenderRequest.metric_depth())
    assert not depth.color and depth.export_depth and depth.export

    segmentation = compile_render_plan(RenderRequest.segmentation())
    assert not segmentation.color and segmentation.export_identity and segmentation.export

    combined = compile_render_plan(
        RenderRequest(RenderProduct.COLOR | RenderProduct.METRIC_DEPTH | RenderProduct.SEGMENTATION)
    )
    assert combined.color and combined.export_depth and combined.export_identity


def test_debug_outputs_are_not_exposed_as_independent_render_flags():
    from mojive.render.backend import DebugView, RenderFlag

    debug_only = {view.value for view in DebugView} - {DebugView.WIREFRAME.value}
    assert debug_only.isdisjoint(flag.value for flag in RenderFlag)


def test_mujoco_visual_audit_covers_every_enum_flag():
    mujoco = pytest.importorskip("mujoco")

    from mojive.mujoco_audit import visual_coverage

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


def test_mujoco_schema_audit_classifies_every_attributed_path():
    from mojive.adapters.mujoco_adapter import _MJCF_SCHEMA_ATTRIBUTES
    from mojive.mujoco_audit import schema_coverage

    report = schema_coverage()
    rows = {item["path"]: item for item in report["rows"]}
    expected = {"/".join(path) for path, fields in _MJCF_SCHEMA_ATTRIBUTES.items() if fields}
    assert set(rows) == expected
    assert rows["mujoco"]["status"] == "structured"
    assert rows["mujoco/compiler"]["status"] == "raw-mjcf-only"
    assert rows["mujoco/asset/hfield"]["status"] == "structured-partial"
    assert rows["mujoco/asset/material/layer"]["status"] == "raw-mjcf-only"
    assert rows["mujoco/asset/skin"]["status"] == "runtime-only"
    assert rows["mujoco/default/material/layer"]["status"] == "raw-mjcf-only"
    assert rows["mujoco/(world)body/site"]["status"] == "structured-partial"
    assert rows["mujoco/deformable/flex"]["status"] == "runtime-only"
    assert rows["mujoco/custom/numeric"]["status"] == "raw-mjcf-only"
    assert rows["mujoco/keyframe/key"]["status"] == "structured-partial"
    assert rows["mujoco/extension/plugin"]["status"] == "plugin-out-of-scope"
    assert rows["mujoco/actuator/plugin"]["status"] == "plugin-out-of-scope"
    assert {item["path"] for item in report["source_meta"]} == {
        "include",
        "frame",
        "replicate",
    }


def test_camera_preset_tables_agree_on_which_way_is_up():

    from mojive.ui.camera import PRESETS as CAM
    from mojive.ui.panels.camera import PRESETS as PANEL

    panel = {name: (yaw, pitch) for name, yaw, pitch in PANEL}
    assert set(panel) == set(CAM)
    for name, (yaw, pitch) in panel.items():
        cam_yaw, cam_pitch = CAM[name]
        assert np.sign(pitch) == np.sign(cam_pitch)
        assert abs(pitch - cam_pitch) < 1.0
        assert abs(((yaw - cam_yaw + 180) % 360) - 180) < 1.0


def test_camera_slider_reset_matches_the_default_camera():

    from mojive.ui.camera import OrbitCamera
    from mojive.ui.panels.camera import PARAM_SLIDERS

    initial = {attr: init for attr, _lo, _hi, _fmt, init in PARAM_SLIDERS}
    camera = OrbitCamera()
    assert initial["yaw"] == camera.yaw
    assert initial["pitch"] == camera.pitch


def test_the_two_picking_coordinate_paths_agree():

    from mojive.render.opengl.picking import viewport_point_to_target_pixel
    from mojive.types import ViewportImage

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

    from mojive.types import ViewportImage

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

    from mojive.render.opengl.instances import (
        IDENTITY_ATTRIBUTES,
        IDENTITY_BYTES,
        INSTANCE_ATTRIBUTES,
        INSTANCE_BYTES,
        INSTANCE_WORDS,
        POSE_ATTRIBUTES,
        POSE_BYTES,
        VISUAL_ATTRIBUTES,
        VISUAL_BYTES,
    )
    from mojive.render.scene import INSTANCE_FLOATS, INSTANCE_STRIDE

    assert INSTANCE_FLOATS == 32, "transform 16 + color 4 + material 4 + tex_coef 4 + cube_coef 4"
    assert INSTANCE_WORDS == 36
    assert (POSE_BYTES, VISUAL_BYTES, IDENTITY_BYTES) == (64, 64, 16)
    assert INSTANCE_BYTES == 144
    assert INSTANCE_STRIDE == INSTANCE_BYTES

    for entries, size in (
        (POSE_ATTRIBUTES, POSE_BYTES),
        (VISUAL_ATTRIBUTES, VISUAL_BYTES),
        (IDENTITY_ATTRIBUTES, IDENTITY_BYTES),
    ):
        cursor = 0
        for _name, _fmt, nbytes, _comps, off, _t in entries:
            assert off == cursor
            cursor += nbytes
        assert cursor == size
    assert len(INSTANCE_ATTRIBUTES) == 12


def test_object_id_attribute_is_an_integer_type():

    from mojive.render.opengl import gl_native as G
    from mojive.render.opengl.instances import INSTANCE_ATTRIBUTES

    ids = [a for a in INSTANCE_ATTRIBUTES if a[0] == "in_object_id"]
    assert len(ids) == 1
    _name, fmt, nbytes, comps, _off, gl_type = ids[0]
    assert gl_type == G.GL_UNSIGNED_INT
    assert fmt == "1u"
    assert (nbytes, comps) == (4, 1)


def test_shadow_clip_survives_the_whole_pathway():

    from mojive.adapters.base import SceneSource
    from mojive.render.opengl.cascades import DEFAULT_SHADOW_CLIP, cascade_radii
    from mojive.render.scene import RenderScene

    assert hasattr(SceneSource(), "shadow_clip")
    assert hasattr(RenderScene(), "shadow_clip")
    assert SceneSource().shadow_clip == DEFAULT_SHADOW_CLIP == 1.0

    r1 = cascade_radii(10.0, 1.0)
    r2 = cascade_radii(10.0, 1.5)
    assert np.allclose(r1, [10.0 / 9.0, 10.0 / 3.0, 10.0])
    assert np.allclose(r2, r1 * 1.5)


def test_pass_order_is_the_one_the_spec_pins():

    from mojive.render.opengl.registry import PASS_ORDER

    assert PASS_ORDER == (
        "shadow",
        "reflect",
        "opaque",
        "id",
        "export",
        "skybox",
        "tendon",
        "transparent",
        "outline",
        "debug",
        "gizmo",
        "present",
    )


def test_registering_an_unknown_pass_name_is_refused():

    from mojive.render.opengl.registry import register_pass

    with pytest.raises(ValueError, match="Unknown pass"):
        register_pass("bloom", lambda: None)


def test_render_flags_cover_the_reference_renderer_feature_switches():

    from mojive.render.backend import DebugView, RenderFlag

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

    from mojive.ui.panels import settings as settings_panel

    src = textwrap.dedent(inspect.getsource(settings_panel.SettingsPanel._debug_view))
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    assert loops
    iterated = {ast.unparse(n.iter) for n in loops}
    assert "DebugView" in iterated


def test_settings_panel_reads_the_debug_view_from_the_backend():

    from mojive.render.backend import DebugView, NullBackend
    from mojive.ui.panels.settings import SettingsPanel

    panel = SettingsPanel()
    backend = NullBackend()
    backend._view = DebugView.NORMAL
    assert panel.current_view(backend) is DebugView.NORMAL

    class NoGetter:
        pass

    assert panel.current_view(NoGetter()) is DebugView.SHADED


@pytest.mark.integration
@pytest.mark.parametrize(
    "imports",
    [
        "from mojive.types import MeshData; from mojive.commands import Select",
        "from mojive import Scene, SceneAdapterBase, InputClaim, SceneFrame, CameraView, SceneRenderer",
        "from mojive.adapters import WorkspaceAdapter; from mojive import Scene",
    ],
)
def test_data_imports_do_not_load_window_render_or_physics_packages(imports):
    import subprocess
    import sys

    code = f"""
import sys
from importlib.abc import MetaPathFinder
class BlockRuntime(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'imgui_bundle', 'glfw', 'moderngl', 'mujoco', 'wgpu'}}:
            raise AssertionError('Unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
{imports}
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lazy_public_exports_remain_discoverable_and_cached():
    import mojive
    from mojive.scene import Scene

    assert set(mojive.__all__) <= set(dir(mojive))
    assert mojive.Scene is Scene
    assert vars(mojive)["Scene"] is Scene
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = mojive.missing_export
