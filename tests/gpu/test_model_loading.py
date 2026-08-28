from __future__ import annotations

import gc
import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")
pytest.importorskip("mujoco")

from forge_viewer.adapters.base import (  # noqa: E402
    GeometryShapeProperties,
    ModelAssetInfo,
    NodeType,
)
from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.commands import AddModelComponent, SelectNode  # noqa: E402
from forge_viewer.composition import build, build_editor, build_scene  # noqa: E402
from forge_viewer.scene import Scene  # noqa: E402
from forge_viewer.types import CameraView  # noqa: E402
from forge_viewer.ui.app import (  # noqa: E402
    IMAGE_FILTERS,
    MESH_FILTERS,
    MODEL_FILTERS,
    _ModelLoadJob,
)
from forge_viewer.ui.camera_preview import CameraPreview  # noqa: E402


@pytest.fixture(scope="module")
def viewer():
    instance = build(resolve("empty"), paused=True, vsync=False, width=960, height=640)
    try:
        yield instance
    finally:
        instance.release()


def _click(viewer, point: tuple[float, float]) -> None:
    from imgui_bundle import imgui

    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()


def _item_center(viewer, function_name: str, label: str) -> tuple[float, float]:
    from imgui_bundle import imgui

    original = getattr(imgui, function_name)
    found = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append(((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5))
        return result

    setattr(imgui, function_name, spy)
    try:
        viewer.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def _activate_panel(viewer, name: str) -> None:
    from imgui_bundle import imgui

    window = imgui.internal.find_window_by_name(name)
    assert window is not None
    node = window.dock_node
    if node is None or node.selected_tab_id == window.tab_id:
        return
    tab = next(item for item in node.tab_bar.tabs if item.id_ == window.tab_id)
    bar = node.tab_bar.bar_rect
    _click(
        viewer,
        (bar.min.x + tab.offset + tab.width * 0.5, (bar.min.y + bar.max.y) * 0.5),
    )
    for _ in range(2):
        viewer.sync()
    assert not imgui.internal.find_window_by_name(name).hidden


def _trigger_button(viewer, monkeypatch, label: str) -> None:
    from imgui_bundle import imgui

    original = imgui.button
    found = []

    def trigger(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            found.append(True)
            return True
        return result

    monkeypatch.setattr(imgui, "button", trigger)
    viewer.sync()
    monkeypatch.setattr(imgui, "button", original)
    assert found


def test_file_menu_opens_model_browser(viewer, monkeypatch):
    opened = []
    viewer.sync()
    monkeypatch.setattr(viewer.app, "_open_model_dialog", lambda: opened.append(True))
    _click(viewer, _item_center(viewer, "begin_menu", "File"))
    _click(viewer, _item_center(viewer, "menu_item", "Open Model (MJCF / URDF)..."))
    assert opened


def test_add_model_dialog_filters_formats_and_accepts_multiple_files(viewer, monkeypatch):
    from imgui_bundle import portable_file_dialogs

    paths = ["/tmp/robot.xml", "/tmp/arm.urdf"]
    opened = {}

    class Dialog:
        def ready(self, _timeout):
            return True

        def result(self):
            return paths

    def open_file(title, default_path, filters, options):
        opened.update(title=title, default_path=default_path, filters=filters, options=options)
        return Dialog()

    loaded = []
    monkeypatch.setattr(portable_file_dialogs, "open_file", open_file)
    monkeypatch.setattr(
        viewer.app,
        "_queue_model_load",
        lambda action, path, position=None: loaded.append((action, Path(path), position)),
    )

    viewer.app._open_model_dialog("add")
    viewer.app._poll_model_dialog()

    assert opened["title"] == "Add MJCF or URDF models"
    assert opened["filters"] == MODEL_FILTERS
    assert opened["options"] == portable_file_dialogs.opt.multiselect
    assert [item[:2] for item in loaded] == [("add", Path(path)) for path in paths]


def test_texture_dialog_uses_image_filters_and_unique_model_name(viewer, monkeypatch, tmp_path):
    from imgui_bundle import portable_file_dialogs

    import forge_viewer.commands as cmd

    path = tmp_path / "surface.png"
    path.write_bytes(b"dialog only")
    opened = {}

    class Dialog:
        def ready(self, _timeout):
            return True

        def result(self):
            return str(path)

    def open_file(title, default_path, filters):
        opened.update(title=title, default_path=default_path, filters=filters)
        return Dialog()

    submitted = []
    monkeypatch.setattr(portable_file_dialogs, "open_file", open_file)
    monkeypatch.setattr(viewer.session, "model_texture_names", lambda _model_id: ("surface",))
    monkeypatch.setattr(
        viewer.session,
        "submit",
        lambda command: submitted.append(command) or cmd.CommandResult.good(""),
    )

    viewer.app._open_texture_dialog(2, 7)
    viewer.app._poll_texture_dialog()

    assert opened["title"] == "Import 2D texture"
    assert opened["filters"] == IMAGE_FILTERS
    assert len(submitted) == 1
    assert isinstance(submitted[0], cmd.ImportModelTexture)
    assert (submitted[0].model_id, submitted[0].material_index, submitted[0].name) == (
        2,
        7,
        "surface2",
    )


def test_geometry_resource_dialog_imports_and_assigns_a_unique_mesh(viewer, monkeypatch, tmp_path):
    from imgui_bundle import portable_file_dialogs

    import forge_viewer.commands as cmd

    path = tmp_path / "part.obj"
    path.write_bytes(b"dialog only")
    opened = {}

    class Dialog:
        def ready(self, _timeout):
            return True

        def result(self):
            return str(path)

    def open_file(title, default_path, filters):
        opened.update(title=title, default_path=default_path, filters=filters)
        return Dialog()

    submitted = []
    monkeypatch.setattr(portable_file_dialogs, "open_file", open_file)
    monkeypatch.setattr(
        viewer.session,
        "geometry_shape_properties",
        lambda node_id: GeometryShapeProperties(node_id, "box", "", ("part",), ()),
    )
    monkeypatch.setattr(
        viewer.session,
        "submit",
        lambda command: submitted.append(command) or cmd.CommandResult.good(""),
    )

    viewer.app._open_geometry_resource_dialog(12, "mesh")
    viewer.app._poll_geometry_resource_dialog()

    assert opened["title"] == "Import mesh"
    assert opened["filters"] == MESH_FILTERS
    assert len(submitted) == 1
    assert isinstance(submitted[0], cmd.ImportModelGeometryResource)
    assert (submitted[0].node_id, submitted[0].resource_type, submitted[0].name) == (
        12,
        "mesh",
        "part2",
    )


def test_model_asset_dialog_imports_standalone_and_replaces_by_name(viewer, monkeypatch, tmp_path):
    from imgui_bundle import portable_file_dialogs

    import forge_viewer.commands as cmd

    path = tmp_path / "terrain.png"
    path.write_bytes(b"dialog only")
    opened = []

    class Dialog:
        def ready(self, _timeout):
            return True

        def result(self):
            return str(path)

    def open_file(title, default_path, filters):
        opened.append((title, default_path, filters))
        return Dialog()

    submitted = []
    monkeypatch.setattr(portable_file_dialogs, "open_file", open_file)
    monkeypatch.setattr(
        viewer.session,
        "model_assets",
        lambda model_id: (ModelAssetInfo(model_id, "hfield", "terrain", 0),),
    )
    monkeypatch.setattr(
        viewer.session,
        "submit",
        lambda command: submitted.append(command) or cmd.CommandResult.good(""),
    )

    fields = (("size", "2 3 0.4 0.2"),)
    viewer.app._open_model_asset_import_dialog(4, "hfield", fields)
    viewer.app._poll_model_asset_dialog()
    assert opened[-1][0] == "Import PNG height field"
    assert opened[-1][2] == IMAGE_FILTERS
    assert isinstance(submitted[-1], cmd.ImportModelAsset)
    assert (submitted[-1].model_id, submitted[-1].name, submitted[-1].fields) == (
        4,
        "terrain2",
        fields,
    )

    viewer.app._open_model_asset_replace_dialog(4, "hfield", "terrain")
    viewer.app._poll_model_asset_dialog()
    assert opened[-1][0] == "Replace PNG height field"
    assert isinstance(submitted[-1], cmd.ReplaceModelAssetFile)
    assert (submitted[-1].model_id, submitted[-1].name, submitted[-1].path) == (
        4,
        "terrain",
        path,
    )

    viewer.app._open_model_asset_replace_dialog(4, "texture", "surface")
    viewer.app._poll_model_asset_dialog()
    assert opened[-1][0] == "Replace PNG texture"
    assert opened[-1][2] == IMAGE_FILTERS
    assert isinstance(submitted[-1], cmd.ReplaceModelAssetFile)
    assert (submitted[-1].asset_type, submitted[-1].name) == ("texture", "surface")


def test_runtime_model_loading_rebuilds_gpu_scene(viewer):
    viewer.sync()
    assert viewer.backend.stats.instances == 0

    viewer.window._file_drag_active = True
    viewer.window._on_file_drop(None, [str(resolve("test_scene.xml"))])
    assert not viewer.window.file_drag_active
    import forge_viewer.commands as cmd

    assert viewer.session.submit(cmd.Play())
    assert not viewer.session.paused
    viewer.sync()
    assert viewer.session.paused
    viewer.app._model_load_future.result(timeout=10.0)
    viewer.sync()
    assert viewer.session.paused
    assert viewer.session.asset_path == resolve("test_scene.xml")

    for name in ("test_scene.urdf",):
        result = viewer.app.load_model(resolve(name))
        assert result.ok, result.message
        viewer.sync()
        assert viewer.session.asset_path == resolve(name)
        assert viewer.backend.stats.instances == viewer.session.source.instance_count
        assert viewer.backend.stats.instances > 0


def test_loading_overlay_preserves_the_docked_viewport(viewer, monkeypatch):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    for _ in range(3):
        viewer.sync()
    before_panel = viewer.app._viewport_panel_size
    before_rect = viewer.app._viewport_rect
    path = resolve("test_scene.xml")
    viewer.app._model_load_job = _ModelLoadJob("load", path, cmd.LoadAsset(path))
    viewer.app._model_load_future = object()
    monkeypatch.setattr(viewer.app, "_poll_model_load", lambda: True)
    try:
        viewer.sync()
        loading = imgui.internal.find_window_by_name("Loading###model_loading")
        assert loading is not None
        assert viewer.app._viewport_panel_size == pytest.approx(before_panel, abs=1.0)
        assert viewer.app._viewport_rect == pytest.approx(before_rect, abs=1.0)
        x, y, width, height = viewer.app._viewport_rect
        center = loading.pos + loading.size * 0.5
        assert x <= center.x <= x + width
        assert y <= center.y <= y + height
        assert width < imgui.get_main_viewport().size.x
    finally:
        viewer.app._model_load_future = None
        viewer.app._model_load_job = None


def test_viewer_and_editor_render_the_same_loaded_model_at_the_same_state():
    asset = resolve("parity_scene")
    images = []
    bounds = []
    entry_cameras = []
    camera = CameraView(
        eye=np.array((-4.0, -4.0, 3.0), np.float32),
        target=np.array((0.0, 0.0, 0.5), np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        fov_y=float(np.radians(45.0)),
        near=0.01,
        far=50.0,
        aspect=4.0 / 3.0,
    )
    for mode in ("viewer", "editor"):
        instance = (
            build(resolve("test_scene"), paused=True, vsync=False, width=960, height=640)
            if mode == "viewer"
            else build_editor(vsync=False, width=960, height=640)
        )
        try:
            instance.app.set_fixed_render_size(640, 480)
            instance.sync()
            instance.sync()
            assert instance.app.load_model(asset).ok
            instance.sync()
            instance.sync()
            hint = instance.session.camera_hint()
            assert hint is not None
            entry_camera = instance.app.camera.view()
            np.testing.assert_allclose(entry_camera.eye, hint.eye, atol=1e-6)
            np.testing.assert_allclose(entry_camera.target, hint.target, atol=1e-6)
            entry_cameras.append(entry_camera)
            instance.backend.set_camera(camera)
            instance.backend.highlight(0)
            instance.backend.update(instance.session.frame)
            instance.backend.debug.clear()
            assert instance.backend.render() is not None
            images.append(instance.backend.target.read_color(flip=True).copy())
            bounds.append(
                (instance.session.source.scene_center.copy(), instance.session.source.scene_extent)
            )
        finally:
            instance.release()

    assert bounds[1][0] == pytest.approx(bounds[0][0])
    assert bounds[1][1] == pytest.approx(bounds[0][1])
    np.testing.assert_allclose(entry_cameras[1].eye, entry_cameras[0].eye, atol=1e-6)
    np.testing.assert_allclose(entry_cameras[1].target, entry_cameras[0].target, atol=1e-6)
    delta = np.abs(images[1].astype(np.int16) - images[0].astype(np.int16))
    # Separate GL contexts can round a few shadow-edge samples by one UNORM step.
    assert int(delta.max()) <= 1
    assert float(delta.mean()) < 1e-3


def _replacement_camera(source) -> CameraView:
    center = np.asarray(source.scene_center, np.float32)
    extent = max(float(source.scene_extent), 1.0)
    return CameraView(
        eye=center + extent * np.array([2.4, -3.0, 1.8], np.float32),
        target=center,
        up=np.array([0.0, 0.0, 1.0], np.float32),
        fov_y=float(np.radians(45.0)),
        near=max(extent * 0.01, 1e-3),
        far=max(extent * 40.0, 50.0),
        aspect=4.0 / 3.0,
    )


def _render_loaded_model(instance, camera: CameraView) -> np.ndarray:
    instance.app.set_fixed_render_size(320, 240)
    instance.backend.set_camera(camera)
    instance.backend.highlight(0)
    instance.backend.update(instance.session.frame)
    instance.backend.debug.clear()
    assert instance.backend.render() is not None
    return instance.backend.target.read_color(flip=True).copy()


def test_repeated_model_replacement_matches_fresh_renderer_state():
    assets = tuple(
        resolve(name)
        for name in ("image_light", "parity_texture", "empty", "parity_scene", "test_scene")
    )
    baselines = {}
    for asset in assets:
        fresh = build(asset, paused=True, vsync=False, width=640, height=480)
        try:
            fresh.sync()
            camera = _replacement_camera(fresh.session.source)
            baselines[asset] = (
                _render_loaded_model(fresh, camera),
                camera,
                fresh.session.source.scene_center.copy(),
                fresh.session.source.scene_extent,
            )
        finally:
            fresh.release()

    instance = build(resolve("test_scene"), paused=True, vsync=False, width=640, height=480)
    try:
        instance.sync()
        _render_loaded_model(instance, _replacement_camera(instance.session.source))
        for asset in assets:
            assert instance.app.load_model(asset).ok
            instance.sync()
            expected, camera, center, extent = baselines[asset]
            assert instance.session.source.scene_center == pytest.approx(center)
            assert instance.session.source.scene_extent == pytest.approx(extent)
            actual = _render_loaded_model(instance, camera)
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            assert int(delta.max()) <= 1, asset.name
            assert float(delta.mean()) < 1e-3, asset.name
    finally:
        instance.release()


def test_camera_preview_matches_the_main_backend_for_the_same_camera_and_size():
    instance = build(resolve("parity_scene"), paused=True, vsync=False, width=960, height=640)
    preview = CameraPreview()
    try:
        instance.app.set_fixed_render_size(640, 480)
        instance.sync()
        camera = instance.session.camera_view(0).with_aspect(640.0 / 480.0)
        frame = instance.session.frame
        instance.backend.set_camera(camera)
        instance.backend.highlight(0)
        instance.backend.update(frame)
        instance.backend.debug.clear()
        assert instance.backend.render() is not None
        main = instance.backend.target.read_color(flip=True).copy()

        preview.update(
            instance.backend,
            instance.session.source,
            instance.session.structure_generation,
            frame,
            camera,
            (640, 480),
        )
        peer = preview._backend
        assert peer is not None
        assert np.array_equal(peer.target.read_color(flip=True), main)
    finally:
        preview.release()
        instance.release()


def test_dragging_camera_preview_does_not_orbit_scene_camera():
    from imgui_bundle import imgui

    instance = build(resolve("parity_scene"), paused=True, vsync=False, width=960, height=640)
    try:
        instance.sync()
        camera_node = next(node for node in instance.session.nodes if node.type is NodeType.CAMERA)
        assert instance.session.submit(SelectNode(camera_node.node_id))
        instance.sync()
        preview = instance.app.camera_preview
        assert preview._position is not None
        before_position = preview._position
        before_camera = instance.app.camera.view()
        scale = instance.window.style_scale
        start = (before_position[0] + 48.0 * scale, before_position[1] + 12.0 * scale)
        io = imgui.get_io()
        io.add_mouse_pos_event(*start)
        instance.sync()
        io.add_mouse_button_event(0, True)
        instance.sync()
        io.add_mouse_pos_event(start[0] - 40.0 * scale, start[1] - 24.0 * scale)
        instance.sync()
        io.add_mouse_button_event(0, False)
        instance.sync()

        assert preview._position != before_position
        after_camera = instance.app.camera.view()
        np.testing.assert_allclose(after_camera.eye, before_camera.eye)
        np.testing.assert_allclose(after_camera.target, before_camera.target)
    finally:
        imgui.get_io().add_mouse_button_event(0, False)
        instance.release()


def test_viewer_frames_reuse_adapter_buffers_without_python_growth():
    instance = build(resolve("actuator_visuals"), paused=False, vsync=False, width=640, height=480)
    tracemalloc.start()
    try:
        for _ in range(32):
            instance.sync()
        adapter = instance.session.adapter
        buffers = (
            id(adapter._geom_xpos_buf),
            id(adapter._geom_xmat_buf),
            id(adapter._body_xpos_buf),
            id(adapter._body_xmat_buf),
        )
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(300):
            instance.sync()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]

        assert buffers == (
            id(adapter._geom_xpos_buf),
            id(adapter._geom_xmat_buf),
            id(adapter._body_xpos_buf),
            id(adapter._body_xmat_buf),
        )
        assert after - before < 4 * 1024 * 1024
    finally:
        tracemalloc.stop()
        instance.release()


def test_model_component_inspector_tracks_structured_edits():
    instance = build_editor(vsync=False, width=960, height=640)
    try:
        assert instance.app.add_model(resolve("actuator_visuals"))
        model = next(node for node in instance.session.nodes if node.type is NodeType.MODEL)
        assert instance.session.submit(SelectNode(model.node_id))
        instance.sync()
        inspector = instance.app.panels.get("Inspector")
        assert inspector._component_cache["actuator"]
        assert "jointpos" in inspector._component_presets["sensor"]
        assert inspector._component_presets["custom"] == ("numeric", "text", "tuple")

        assert instance.session.submit(
            AddModelComponent(model.model_id, "sensor", "jointpos", "angle")
        )
        assert instance.session.submit(
            AddModelComponent(model.model_id, "custom", "tuple", "selection")
        )
        inspector._refresh_component_cache(
            SimpleNamespace(session=instance.session), model.model_id
        )
        assert [item.name for item in inspector._component_cache["sensor"]] == ["angle"]
        custom = inspector._component_cache["custom"]
        assert [(item.name, item.subtype) for item in custom] == [("selection", "tuple")]
        assert custom[0].path[0].fields[0].choices[0] == "body"
        inspector._begin_component_edit(custom[0])
        instance.sync()
        assert inspector._component_edit == custom[0]
    finally:
        instance.release()


def test_model_placement_requires_explicit_unlock_and_apply(monkeypatch):
    monkeypatch.setenv("FORGE_VIEWER_UI_SCALE", "1")
    instance = build_editor(vsync=False, width=1280, height=800)
    try:
        assert instance.app.add_model(resolve("actuator_visuals"))
        model = next(node for node in instance.session.nodes if node.type is NodeType.MODEL)
        assert instance.session.submit(SelectNode(model.node_id))
        for _ in range(3):
            instance.sync()
        _activate_panel(instance, "Inspector")

        primary = instance.session.adapter.primary
        compile_count = 0
        compile_model = primary._compile_composed_model

        def counted_compile():
            nonlocal compile_count
            compile_count += 1
            return compile_model()

        monkeypatch.setattr(primary, "_compile_composed_model", counted_compile)
        assert not instance.app.gizmo.visible
        assert "Edit Placement" in instance.app.gizmo.last_verdict.reason

        _trigger_button(instance, monkeypatch, "Edit Placement")
        assert instance.app.gizmo.model_placement_active(instance.session, model.model_id)
        instance.sync()
        assert instance.app.gizmo.visible
        assert instance.app.gizmo.preview_model_placement(
            instance.session,
            model.model_id,
            np.array((1.0, 0.0, 0.0)),
            np.eye(3),
        )
        instance.sync()
        assert compile_count == 0

        _trigger_button(instance, monkeypatch, "Cancel##model-placement")
        assert compile_count == 0
        assert not instance.app.gizmo.model_placement_active(instance.session, model.model_id)
        instance.sync()
        assert not instance.app.gizmo.visible

        _trigger_button(instance, monkeypatch, "Edit Placement")
        assert instance.app.gizmo.preview_model_placement(
            instance.session,
            model.model_id,
            np.array((1.0, 0.0, 0.0)),
            np.eye(3),
        )
        _trigger_button(instance, monkeypatch, "Apply Placement")

        assert compile_count == 1
        assert instance.session.can_undo
        assert not instance.app.gizmo.model_placement_active(instance.session, model.model_id)
    finally:
        instance.release()


def test_missing_workspace_resources_can_be_repaired_from_directory(tmp_path):
    instance = build_editor(vsync=False, width=960, height=640)
    document = tmp_path / "workspace" / "repair.forge.json"
    replacement = tmp_path / "recovered" / "robot.xml"
    try:
        assert instance.app.add_model(resolve("test_scene.xml"))
        assert instance.app.save_scene(document)
        payload = json.loads(document.read_text(encoding="utf-8"))
        payload["models"][0]["path"] = "old/robot.xml"
        document.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        replacement.parent.mkdir()
        replacement.write_text(resolve("test_scene.xml").read_text(), encoding="utf-8")

        result = instance.app.open_scene(document)
        assert not result.ok
        assert instance.app._resource_repair_path == document.resolve()

        class Dialog:
            def ready(self, _timeout):
                return True

            def result(self):
                return str(replacement.parent)

        instance.app._resource_repair_dialog = Dialog()
        instance.app._resource_repair_dialog_action = "search"
        instance.app._poll_resource_repair_dialog()
        instance.sync()
        instance.app._model_load_future.result(timeout=10.0)
        instance.sync()

        assert instance.session.asset_path == document.resolve()
        assert instance.session.scene_models[0].path == replacement.resolve()
        assert instance.app._resource_repair_path is None
    finally:
        instance.release()


def test_static_scene_file_menu_renders():
    instance = build_scene(Scene(), vsync=False, width=960, height=640)
    try:
        instance.sync()
        _click(instance, _item_center(instance, "begin_menu", "File"))
        _item_center(instance, "menu_item", "New Scene")
    finally:
        instance.release()
