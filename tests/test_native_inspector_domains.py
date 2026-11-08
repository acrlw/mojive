"""Inspector domains follow the inspected entity without requiring a physics backend."""

from types import SimpleNamespace

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.adapters.static import StaticSceneAdapter
from mojive.session import Session
from mojive.types import CameraView, Light
from mojive.ui.workspace.camera_panel import scene_camera_selected, select_scene_camera
from mojive.ui.workspace.inspector_components import inspector_tabs


def test_static_entities_keep_appearance_without_empty_physics_or_joint_tabs():
    scene = Scene()
    scene.box(name="box")
    scene.add_camera("camera", CameraView())
    scene.add_light("light", Light())
    session = Session(StaticSceneAdapter(scene))
    try:
        camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
        assert session.submit(cmd.SelectNode(camera.node_id))
        for node in session.nodes:
            colorable = node.type is NodeType.GEOM
            tabs = inspector_tabs(session, {"node_id": node.node_id, "colorable": colorable})
            assert tabs == (("Properties", "Material") if colorable else ("Properties",))
        assert session.selected_node is camera
    finally:
        session.release()


def test_scene_camera_selection_survives_frames_without_models_and_yields_to_entity_selection():
    scene = Scene()
    obj = scene.box(name="box")
    session = Session(StaticSceneAdapter(scene))
    views = []
    workspace = SimpleNamespace(
        app=SimpleNamespace(
            session=session, select_model_camera=lambda camera, **kwargs: views.append(camera)
        ),
        open={"Inspector": False},
        pinned="old",
        inspector_tab="Physics",
    )
    try:
        select_scene_camera(workspace)
        assert views == [-1]
        assert workspace.open["Inspector"] and workspace.pinned is None
        assert workspace.inspector_tab == "Properties"
        assert scene_camera_selected(workspace)
        assert not session.scene_models
        assert session.submit(cmd.Select(obj.object_id))
        assert not scene_camera_selected(workspace)
    finally:
        session.release()
