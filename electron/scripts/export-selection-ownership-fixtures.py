"""Export real native selection IDs and matching rendered instances.

Run from the repository root with its Python environment. Temporary MuJoCo
authoring files are removed after inspecting positive geometry/model identities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mojive import Scene, commands as cmd
from mojive.adapters.base import NodeType
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.worlds import WorldInstances
from mojive.session import Session


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/selection-ownership-native.json"
AUTHORING_XML = """<mujoco model="ownership">
  <worldbody>
    <geom name="finite_floor" type="plane" size="2 2 .05"/>
    <body name="body" pos="0 0 1">
      <joint name="hinge" axis="0 1 0"/>
      <geom name="cube" type="box" size=".2 .3 .4"/>
      <camera name="camera" pos="0 -2 1"/>
      <light name="light" pos="0 0 2"/>
    </body>
  </worldbody>
</mujoco>"""
PAYLOAD_XML = """<mujoco model="payload"><worldbody>
  <body name="payload"><geom name="payload_geom" type="sphere" size=".1"/></body>
</worldbody></mujoco>"""


def capture(session, name, selected_nodes):
    source = session.source
    instances = [
        {
            "index": index,
            "objectId": int(source.geom_object_id[index]),
            "bodyIndex": int(source.geom_body[index]),
            "geomIndex": int(source.geom_source[index]),
            "shape": source.geom_mesh[index].shape.value,
        }
        for index in range(source.instance_count)
    ]
    selections = []
    for node in selected_nodes:
        result = session.submit(cmd.SelectNode(node.node_id))
        assert result.ok, result.message
        highlight = int(session.selection_highlight_object_id)
        matching = [item["index"] for item in instances if highlight > 0 and item["objectId"] == highlight]
        selections.append({
            "name": node.name,
            "kind": node.type.value,
            "nodeId": int(node.node_id),
            "parentId": int(node.parent),
            "bodyIndex": int(node.body_index),
            "geomIndex": int(node.geom_index),
            "objectId": int(node.object_id),
            "selectedObjectId": int(session.selected),
            "highlightObjectId": highlight,
            "matchingInstances": matching,
        })
    return {"id": name, "instances": instances, "selections": selections}


def main():
    adapter = MuJoCoAdapter(ROOT / "assets/joint_gizmo.xml")
    try:
        session = Session(adapter)
        wanted = {"01_revolute_anchor", "01_revolute", "01_revolute_y", "02_prismatic_x", "03_ball_xyz"}
        body_ids = {node.body_index for node in session.nodes if node.name in wanted}
        nodes = [node for node in session.nodes if node.name in wanted
                 or node.type in (NodeType.WORLD, NodeType.CAMERA, NodeType.LIGHT)
                 or node.type is NodeType.GEOM and node.body_index in body_ids]
        joint_scene = capture(session, "joint_gizmo", nodes)
        anchor = next(item for item in joint_scene["selections"] if item["name"] == "01_revolute_anchor")
        child = next(item for item in joint_scene["selections"] if item["name"] == "01_revolute")
        joint = next(item for item in joint_scene["selections"] if item["name"] == "01_revolute_y")
        assert anchor["highlightObjectId"] != child["highlightObjectId"]
        assert joint["matchingInstances"] == child["matchingInstances"]
        assert all(not item["matchingInstances"] for item in joint_scene["selections"] if item["kind"] == "geom" and item["objectId"] == 0)
    finally:
        adapter.release()

    with TemporaryDirectory(prefix="selection-ownership-", dir=ROOT / "output") as temporary:
        root, payload = Path(temporary) / "root.xml", Path(temporary) / "payload.xml"
        root.write_text(AUTHORING_XML)
        payload.write_text(PAYLOAD_XML)
        adapter = MuJoCoAdapter(root)
        try:
            session = Session(adapter, root)
            assert session.submit(cmd.Pause()).ok
            cube = next(node for node in session.nodes if node.name == "cube")
            duplicated = session.submit(cmd.DuplicateModelElement(cube.node_id))
            assert duplicated.ok, duplicated.message
            attached = session.submit(cmd.AddSceneModel(payload, (2, 0, 0)))
            assert attached.ok, attached.message
            authored = capture(session, "authored_geometry_and_model", session.nodes)
            positive = [item for item in authored["selections"] if item["kind"] == "geom" and item["objectId"] > 0]
            assert len(positive) >= 2 and all(item["matchingInstances"] for item in positive)
            models = [item for item in authored["selections"] if item["kind"] == "model"]
            assert models and all(item["highlightObjectId"] > 0 and not item["matchingInstances"] for item in models)
        finally:
            adapter.release()

    scene = Scene()
    scene.box(name="box")
    scene.sphere(name="sphere", position=(1, 0, 0))
    worlds = Session(WorldInstances(scene.source, scene.frame, np.array(((0, 0, 0), (3, 0, 0))), world_ids=(7, 11)))
    repeated = capture(worlds, "backend_neutral_world_instances", worlds.nodes)
    assert all(item["matchingInstances"] for item in repeated["selections"] if item["kind"] == "model")

    paths = ("python/session/core.py", "python/session/dispatch/transforms.py",
             "python/adapters/mujoco/source.py", "python/adapters/worlds.py",
             "python/render/opengl/shaders/scene.vert", "assets/joint_gizmo.xml")
    result = {
        "schema": 1,
        "source": {
            "generator": "electron/scripts/export-selection-ownership-fixtures.py",
            "files": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
            "authoringXML": AUTHORING_XML,
            "attachedModelXML": PAYLOAD_XML,
        },
        "rule": "SelectNode copies node.object_id; positive IDs select equal rendered IDs. Only zero-ID JOINT selection falls back to a same-body ancestor. Zero never highlights.",
        "notes": [
            "Native legacy GEOM object_id=0 selects no fill/outline; Electron geoms already have explicit identities and use their own geometry.",
            "A native positive geometry ID replaces body ownership for its instances, including after geometry duplication.",
            "MuJoCo MODEL containers have reserved IDs without matching meshes; backend-neutral WorldInstances MODEL nodes deliberately own matching meshes.",
            "Electron robot nodes represent imported model containers, not MuJoCo ROBOT bodies or WorldInstances models.",
        ],
        "scenes": [joint_scene, authored, repeated],
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Exported {sum(len(scene['selections']) for scene in result['scenes'])} native selection cases across {len(result['scenes'])} scene adapters.")


if __name__ == "__main__":
    main()
