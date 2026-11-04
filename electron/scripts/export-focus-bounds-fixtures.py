"""Export hierarchy/local bounds from native scene geometry, independent of Three.js."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from mojive import Scene
from mojive.adapters.base import NodeType, SceneNode
from mojive.math3d import axis_angle_to_mat3
from mojive.scene.bounds import SceneBounds, _node_local_bounds, _node_world_bounds

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/focus-bounds-native.json"


def pose(position, angle=0):
    matrix=np.eye(4)
    matrix[:3, :3]=axis_angle_to_mat3((0,0,1), angle)
    matrix[:3, 3]=position
    return matrix


def centered(bounds):
    return None if bounds is None else {"center": bounds.center.tolist(), "half": bounds.half_extent.tolist()}


def main():
    scene=Scene()
    sizes=((.5,1,.2),(.3,.4,.6),(2,.5,.25),(100,100,.01))
    for size in sizes:
        scene.box(size=size)
    source,frame=scene.source,scene.frame
    body1=pose((2,-1,3), np.pi/4)
    body2=pose((8,4,1), -np.pi/6)
    geom_world=[pose((2,-1,3)),body1@pose((2,0,.5),.2),body2@pose((.5,-.25,0),.4),pose((0,0,-.05))]
    source.geom_body[:]=[1,1,2,0]
    source.geom_infinite_plane[:]=[False,False,False,True]
    frame.body_xpos[:3]=[np.zeros(3),body1[:3,3],body2[:3,3]]
    frame.body_xmat[:3]=[np.eye(3),body1[:3,:3],body2[:3,:3]]
    frame.geom_xpos[:]=[matrix[:3,3] for matrix in geom_world]
    frame.geom_xmat[:]=[matrix[:3,:3] for matrix in geom_world]
    nodes=[
        SceneNode(0,"world",NodeType.WORLD,children=[100,204],body_index=0),
        SceneNode(100,"model",NodeType.MODEL,parent=0,children=[101]),
        SceneNode(101,"body",NodeType.LINK,parent=100,children=[201,202,102,301],body_index=1),
        SceneNode(201,"first",NodeType.GEOM,parent=101,body_index=1,geom_index=0),
        SceneNode(202,"hidden",NodeType.GEOM,parent=101,body_index=1,geom_index=1,visible=False),
        SceneNode(102,"child",NodeType.LINK,parent=101,children=[203],body_index=2),
        SceneNode(203,"child_geom",NodeType.GEOM,parent=102,body_index=2,geom_index=2),
        SceneNode(204,"infinite_ground",NodeType.GEOM,parent=0,body_index=0,geom_index=3),
        SceneNode(301,"camera",NodeType.CAMERA,parent=101,body_index=1),
    ]
    source.nodes=nodes
    poses={0:np.eye(4),100:np.eye(4),101:body1,102:body2,301:body1@pose((0,2,4))}
    for i,node_id in enumerate((201,202,203,204)):
        poses[node_id]=geom_world[i]
    descriptions=[]
    for node in nodes:
        item={"id":str(node.node_id),"kind":"robot" if node.type is NodeType.MODEL else node.type.value,
              "parent":None if node.parent<0 else str(node.parent),"matrix":poses[node.node_id].tolist(),"visible":node.visible}
        if node.type is NodeType.GEOM:
            item.update(size=source.geom_size[node.geom_index].tolist(),infinite=bool(source.geom_infinite_plane[node.geom_index]))
        descriptions.append(item)
    checks=[{"id":str(node.node_id),"world":centered(_node_world_bounds(source,frame,node,nodes)),"local":centered(_node_local_bounds(source,frame,node))} for node in nodes if node.type is not NodeType.WORLD]
    world=SceneBounds(source).world(frame)
    result={
        "schema":1,
        "source":{"generator":"electron/scripts/export-focus-bounds-fixtures.py","sha256":hashlib.sha256((ROOT/"python/scene/bounds.py").read_bytes()).hexdigest()},
        "nodes":descriptions,"checks":checks,
        "scene":{"center":((world.minimum+world.maximum)*.5).tolist(),"half":((world.maximum-world.minimum)*.5).tolist()},
    }
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    DESTINATION.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Exported native bounds for {len(checks)} nodes in local/world frames.")


if __name__=="__main__":
    main()
