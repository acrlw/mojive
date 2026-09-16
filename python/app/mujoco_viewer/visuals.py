"""Translate MuJoCo camera, visibility and user geometry into Mojive presentation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mojive.app.mujoco_visuals import (
    FRAME_MODES,
    LABEL_MODES,
    RND_FLAGS,
    VIS_FLAGS,
    apply_render_options,
    camera_view,
)
from mojive.render.debugdraw import Occlusion


@dataclass(frozen=True)
class UserGeometry:
    kind: int
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    color: np.ndarray
    label: str


_SUPPORTED_GEOMS = {
    int(getattr(mujoco.mjtGeom, "mjGEOM_" + name))
    for name in (
        "PLANE",
        "SPHERE",
        "CAPSULE",
        "ELLIPSOID",
        "CYLINDER",
        "BOX",
        "ARROW",
        "LINE",
        "LABEL",
    )
}


def snapshot_geometries(scene):
    if not 0 <= scene.ngeom <= scene.maxgeom:
        raise ValueError("user_scn.ngeom is outside the scene capacity")
    result = []
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if int(geom.type) not in _SUPPORTED_GEOMS or geom.matid >= 0:
            raise NotImplementedError(
                f"user_scn geometry {index}: Mojive supports untextured primitive geoms; "
                "mesh, heightfield and material-bound user geoms require Mojive Scene APIs"
            )
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE and (geom.size[0] <= 0 or geom.size[1] <= 0):
            raise NotImplementedError("Infinite user_scn planes require a Mojive Scene plane")
        if geom.type == mujoco.mjtGeom.mjGEOM_ARROW and geom.rgba[3] != 1:
            raise NotImplementedError("user_scn arrows currently require an opaque color")
        result.append(
            UserGeometry(
                int(geom.type),
                geom.size.copy(),
                geom.pos.copy(),
                geom.mat.reshape(3, 3).copy(),
                geom.rgba.copy(),
                geom.label,
            )
        )
    return tuple(result)


def draw_user_geometries(draw, geoms):
    layer = draw.layer("mujoco.user_scn", Occlusion.DEPTH)
    layer.clear()
    kind = mujoco.mjtGeom
    for index, geom in enumerate(geoms):
        name = str(index)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = geom.position
        scale = geom.size.copy()
        if geom.kind == kind.mjGEOM_SPHERE:
            scale[:] = scale[0]
        if geom.kind in (kind.mjGEOM_CYLINDER, kind.mjGEOM_CAPSULE):
            scale = np.array((scale[0], scale[0], scale[1]))
        if geom.kind == kind.mjGEOM_PLANE:
            if scale[0] <= 0 or scale[1] <= 0:
                raise NotImplementedError("Infinite user_scn planes require a Mojive Scene plane")
            scale[2] = 1e-5
        transform[:3, :3] = geom.rotation @ np.diag(scale)
        if geom.kind in (kind.mjGEOM_SPHERE, kind.mjGEOM_ELLIPSOID):
            layer.sphere(name, transform, geom.color)
        elif geom.kind in (kind.mjGEOM_BOX, kind.mjGEOM_PLANE):
            layer.box(name, transform, geom.color)
        elif geom.kind in (kind.mjGEOM_CYLINDER, kind.mjGEOM_CAPSULE):
            layer.cylinder(name, transform, geom.color)
            if geom.kind == kind.mjGEOM_CAPSULE:
                for sign in (-1, 1):
                    cap = np.eye(4, dtype=np.float32)
                    cap[:3, :3] *= scale[0]
                    cap[:3, 3] = geom.position + sign * scale[2] * geom.rotation[:, 2]
                    layer.sphere(f"{name}.cap{sign}", cap, geom.color)
        elif geom.kind in (kind.mjGEOM_ARROW, kind.mjGEOM_LINE):
            end = geom.position + geom.size[2] * geom.rotation[:, 2]
            if geom.kind == kind.mjGEOM_ARROW:
                layer.arrow_3d(
                    name,
                    geom.position,
                    end,
                    geom.color,
                    float(geom.size[0]),
                    head_radius=float(geom.size[1]),
                )
            else:
                layer.line(name, geom.position, end, geom.color, max(1.0, float(geom.size[0])))
        if geom.label:
            layer.text(f"{name}.label", geom.position, geom.label, geom.color)


class Presentation:
    """Apply queued visual edits on the UI thread and read back navigation changes."""

    def __init__(self, exchange, scene):
        self.exchange = exchange
        self.scene = scene
        self.camera_scene = mujoco.MjvScene(exchange.display_model, maxgeom=0)
        self.last_view = None

    def before_frame(self, viewer):
        exchange = self.exchange
        adapter = viewer.session.adapter
        adapter.refresh_model_visuals()
        if exchange.options_changed:
            adapter.apply_scene_option(exchange.display_opt)
            apply_render_options(viewer.backend, exchange.display_opt, self.scene)
            exchange.options_changed = False
        cam = exchange.display_cam
        if exchange.camera_changed or cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
            mujoco.mjv_updateCamera(
                exchange.display_model, exchange.display_data, cam, self.camera_scene
            )
            view = camera_view(self.camera_scene, exchange.display_model, viewer.app.camera.aspect)
            # MuJoCo's lookat is the orbit pivot, not an arbitrary point along the ray.
            if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                from dataclasses import replace

                view = replace(view, target=np.array(cam.lookat, np.float32))
            if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
                viewer.app.set_viewport_camera(view, camera_id=cam.fixedcamid)
            else:
                viewer.set_camera(view)
            exchange.camera_changed = False
        self.last_view = viewer.app.camera.view()

    def after_frame(self, viewer):
        exchange = self.exchange
        orbit = viewer.app.camera
        model_camera = viewer.app._model_camera_id
        view = viewer.session.camera if model_camera >= 0 else orbit.view()
        if 0 <= model_camera < exchange.display_model.ncam:
            exchange.display_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            exchange.display_cam.fixedcamid = model_camera
        elif self.last_view is not None and (
            not np.array_equal(view.eye, self.last_view.eye)
            or not np.array_equal(view.target, self.last_view.target)
            or view.orthographic != self.last_view.orthographic
            or exchange.display_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
        ):
            cam = exchange.display_cam
            delta = view.eye - view.target
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = view.target
            cam.distance = distance
            cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0]))) + 180.0
            cam.elevation = -float(np.degrees(np.arcsin(np.clip(delta[2] / distance, -1, 1))))
            cam.orthographic = int(view.orthographic)
        opt = exchange.display_opt
        for name, flag in VIS_FLAGS.items():
            if viewer.backend.caps.supports(flag):
                opt.flags[int(getattr(mujoco.mjtVisFlag, name))] = viewer.backend.get_flag(flag)
        for name, flag in RND_FLAGS.items():
            if viewer.backend.caps.supports(flag):
                self.scene.flags[int(getattr(mujoco.mjtRndFlag, name))] = viewer.backend.get_flag(
                    flag
                )
        for name, mode in LABEL_MODES.items():
            if viewer.backend.get_label_mode() == mode:
                opt.label = int(getattr(mujoco.mjtLabel, name))
        for name, mode in FRAME_MODES.items():
            if viewer.backend.get_frame_mode() == mode:
                opt.frame = int(getattr(mujoco.mjtFrame, name))
        opt.bvh_depth = viewer.backend.get_bvh_depth()
        groups = {
            "geom": "geomgroup",
            "site": "sitegroup",
            "joint": "jointgroup",
            "tendon": "tendongroup",
            "flex": "flexgroup",
            "skin": "skingroup",
        }
        for group in viewer.session.adapter.visual_groups():
            field = groups.get(group.category)
            if field is not None:
                getattr(opt, field)[:] = group.visible
