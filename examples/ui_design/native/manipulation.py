"""Reference transform gestures using the production handle hit geometry."""

from __future__ import annotations

import copy

import numpy as np
from imgui_bundle import imgui

from mojive import math3d
from mojive.interaction.gizmo import GizmoHandle, GizmoMode, hit_test
from mojive.ui.gizmo.projection import _cursor_plane


class Manipulation:
    def __init__(self):
        self.drag = None

    def update(self, study, rect, cursor, enabled):
        p = study.preview
        e = p.entity
        if not e or not p.gizmos or p.tool == "select" or study.playing:
            p.hovered = GizmoHandle.NONE
            return False
        cam = p.camera.view()
        origin = np.array(e["position"], float)
        basis = (
            math3d.euler_xyz_to_mat3(np.radians(e["rotation"]))
            if study.space == "Body"
            else np.eye(3)
        )
        modes = {
            "move": GizmoMode.TRANSLATE,
            "rotate": GizmoMode.ROTATE,
            "scale": GizmoMode.DIMENSIONS,
        }
        p.hovered = (
            hit_test(cam, origin, basis, rect, cursor, modes[p.tool], study.s)[0]
            if enabled
            else GizmoHandle.NONE
        )
        if (
            self.drag is None
            and enabled
            and imgui.is_mouse_clicked(0)
            and p.hovered != GizmoHandle.NONE
        ):
            handle = p.hovered
            axis_index = None
            if handle in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
                axis_index = int(handle) - 1
            if handle in (GizmoHandle.ROTATE_X, GizmoHandle.ROTATE_Y, GizmoHandle.ROTATE_Z):
                axis_index = int(handle) - 8
            forward = math3d.normalize(cam.target - cam.eye)
            axis = basis[:, axis_index] if axis_index is not None else forward
            if p.tool == "rotate":
                normal = axis
            elif handle in (GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY):
                normal = basis[:, int(handle) - 4]
            elif axis_index is not None:
                normal = math3d.normalize(forward - axis * np.dot(forward, axis))
            else:
                normal = forward
            point = _cursor_plane(cam, rect, cursor, origin, normal)
            if point is not None:
                self.drag = {
                    "before": copy.deepcopy(p.entities),
                    "entity": e,
                    "origin": origin,
                    "rotation": math3d.euler_xyz_to_mat3(np.radians(e["rotation"])),
                    "scale": np.array(e["scale"]),
                    "normal": normal,
                    "axis": axis,
                    "axis_index": axis_index,
                    "point": point,
                    "cursor": np.array(cursor),
                    "handle": handle,
                    "tool": p.tool,
                }
                p.active = handle
        drag = self.drag
        if drag is None:
            return p.hovered != GizmoHandle.NONE
        if imgui.is_key_pressed(imgui.Key.escape):
            p.entities[:] = drag["before"]
            self.drag = None
            p.active = GizmoHandle.NONE
            return True
        if imgui.is_mouse_released(0):
            study.change(drag["before"])
            self.drag = None
            p.active = GizmoHandle.NONE
            return True
        if not imgui.is_mouse_down(0):
            return True
        point = _cursor_plane(cam, rect, cursor, drag["origin"], drag["normal"])
        if point is None:
            return True
        e = drag["entity"]
        tool = drag["tool"]
        axis = drag["axis"]
        axis_index = drag["axis_index"]
        if tool == "move":
            delta = point - drag["point"]
            if axis_index is not None:
                delta = axis * np.dot(delta, axis)
            if study.snap:
                delta = np.round(delta / 0.1) * 0.1
            e["position"] = (drag["origin"] + delta).tolist()
        elif tool == "rotate":
            if drag["handle"] == GizmoHandle.ROTATE_TRACKBALL:
                delta = (np.array(cursor) - drag["cursor"]) * 0.008
                right, up, _ = math3d.camera_basis(cam)
                rot = math3d.axis_angle_to_mat3(up, float(delta[0])) @ math3d.axis_angle_to_mat3(
                    right, float(delta[1])
                )
            else:
                a = math3d.normalize(drag["point"] - drag["origin"])
                b = math3d.normalize(point - drag["origin"])
                angle = float(np.arctan2(np.dot(axis, np.cross(a, b)), np.dot(a, b)))
                if study.snap:
                    angle = round(angle / np.radians(15)) * np.radians(15)
                rot = math3d.axis_angle_to_mat3(axis, angle)
            e["rotation"] = np.degrees(math3d.mat3_to_euler_xyz(rot @ drag["rotation"])).tolist()
        else:
            delta = (cursor[0] - drag["cursor"][0] - (cursor[1] - drag["cursor"][1])) / (
                140 * study.s
            )
            value = drag["scale"].copy()
            factor = np.exp(delta)
            if axis_index is None:
                value *= factor
            else:
                value[axis_index] *= factor
            if study.snap:
                value = np.round(value / 0.1) * 0.1
            e["scale"] = np.clip(value, 0.01, 100).tolist()
        return True
