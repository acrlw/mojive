from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ... import math3d
from ...adapters.base import FrameNeeds, NodeKind, SceneNode
from ...types import LightKind
from . import Panel, PanelContext, begin_kv_table, labeled

GIZMO_REFUSAL_RUNNING = "physics is running; pause to move things"
GIZMO_REFUSAL_DRIVEN = "this link is driven by joints; use the Joints panel"


def gizmo_refusal_reason(paused: bool, posable: bool) -> str | None:

    if not paused:
        return GIZMO_REFUSAL_RUNNING
    if not posable:
        return GIZMO_REFUSAL_DRIVEN
    return None


class InspectorPanel(Panel):
    name = "Inspector"
    default_open = True
    shortcut = "F4"

    def __init__(self) -> None:
        super().__init__()
        self.show_transform = True
        self._transform_velocity = False
        self.show_velocity = False
        self._rotation_node = -1
        self._rotation_euler = np.zeros(3, np.float64)
        self._rotation_matrix = np.eye(3, dtype=np.float64)

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(
            poses=True,
            qvel=(self.show_transform and self._transform_velocity) or self.show_velocity,
        )

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        node = s.selected_node
        if node is None:
            self._transform_velocity = False
            imgui.text_disabled("nothing selected")
            imgui.text_disabled("click an object in the viewport or the Hierarchy panel")
            return

        color = ctx.theme.node_color(node.kind)
        imgui.text_colored(imgui.ImVec4(*color), node.name or "?")
        imgui.same_line()
        imgui.text_disabled(f"({node.kind})")

        self._identity(node)
        if node.kind is NodeKind.LIGHT:
            self._light(ctx, node)
            return
        self._transform(ctx, node)
        self._gizmo_reason(ctx, node)
        self._velocity(ctx, node)
        self._material(ctx, node)

    def _identity(self, node: SceneNode) -> None:
        if begin_kv_table("insp_id"):
            labeled("node id", str(node.node_id))
            labeled("object id", str(node.object_id) if node.object_id else "— (not pickable)")
            labeled("body", str(node.body_index) if node.body_index >= 0 else "—")
            labeled("posable", "yes" if node.posable else "no")
            imgui.end_table()

    def _transform(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_transform = imgui.collapsing_header(
            "transform", imgui.TreeNodeFlags_.default_open
        )
        if not self.show_transform:
            return
        frame = ctx.session.frame
        pos, mat = _body_pose(frame.body_xpos, frame.body_xmat, node.body_index)
        if pos is None:
            imgui.text_disabled("no pose this frame")
            return
        mat = np.eye(3, dtype=np.float32) if mat is None else np.asarray(mat).reshape(3, 3)
        euler = self._continuous_euler(node.node_id, mat)
        editable = _pose_editable(
            ctx.session.adapter.caps.write_pose, ctx.session.paused, node.posable
        )

        compact = _compact_transform(imgui.get_content_region_avail().x, ctx.style_scale)
        flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        columns = 1 if compact else 7
        if not imgui.begin_table("insp_transform", columns, flags):
            return
        if compact:
            imgui.table_setup_column("transform", imgui.TableColumnFlags_.width_stretch)
        else:
            imgui.table_setup_column(
                "value", imgui.TableColumnFlags_.width_fixed, 96.0 * ctx.style_scale
            )
            for axis in "xyz":
                imgui.table_setup_column(
                    axis, imgui.TableColumnFlags_.width_fixed, 20.0 * ctx.style_scale
                )
                imgui.table_setup_column(
                    f"{axis} value", imgui.TableColumnFlags_.width_stretch, 1.0
                )

        pos_changed, new_pos = _vector_row(
            ctx,
            node,
            "position",
            np.asarray(pos, np.float64),
            editable=editable,
            speed=0.01,
            fmt="%.3f",
            compact=compact,
        )
        rot_changed, new_euler = _vector_row(
            ctx,
            node,
            "rotation",
            euler,
            editable=editable,
            speed=0.5,
            fmt="%.1f",
            compact=compact,
        )

        self._transform_velocity = _has_free_velocity(ctx.session.joints, node.body_index)
        velocity = _free_velocity(ctx.session.frame.qvel, ctx.session.joints, node.body_index)
        if velocity is not None:
            _vector_row(
                ctx,
                node,
                "linear velocity",
                velocity[0],
                editable=False,
                speed=0.0,
                fmt="%.3f",
                compact=compact,
            )
            _vector_row(
                ctx,
                node,
                "angular velocity",
                velocity[1],
                editable=False,
                speed=0.0,
                fmt="%.3f",
                compact=compact,
            )
        imgui.end_table()

        if pos_changed or rot_changed:
            rotation = math3d.euler_xyz_to_mat3(np.radians(new_euler))
            if rot_changed:
                self._rotation_euler[:] = new_euler
                self._rotation_matrix[:] = rotation
            ctx.submit(
                cmd.SetPose(
                    node.node_id,
                    np.asarray(new_pos, np.float32),
                    rotation,
                )
            )

    def _continuous_euler(self, node_id: int, matrix) -> np.ndarray:
        matrix = np.asarray(matrix, np.float64).reshape(3, 3)
        same_node = node_id == self._rotation_node
        if same_node and np.allclose(matrix, self._rotation_matrix, atol=2e-5):
            return self._rotation_euler.copy()
        reference = self._rotation_euler if same_node else None
        self._rotation_node = node_id
        self._rotation_euler[:] = _nearest_euler_degrees(matrix, reference)
        self._rotation_matrix[:] = matrix
        return self._rotation_euler.copy()

    def _gizmo_reason(self, ctx: PanelContext, node: SceneNode) -> None:
        reason = gizmo_refusal_reason(ctx.session.paused, node.posable)
        write_pose = ctx.session.adapter.caps.write_pose
        imgui.separator()
        if not write_pose:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{ctx.session.adapter.caps.name} cannot write poses",
            )
            return
        if reason is None:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.primary),
                f"gizmo: active ({ctx.gizmo.space} frame; g/r mode, t frame)",
            )
            return
        imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "gizmo hidden")
        imgui.text_wrapped(reason)

    def _velocity(self, ctx: PanelContext, node: SceneNode) -> None:

        self.show_velocity = imgui.collapsing_header("velocity")
        if not self.show_velocity:
            return
        qvel = ctx.session.frame.qvel
        if qvel is None:
            imgui.text_disabled("waiting for the next frame (qvel is produced on demand)")
            return
        dofs = [j for j in ctx.session.joints if j.body == node.body_index]
        if not dofs:
            imgui.text_disabled("no joint on this body")
            return
        if begin_kv_table("insp_vel"):
            for j in dofs:
                lo = j.qvel_adr
                hi = min(lo + max(1, j.dof), len(qvel))
                labeled(j.name or f"dof{lo}", "  ".join(f"{v:+.4f}" for v in qvel[lo:hi]))
            imgui.end_table()

    def _material(self, ctx: PanelContext, node: SceneNode) -> None:
        if not imgui.collapsing_header("material"):
            return
        src = ctx.session.source
        if src is None or node.body_index < 0 or len(src.geom_body) == 0:
            imgui.text_disabled("no geometry")
            return
        idx = np.flatnonzero(np.asarray(src.geom_body) == node.body_index)
        if len(idx) == 0:
            imgui.text_disabled("no geometry on this body")
            return
        imgui.text_disabled(f"{len(idx)} geom instance(s)")
        if not begin_kv_table("insp_mat"):
            return
        for i in idx[:8]:
            mi = src.geom_material[i] if i < len(src.geom_material) else -1
            mat = src.materials[mi] if 0 <= mi < len(src.materials) else None
            rgba = src.geom_rgba[i] if i < len(src.geom_rgba) else None
            name = mat.name if mat is not None else "—"
            tail = (
                f"  rgba {rgba[0]:.2f} {rgba[1]:.2f} {rgba[2]:.2f} {rgba[3]:.2f}"
                if rgba is not None
                else ""
            )
            labeled(str(src.geom_mesh[i]) if i < len(src.geom_mesh) else f"geom{i}", name + tail)
        imgui.end_table()

    def _light(self, ctx: PanelContext, node: SceneNode) -> None:
        source = ctx.session.source
        index = node.light_index
        if source is None or not 0 <= index < len(source.lights.lights):
            imgui.text_disabled("light data is unavailable")
            return
        light = source.lights.lights[index]

        changed, active = imgui.checkbox("enabled", light.active)
        kind_changed, kind_index = imgui.combo(
            "type", int(light.kind), ["directional", "point", "spot", "area"]
        )
        changed |= kind_changed

        intensity = float(np.max(light.diffuse))
        color = light.diffuse / intensity if intensity > 0.0 else np.ones(3, np.float32)
        color_changed, color = imgui.color_edit3("color", color)
        intensity_changed, intensity = imgui.drag_float(
            "intensity", intensity, 0.01, 0.0, 10.0, "%.2f"
        )
        changed |= color_changed or intensity_changed
        diffuse = np.asarray(color, np.float32) * intensity

        specular_changed, specular = imgui.color_edit3("specular", light.specular)
        ambient_changed, ambient = imgui.color_edit3("ambient", light.ambient)
        pos_changed, position = imgui.drag_float3(
            "position (local)", light.position, 0.01, 0.0, 0.0, "%.3f"
        )
        dir_changed, direction = imgui.drag_float3(
            "direction (local)", light.direction, 0.01, 0.0, 0.0, "%.3f"
        )
        changed |= specular_changed or ambient_changed or pos_changed or dir_changed

        range_changed = cutoff_changed = exponent_changed = attenuation_changed = False
        area_changed = False
        light_range, cutoff, exponent = light.range, light.cutoff, light.exponent
        area_radius = light.area_radius
        attenuation = light.attenuation
        kind = LightKind(kind_index)
        if kind in (LightKind.POINT, LightKind.SPOT, LightKind.AREA):
            range_changed, light_range = imgui.drag_float(
                "range (0 = unlimited)", light.range, 0.05, 0.0, 10000.0, "%.2f"
            )
            attenuation_changed, attenuation = imgui.drag_float3(
                "attenuation", light.attenuation, 0.01, 0.0, 100.0, "%.3f"
            )
        if kind is LightKind.SPOT:
            cutoff_changed, cutoff = imgui.drag_float(
                "cutoff", light.cutoff, 0.25, 0.1, 89.9, "%.1f deg"
            )
            exponent_changed, exponent = imgui.drag_float(
                "falloff exponent", light.exponent, 0.1, 0.0, 100.0, "%.1f"
            )
        if kind is LightKind.AREA:
            area_changed, area_radius = imgui.drag_float(
                "source radius", light.area_radius, 0.01, 0.0, 1000.0, "%.3f"
            )
        shadow_changed, cast_shadow = imgui.checkbox("cast shadow", light.cast_shadow)
        changed |= (
            range_changed
            or cutoff_changed
            or exponent_changed
            or attenuation_changed
            or area_changed
            or shadow_changed
        )
        if changed:
            ctx.submit(
                cmd.SetLight(
                    index,
                    replace(
                        light,
                        kind=kind,
                        active=active,
                        diffuse=diffuse,
                        specular=np.asarray(specular, np.float32),
                        ambient=np.asarray(ambient, np.float32),
                        position=np.asarray(position, np.float32),
                        direction=np.asarray(direction, np.float32),
                        attenuation=np.asarray(attenuation, np.float32),
                        range=float(light_range),
                        area_radius=float(area_radius),
                        cutoff=float(cutoff),
                        exponent=float(exponent),
                        cast_shadow=cast_shadow,
                    ),
                )
            )


def _body_pose(xpos, xmat, body_index: int):
    if xpos is None or body_index < 0 or body_index >= len(xpos):
        return None, None
    mat = xmat[body_index] if xmat is not None and body_index < len(xmat) else None
    return xpos[body_index], mat


def _pose_editable(write_pose: bool, paused: bool, posable: bool) -> bool:
    return bool(write_pose and paused and posable)


def _nearest_euler_degrees(matrix, reference=None) -> np.ndarray:

    base = np.degrees(math3d.mat3_to_euler_xyz(matrix)).astype(np.float64)
    if reference is None:
        return base
    reference = np.asarray(reference, np.float64)
    alternate = np.array((base[0] + 180.0, 180.0 - base[1], base[2] + 180.0))
    candidates = []
    for value in (base, alternate):
        candidates.append(value + 360.0 * np.round((reference - value) / 360.0))
    return min(candidates, key=lambda value: float(np.linalg.norm(value - reference)))


def _free_velocity(qvel, joints, body_index: int):
    if qvel is None:
        return None
    joint = next(
        (j for j in joints if j.body == body_index and j.kind == "free" and j.dof >= 6), None
    )
    if joint is None or joint.qvel_adr + 6 > len(qvel):
        return None
    values = np.asarray(qvel[joint.qvel_adr : joint.qvel_adr + 6], np.float64)
    return values[:3], values[3:]


def _has_free_velocity(joints, body_index: int) -> bool:
    return any(j.body == body_index and j.kind == "free" and j.dof >= 6 for j in joints)


def _compact_transform(width: float, style_scale: float) -> bool:
    return float(width) < 420.0 * float(style_scale)


def _vector_row(
    ctx: PanelContext,
    node: SceneNode,
    name: str,
    values,
    *,
    editable: bool,
    speed: float,
    fmt: str,
    compact: bool,
) -> tuple[bool, np.ndarray]:
    out = np.asarray(values, np.float64).copy()
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text(name)
    group_hovered = imgui.is_item_hovered()
    if group_hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
        imgui.set_clipboard_text(_format_vector(out))
    if group_hovered:
        imgui.set_tooltip("right-click: copy XYZ")

    if compact:
        row_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        if not imgui.begin_table(f"##{name}_axes_{node.node_id}", 6, row_flags):
            return False, out
        for axis in "xyz":
            imgui.table_setup_column(
                axis, imgui.TableColumnFlags_.width_fixed, 20.0 * ctx.style_scale
            )
            imgui.table_setup_column(f"{axis} value", imgui.TableColumnFlags_.width_stretch, 1.0)

    changed = False
    for axis, label in enumerate("XYZ"):
        imgui.table_next_column()
        color = ctx.theme.axis_color(axis)
        hovered_color = _lift_color(color, 0.20)
        active_color = _lift_color(color, 0.32)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*color))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hovered_color))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*active_color))
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ctx.theme.bg_child))
        imgui.begin_disabled(not editable)
        reset = imgui.button(
            f"{label}##{name}_{axis}_{node.node_id}",
            imgui.ImVec2(imgui.get_content_region_avail().x, 0.0),
        )
        imgui.end_disabled()
        imgui.pop_style_color(4)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip("click: reset to 0" if editable else "read only")
        if reset:
            out[axis] = 0.0
            changed = True

        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        imgui.begin_disabled(not editable)
        edited, value = imgui.drag_float(
            f"##{name}_{axis}_{node.node_id}", float(out[axis]), speed, 0.0, 0.0, fmt
        )
        imgui.end_disabled()
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
            imgui.set_clipboard_text(fmt % out[axis])
        if hovered:
            hint = "drag: edit · right-click: copy" if editable else "read only · right-click: copy"
            imgui.set_tooltip(hint)
        if edited:
            out[axis] = value
            changed = True
    if compact:
        imgui.end_table()
    return changed, out


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)


def _lift_color(color, amount: float):

    return (*(c + (1.0 - c) * amount for c in color[:3]), color[3])
