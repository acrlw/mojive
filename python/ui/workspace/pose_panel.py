"""Native joint pose fields backed by atomic generalized-position commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.numeric_text import fit_numeric_text, numeric_text
from mojive.ui.panels import copyable_name_item, pointer_pressed, publish_focus_item_hint
from mojive.ui.panels.inspector import _nearest_euler_degrees
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.theme import THEME, mix_color
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.widgets import MUTED, TEXT, NumericEditState

_QPOS_SIZE = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}
_JOINT_LABELS = {
    "hinge": "Hinge joint",
    "slide": "Slide joint",
    "ball": "Ball joint",
    "free": "Free joint",
}


@dataclass(frozen=True)
class _SummaryLayout:
    label_width: float
    cell_width: float
    line_height: float
    gap: float
    label_gap: float
    columns: int
    above: bool
    axis_width: float
    badge_size: float

    @property
    def height(self):
        return self.line_height * ((3 // self.columns) + self.above) + self.label_gap


@dataclass(frozen=True)
class _HeadingLayout:
    heading_width: float
    disclosure_width: float
    gap: float


def joint_icon(kind):
    """Use motion-specific symbols for the four supported joint types."""
    return "joint-" + kind if kind in _QPOS_SIZE else "joint"


def joint_heading_context(joint, translate=str):
    """Expose an editing constraint rather than repeating the joint-type glyph."""
    if joint.type in ("hinge", "slide"):
        if not joint.limited:
            return translate("Unbounded")
        low, high = joint.range
        if not (isfinite(low) and isfinite(high) and low < high):
            return translate("Invalid joint range")
        if joint.type == "hinge":
            low, high = np.degrees((low, high))
        values = (numeric_text(value, "%.4g").replace("-", "−") for value in (low, high))
        return "…".join(values) + ("°" if joint.type == "hinge" else " m")
    return f"{joint.dof} {translate('DoF')}" if joint.dof > 0 else ""


def _selected_joint(selected, joint):
    if selected is None:
        return False
    return (
        selected.joint_index == joint.joint_id
        if selected.type is NodeType.JOINT
        else selected.body_index == joint.body
    )


def joint_pose_command(joint, qpos, *, position=None, rotation=None):
    """Convert displayed XYZ degrees to a normalized wxyz joint quaternion.

    Ball coordinates are relative to the joint frame. Free coordinates already
    describe the world pose, including any composed-model placement.
    """
    size = _QPOS_SIZE.get(joint.type)
    start = joint.qpos_adr
    if size is None or start < 0 or qpos is None or start + size > len(qpos):
        raise ValueError("Joint positions are unavailable")
    if joint.type not in ("ball", "free") or (position is None and rotation is None):
        raise ValueError("A ball or free joint pose is required")
    indices, values = [], []
    if position is not None:
        if joint.type != "free":
            raise ValueError("Only free joints have position coordinates")
        coordinates = np.asarray(position, np.float64).reshape(3)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Joint position must be finite")
        indices.extend(range(start, start + 3))
        values.extend(coordinates)
    if rotation is not None:
        angles = np.asarray(rotation, np.float64).reshape(3)
        if not np.all(np.isfinite(angles)):
            raise ValueError("Joint rotation must be finite")
        quaternion = np.asarray(
            math3d.mat3_to_quat(math3d.euler_xyz_to_mat3(np.radians(np.remainder(angles, 360)))),
            np.float64,
        )
        quaternion /= np.linalg.norm(quaternion)
        offset = start + (3 if joint.type == "free" else 0)
        if float(np.dot(quaternion, qpos[offset : offset + 4])) < 0:
            quaternion *= -1
        indices.extend(range(offset, offset + 4))
        values.extend(quaternion)
    return cmd.SetQposBatch(np.asarray(indices, np.intp), np.asarray(values, np.float64))


class NativePoseEditor:
    """Own joint value presentation; Navigation owns browsing and selection."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._generation = None
        self._nodes = {}
        self._rotations = {}
        self._failed_gestures = set()
        self._active_joint = None
        self._expanded = {}
        self._selection = {}

    def draw(
        self,
        ctx,
        x,
        y,
        w,
        *,
        joints,
        select_joint,
        initial_qpos=None,
        show_names=True,
        show_types=False,
        show_context=False,
        joint_color=None,
        collapsible_poses=False,
    ):
        u, s = self.workspace.ui, self.workspace.s
        if initial_qpos is None:
            initial_qpos = getattr(ctx.session.source, "initial_qpos", ())
        generation = (ctx.session.document_id, ctx.session.structure_generation)
        if self._generation != generation:
            self._generation = generation
            self._rotations.clear()
            self._nodes = {
                node.joint_index: node for node in ctx.session.nodes if node.type is NodeType.JOINT
            }
            self._active_joint = None
            self._expanded.clear()
            self._selection.clear()
        window = imgui.internal.get_current_window()
        if collapsible_poses:
            selected = ctx.session.selected_node
            token = selected.node_id if selected is not None else None
            if self._selection.get(window.id_) != token:
                self._selection[window.id_] = token
                if selected is not None:
                    for joint in joints:
                        if _selected_joint(selected, joint) and joint.type in ("ball", "free"):
                            self._expanded[window.id_, joint.joint_id] = True
        clip_top, clip_bottom = window.clip_rect.min.y, window.clip_rect.max.y
        if not imgui.is_any_item_active():
            self._active_joint = None
        qpos = ctx.session.frame.qpos
        style = u.component_style("joint")
        pads, margin = style.edges("padding", s), style.edges("margin", s)
        inner_width = u.component_box("joint", 0, 0, w, 0, observe=False).content.width
        gap = style.get("rowGap", style.get("gap", 8)) * s
        item_gap = style.get("gap", 8) * s
        vector_layout = None
        summary_layout = None
        scalar_heights = {}
        with u.component_scope("joint"):
            heading_height = (
                control_size(
                    u, "pose-joint", inner_width / s, 28, label="Joint", icon="joint-hinge"
                )[1]
                if show_names
                else 0
            )
            body_column = self._body_column(inner_width, heading_height, show_names=show_names)
        heading_gap = gap if show_names else 0
        for joint in joints:
            top = y
            foldable = collapsible_poses and show_names and joint.type in ("ball", "free")
            owner = (window.id_, joint.joint_id)
            if foldable and owner not in self._expanded:
                self._expanded[owner] = _selected_joint(selected, joint)
            expanded = (
                not foldable or self._expanded.get(owner, False) or self._active_joint == owner
            )
            size = _QPOS_SIZE.get(joint.type)
            available = (
                qpos is not None
                and size is not None
                and joint.qpos_adr >= 0
                and joint.qpos_adr + size <= len(qpos)
            )
            with u.component_scope("joint"):
                field_gap = gap
                if available and joint.type in ("ball", "free"):
                    if expanded:
                        if vector_layout is None:
                            vector_layout = u.vector_layout(body_column.width, format="%.3f")
                        row_height = vector_layout.height
                    else:
                        if summary_layout is None:
                            summary_layout = self._summary_layout(body_column.width)
                        row_height = summary_layout.height
                        field_gap = style.get("summaryGroupGap", 12) * s
                    field_height = row_height * (2 if joint.type == "free" else 1) + field_gap * (
                        1 if joint.type == "free" else 0
                    )
                else:
                    if available and joint.limited not in scalar_heights:
                        scalar_heights[joint.limited] = u.rail_extent(
                            body_column.width, bounded=joint.limited
                        )
                    field_height = scalar_heights[joint.limited] if available else 28 * s
            height = (
                heading_height
                + heading_gap
                + field_height
                + pads[1]
                + pads[3]
                + margin[1]
                + margin[3]
            )
            bottom = top + height + item_gap
            # Keep the full scroll extent without submitting offscreen text, icons,
            # quaternion conversions, or controls. An active drag keeps its owner alive.
            if (bottom < clip_top or top > clip_bottom) and self._active_joint != (
                window.id_,
                joint.joint_id,
            ):
                y = bottom
                continue
            node = self._nodes.get(joint.joint_id)
            box = u.component_box(
                "joint", x, top, w, height, key=f"joint-card-{joint.joint_id}", observe=False
            )
            bx, y, bw, _ = box.content.tuple()
            fields = Rect(
                bx + body_column.x,
                y + heading_height + heading_gap,
                min(body_column.width, max(0, bw - body_column.x)),
                field_height,
            )
            with u.component_scope("joint"):
                heading_layout = self._heading_layout(joint, bw, foldable)
            regions = {
                "gap": [x, box.outer.bottom, w, item_gap],
                "rowGap": [
                    bx,
                    y + heading_height,
                    bw,
                    heading_gap,
                ],
            }
            if available and not expanded:
                if summary_layout.above:
                    regions["summaryLabelGap"] = [
                        fields.x,
                        fields.y + summary_layout.line_height,
                        fields.width,
                        summary_layout.label_gap,
                    ]
                if joint.type == "free":
                    regions["summaryGroupGap"] = [
                        fields.x,
                        fields.y + summary_layout.height,
                        fields.width,
                        field_gap,
                    ]
            if foldable:
                regions["columnGap"] = [
                    bx + heading_layout.heading_width,
                    y,
                    heading_layout.gap,
                    heading_height,
                ]
            u.record_component_box(
                "joint",
                box,
                key=f"joint-card-{joint.joint_id}",
                fieldRegions=regions,
                bodyColumn=list(fields.tuple()),
                headingContentColumn=[fields.x, y, fields.width, heading_height]
                if show_names
                else None,
                alignmentAxis="leading-content" if show_names else "unheaded",
            )
            with u.component_scope("joint"):
                if show_names:
                    self._heading(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        bx,
                        y,
                        bw,
                        joint_color,
                        heading_height,
                        heading_layout,
                        show_type=show_types,
                        show_context=show_context,
                        expanded=expanded if foldable else None,
                    )
                    y += heading_height + heading_gap
                field_x, field_y, field_width, _ = fields.tuple()
                enabled = (
                    available
                    and ctx.session.paused
                    and not ctx.session.state_take_playing
                    and ctx.session.adapter.caps.write_qpos
                )
                if not available:
                    u.component_text("joint", fields.tuple(), "qpos not produced this frame", MUTED)
                elif joint.type in ("hinge", "slide"):
                    self._scalar(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        field_x,
                        field_y,
                        field_width,
                        enabled,
                        initial_qpos,
                    )
                elif not expanded:
                    self._summary(
                        ctx, joint, field_x, field_y, field_width, summary_layout, field_gap
                    )
                else:
                    self._pose(
                        ctx,
                        joint,
                        node,
                        select_joint,
                        field_x,
                        field_y,
                        field_width,
                        enabled,
                        vector_layout,
                        gap,
                    )
            y = bottom
            u.hits[f"pose-fields-{joint.joint_id}"] = (x, top, w, y - top)
        if joints:
            publish_focus_item_hint(ctx)
        else:
            u.text(x, y + 8 * s, "No matching joints", MUTED)
            y += 32 * s
        imgui.set_cursor_screen_pos((x, y))
        imgui.dummy((1, 1))
        return y

    def _body_column(self, width, heading_height, *, show_names):
        """Align body fields with the heading's icon column without its name indent."""
        column = Rect(0, 0, width, 0)
        if not show_names:
            return column
        u, scale = self.workspace.ui, self.workspace.s
        key, label = "pose-joint", "Joint"
        parts = u.design.button_parts(key, label) if u.design is not None else "button"
        style = u.component_style(parts)
        frame_width, frame_height = control_frame_size(u, key, width, heading_height, label=label)
        box = style.box(
            0,
            0,
            frame_width * scale,
            frame_height * scale,
            scale=scale,
            padding=(8 * scale, 0, 8 * scale, 0),
        )
        return column.inset(box.content.x, 0, box.padding[2] + box.margin[2], 0)

    def _heading_layout(self, joint, width, foldable):
        u = self.workspace.ui
        if not foldable:
            return _HeadingLayout(width, 0, 0)
        disclosure_width, _ = control_size(
            u, f"pose-joint-{joint.joint_id}-expand", 28, 28, icon="down"
        )
        disclosure_width = min(width, disclosure_width)
        style = u.component_style("joint")
        gap = min(
            max(0, width - disclosure_width),
            style.get("columnGap", style.get("gap", 8)) * u.s,
        )
        return _HeadingLayout(max(0, width - disclosure_width - gap), disclosure_width, gap)

    def _heading(
        self,
        ctx,
        joint,
        node,
        select_joint,
        x,
        y,
        w,
        color,
        height,
        layout,
        *,
        show_type=False,
        show_context=False,
        expanded=None,
    ):
        u = self.workspace.ui
        key, label = "pose-joint-" + str(joint.joint_id), joint.name or f"joint{joint.joint_id}"
        disclosure_width = layout.disclosure_width
        heading_width = layout.heading_width
        type_width = 0
        context = joint_heading_context(joint, ctx.tr) if show_context and not show_type else ""
        context_width, context_gap = 0, 0
        if show_type:
            kind = ctx.tr(_JOINT_LABELS.get(joint.type, joint.type))
            type_width = min(heading_width * 0.45, u.measure(kind, translate=False) + 4 * u.s)
        elif context:
            context_width, context_gap = self._context_width(
                key, label, context, heading_width, height
            )
        name_width = max(
            0,
            heading_width
            - type_width
            - (DESIGN.item_spacing_x * u.s if show_type else 0)
            - context_width
            - context_gap,
        )
        width, frame_height = control_frame_size(u, key, name_width, height, label=label)
        pressed = u.button(
            key,
            x,
            y,
            width,
            h=frame_height,
            width_limit=width * u.s,
            icon=joint_icon(joint.type),
            label=label,
            active=node is not None and ctx.session.selected_node is node,
            icon_color=color,
            translate=False,
            align="left",
            tooltip=self._heading_details(ctx, joint) if show_context else None,
        )
        focused = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
        if imgui.is_item_clicked() or focused or (pressed and not imgui.is_mouse_released(0)):
            select_joint(ctx, joint, node, focused=focused)
        # The heading button and its context details already own the hover hint.
        copyable_name_item(ctx, joint.name, name_width, tooltip=False)
        if show_type:
            u.component_text(
                "joint",
                (x + heading_width - type_width, y, type_width, height),
                kind,
                MUTED,
                align="end",
                translate=False,
            )
        context_rect = None
        if context_width:
            with part_scope("control_metadata", inherit=False):
                box = u.component_box(
                    "control_metadata",
                    x + heading_width - context_width,
                    y,
                    context_width,
                    height,
                    key=key + "-context",
                )
                u.component_text(
                    "control_metadata",
                    box.content,
                    context,
                    MUTED,
                    size=12,
                    align="end",
                    translate=False,
                )
                context_rect = box.frame.tuple()
                u.hits[key + "-context"] = context_rect
        if expanded is not None:
            width, frame_height = control_frame_size(u, key + "-expand", disclosure_width, height)
            if u.button(
                key + "-expand",
                x + w - disclosure_width,
                y,
                width,
                frame_height,
                icon="down" if expanded else "right",
                tooltip="Collapse joint pose" if expanded else "Expand joint pose",
                width_limit=disclosure_width,
            ):
                self._expanded[
                    imgui.internal.get_current_window().id_, joint.joint_id
                ] = not expanded
        if show_context and u.design is not None:
            u.design.record(
                key + "-heading",
                ("joint",),
                (x, y, w, height),
                "group",
                name=label,
                nameRect=list(u.hits[key]),
                context=context,
                contextVisible=bool(context_width),
                contextRect=list(context_rect) if context_rect else None,
                disclosureRect=list(u.hits[key + "-expand"]) if expanded is not None else None,
            )

    def _context_width(self, key, name, context, width, height):
        """Only spend spare heading space; a complete name always has priority."""
        u, scale = self.workspace.ui, self.workspace.s
        parts = u.design.button_parts(key, name) if u.design is not None else "button"
        button = u.component_style(parts)
        padding, margin = (
            button.edges("padding", scale, (8, 0, 8, 0)),
            button.edges("margin", scale),
        )
        name_width = u.measure(
            name + " ", button.get("fontSize", BODY_SIZE), translate=False, style=button
        )
        name_width += (
            button.get("iconSize", BODY_SIZE) + button.get("columnGap", button.get("gap", 6))
        ) * scale
        name_width += padding[0] + padding[2] + margin[0] + margin[2]
        joint_style = u.component_style("joint")
        gap = joint_style.get("columnGap", joint_style.get("gap", 8)) * scale
        with part_scope("control_metadata", inherit=False):
            style = u.component_style("control_metadata")
            padding, margin = style.edges("padding", scale), style.edges("margin", scale)
            font = style.get("fontSize", 12)
            # A trailing space keeps fractional-origin float error outside the ink budget.
            context_width = u.measure(context + " ", font, translate=False, style=style)
            context_width += padding[0] + padding[2] + margin[0] + margin[2]
            text_height = max(font, style.line_height(font * 1.4)) * scale
            text_height += padding[1] + padding[3] + margin[1] + margin[3]
        if name_width + gap + context_width > width or text_height > height:
            return 0, 0
        return context_width, gap

    def _heading_details(self, ctx, joint):
        lines = [
            joint.name or f"joint{joint.joint_id}",
            ctx.tr(_JOINT_LABELS.get(joint.type, joint.type)),
        ]
        context = joint_heading_context(joint, ctx.tr)
        if context:
            lines[-1] += " · " + context
        if joint.type == "ball" and joint.limited:
            upper = joint.range[1]
            if isfinite(upper) and upper > 0:
                angle = numeric_text(float(np.degrees(upper)), "%.6g")
                lines.append(ctx.tr("limit angle") + f": ≤{angle}°")
        qpos, size = ctx.session.frame.qpos, _QPOS_SIZE.get(joint.type)
        if qpos is not None and size is not None and 0 <= joint.qpos_adr <= len(qpos) - size:
            start = joint.qpos_adr
            lines.append(f"qpos[{start}]" if size == 1 else f"qpos[{start}:{start + size}]")
        return "\n".join(lines)

    def _summary_layout(self, width):
        u, s = self.workspace.ui, self.workspace.s
        style = u.component_style("joint")
        font_size = style.get("fontSize", 12)
        line = max(font_size, style.line_height(font_size * 1.4)) * s
        gap = style.get("columnGap", style.get("gap", 8)) * s
        badge_size = min(line, font_size * 1.25 * s)
        axis_width = badge_size + font_size * 0.4 * s
        label = max(u.measure(text, font_size, style=style) for text in ("Position", "Rotation"))
        label += u.measure(" (m)", font_size, translate=False, style=style) + gap
        minimum = axis_width + u.measure(
            "-0.000", font_size, mono=True, translate=False, style=style
        )
        above = width < label + 3 * minimum + 2 * gap
        available = width if above else width - label
        columns = 3 if available >= 3 * minimum + 2 * gap else 1
        return _SummaryLayout(
            label,
            (available - (columns - 1) * gap) / columns,
            line,
            gap,
            style.get("summaryLabelGap", 4) * s if above else 0,
            columns,
            above,
            axis_width,
            badge_size,
        )

    def _summary(self, ctx, joint, x, y, width, layout, gap):
        u = self.workspace.ui
        rotation = self._rotation(ctx, joint)
        if rotation is None:
            u.component_text(
                "joint", (x, y, width, layout.height), "Invalid joint quaternion", MUTED
            )
            return
        start = joint.qpos_adr
        rows = (
            [("Position", ctx.session.frame.qpos[start : start + 3], "m")]
            if joint.type == "free"
            else []
        )
        rows.append(("Rotation", rotation, "°"))
        style = u.component_style("joint")
        font_size = style.get("fontSize", 12)
        for name, values, unit in rows:
            label_width = width if layout.above else layout.label_width - layout.gap
            label_rect = Rect(x, y, label_width, layout.line_height)
            caption = f"{ctx.tr(name)} ({unit})"
            u.component_text(
                "joint",
                label_rect.tuple(),
                caption,
                MUTED,
                size=font_size,
                translate=False,
            )
            if u.design is not None:
                u.design.record(
                    f"pose-summary-label-{joint.joint_id}-{name.lower()}",
                    ("joint",),
                    label_rect.tuple(),
                    "label",
                    labelRect=list(label_rect.tuple()),
                    text=caption,
                    alignmentAxis="leading-content",
                )
            cells_x = x if layout.above else x + layout.label_width
            cells_y = y + (layout.line_height + layout.label_gap if layout.above else 0)
            for index, (axis, value) in enumerate(zip("XYZ", values, strict=True)):
                cx = cells_x + (index % layout.columns) * (layout.cell_width + layout.gap)
                cy = cells_y + (index // layout.columns) * layout.line_height
                value_width = max(0, layout.cell_width - layout.axis_width)
                shown = fit_numeric_text(
                    float(value),
                    "%.3f",
                    value_width,
                    lambda text: (
                        u.measure(text, font_size, mono=True, translate=False, style=style),
                        0,
                    ),
                )
                badge_y = cy + (layout.line_height - layout.badge_size) / 2
                u.rect(
                    cx,
                    badge_y,
                    layout.badge_size,
                    layout.badge_size,
                    tuple(
                        round(channel * 255)
                        for channel in mix_color(
                            tuple(channel / 255 for channel in u.palette("field")),
                            THEME.axis_color(index),
                            0.56,
                        )
                    ),
                    layout.badge_size / (2 * u.s),
                    color_resolved=True,
                )
                u.component_text(
                    "joint",
                    (cx, badge_y, layout.badge_size, layout.badge_size),
                    axis,
                    TEXT,
                    size=font_size,
                    mono=True,
                    align="center",
                    translate=False,
                )
                u.component_text(
                    "joint",
                    (cx + layout.axis_width, cy, value_width, layout.line_height),
                    shown,
                    MUTED,
                    size=font_size,
                    mono=True,
                    align="end",
                    translate=False,
                )
                u.hits[f"pose-summary-{joint.joint_id}-{name.lower()}-{axis}"] = (
                    cx,
                    cy,
                    layout.cell_width,
                    layout.line_height,
                )
            y += layout.height + gap

    def _rotation(self, ctx, joint):
        qpos = ctx.session.frame.qpos
        offset = joint.qpos_adr + (3 if joint.type == "free" else 0)
        quaternion = np.asarray(qpos[offset : offset + 4], np.float64)
        if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) <= 1e-12:
            return None
        matrix = math3d.quat_to_mat3(quaternion)
        previous = self._rotations.get(joint.joint_id)
        rotation = (
            previous[1].copy()
            if previous is not None and np.allclose(matrix, previous[0], atol=2e-5, rtol=0)
            else _nearest_euler_degrees(matrix, previous[1] if previous is not None else None)
        )
        self._rotations[joint.joint_id] = (matrix.copy(), rotation.copy())
        return rotation

    def _scalar(self, ctx, joint, node, select_joint, x, y, w, enabled, initial_qpos):
        u = self.workspace.ui
        angular = joint.type == "hinge"
        lo, hi = joint.range if joint.limited else (0, 0)
        key = "##pose-joint-" + str(joint.joint_id)
        imgui.begin_disabled(not enabled)
        imgui.begin_group()
        edit = u.rail(
            key,
            x,
            y,
            w,
            float(ctx.session.frame.qpos[joint.qpos_adr]),
            (lo, hi) if joint.limited else None,
            initial=float(initial_qpos[joint.qpos_adr])
            if len(initial_qpos) > joint.qpos_adr
            else float(ctx.session.frame.qpos[joint.qpos_adr]),
            fmt="%.3f",
            show_reset=False,
            unit="rad" if angular else "m",
            angular_degrees=True,
        )
        imgui.end_group()
        active, deactivated = imgui.is_item_active(), imgui.is_item_deactivated()
        state = u.numeric_edits[key] = NumericEditState(
            activated=imgui.is_item_activated(),
            active=active,
            deactivated=deactivated,
            committed=imgui.is_item_deactivated_after_edit(),
            canceled=(active or deactivated) and imgui.is_key_pressed(imgui.Key.escape, False),
        )
        imgui.end_disabled()
        self._edit(
            ctx,
            joint,
            node,
            select_joint,
            key,
            state,
            edit.changed,
            enabled,
            lambda: cmd.SetQpos(joint.qpos_adr, edit.value),
        )
        return y + 36 * u.s

    def _pose(self, ctx, joint, node, select_joint, x, y, w, enabled, layout, gap):
        u = self.workspace.ui
        qpos = ctx.session.frame.qpos
        start = joint.qpos_adr
        offset = start + (3 if joint.type == "free" else 0)
        rotation = self._rotation(ctx, joint)
        if rotation is None:
            u.text(
                x, y + DESIGN.control_padding_y(BODY_SIZE) * u.s, "Invalid joint quaternion", MUTED
            )
            return y + 36 * u.s
        fields = (
            (("position", qpos[start : start + 3], "m", 0.01, "%.3f"),)
            if joint.type == "free"
            else ()
        )
        fields += (("rotation", rotation, "°", 0.5, "%.3f"),)
        initial = ctx.session.source.initial_qpos
        for name, values, unit, speed, fmt in fields:
            key = f"##pose-joint-{joint.joint_id}-{name}"
            if name == "rotation":
                reset = (
                    _nearest_euler_degrees(math3d.quat_to_mat3(initial[offset : offset + 4]))
                    if initial is not None and offset + 4 <= len(initial)
                    else (0, 0, 0)
                )
            else:
                reset = (
                    initial[start : start + 3]
                    if initial is not None and start + 3 <= len(initial)
                    else (0, 0, 0)
                )
            imgui.begin_disabled(not enabled)
            changed, edited = u.vector(
                key,
                x,
                y,
                w,
                name.title(),
                values,
                unit,
                speed=speed,
                format=fmt,
                reset_values=reset,
                layout=layout,
            )
            imgui.end_disabled()
            state = u.numeric_edits[key]
            applied = self._edit(
                ctx,
                joint,
                node,
                select_joint,
                key,
                state,
                changed,
                enabled,
                lambda name=name, edited=edited: joint_pose_command(joint, qpos, **{name: edited}),
            )
            if applied and name == "rotation":
                self._rotations[joint.joint_id] = (
                    math3d.euler_xyz_to_mat3(np.radians(edited)),
                    np.asarray(edited, np.float64),
                )
            y += layout.height + gap
        return y

    def _edit(self, ctx, joint, node, select_joint, key, state, changed, enabled, command):
        if not enabled:
            return False
        edit_key = f"{key}##window-{imgui.internal.get_current_window().id_}"
        owner = (imgui.internal.get_current_window().id_, joint.joint_id)
        if state.active or state.activated:
            self._active_joint = owner
        elif state.deactivated and self._active_joint == owner:
            self._active_joint = None
        if state.activated:
            self._failed_gestures.discard(edit_key)
            select_joint(ctx, joint, node)
        # The same joint may be visible in both Pose and Inspector. Only the
        # window that owns the gesture may finish its transaction.
        self.workspace.ui.numeric_edits[edit_key] = state
        if ctx.session.adapter.caps.edit_history and edit_key not in self._failed_gestures:
            try:
                self.workspace.begin_numeric_edit(edit_key, label="Pose edit")
            except RuntimeError as error:
                ctx.report(str(error), level="error")
                self._failed_gestures.add(edit_key)
            if state.activated and self.workspace._numeric_edit != edit_key:
                self._failed_gestures.add(edit_key)
        if edit_key in self._failed_gestures:
            if state.canceled or (state.deactivated and not state.active):
                self._failed_gestures.discard(edit_key)
            return False
        applied = False
        if changed and not state.canceled:
            try:
                update = command()
            except (TypeError, ValueError) as error:
                ctx.report(str(error), level="error")
            else:
                applied = ctx.submit(update).ok
        self.workspace.finish_numeric_edit(edit_key, state)
        return applied
