"""MuJoCo joint controls and range display."""

from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, JointInfo, NodeType
from . import (
    Panel,
    PanelContext,
    copyable_name_item,
    publish_focus_item_hint,
    searchable_ordered_list_header,
    value_slider,
)

_SCALAR_KINDS = ("hinge", "slide")
_BROWSE_THRESHOLD = 256
_PAGE_SIZE = 128


class JointsPanel(Panel):
    name = "Joints"
    default_open = True
    shortcut = "F5"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._initial_qpos = np.zeros(0, np.float64)
        self._snapshot_generation = -1
        self._joint_page = 0
        self._search = ""
        self._sort_by_name = False
        self._browse_cache_key: tuple[int, str, bool] | None = None
        self._browse_cache: tuple[JointInfo, ...] = ()
        self._joint_nodes_generation = -1
        self._joint_nodes: dict[int, object] = {}

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, qpos=True)

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        frame = s.frame
        self._snapshot(ctx)

        caps = s.adapter.caps
        if not caps.write_qpos:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{caps.name} {ctx.tr('cannot write joint positions')}",
            )
        if frame.qpos is None:
            imgui.text_disabled(ctx.tr("qpos not produced this frame"))
            return
        if not s.joints:
            imgui.text_disabled(ctx.tr("no joints"))
            return

        changed, self._search, sort_changed, self._sort_by_name = searchable_ordered_list_header(
            "##joint_search",
            self._search,
            self._sort_by_name,
            hint=ctx.tr("Search joints"),
            search_tooltip=ctx.tr("Search joints"),
            clear_tooltip=ctx.tr("Clear search"),
            state_order="qpos / qvel",
            translate=ctx.tr,
        )
        if changed or sort_changed:
            self._joint_page = 0

        cache_key = (s.structure_generation, self._search, self._sort_by_name)
        if cache_key != self._browse_cache_key:
            self._browse_cache = sort_joints(
                filter_joints(s.joints, self._search),
                by_name=self._sort_by_name,
            )
            self._browse_cache_key = cache_key
        ordered = self._browse_cache
        if not ordered:
            imgui.text_disabled(ctx.tr("No matching joints"))
            return

        if len(ordered) > _BROWSE_THRESHOLD:
            imgui.text_disabled(f"{len(ordered)} {ctx.tr('joints total')}")
            self._joint_page, start, stop = _page_controls(
                ctx, "joints", len(ordered), self._joint_page
            )
            ordered = ordered[start:stop]

        if self._joint_nodes_generation != s.structure_generation:
            self._joint_nodes = {
                node.joint_index: node for node in s.nodes if node.type is NodeType.JOINT
            }
            self._joint_nodes_generation = s.structure_generation
        if self._joint_nodes:
            publish_focus_item_hint(ctx)
        self._joint_table(ctx, ordered, self._joint_nodes)

    def _joint_table(self, ctx: PanelContext, joints, joint_nodes) -> None:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("joint_values", 2, flags):
            return
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.36)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch, 0.64)
        clipper = imgui.ListClipper()
        clipper.begin(len(joints))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                joint = joints[index]
                self._joint_row(ctx, joint, joint_nodes.get(joint.joint_id))
        clipper.end()
        imgui.end_table()

    def _joint_row(self, ctx: PanelContext, j: JointInfo, joint_node) -> None:
        qpos = ctx.session.frame.qpos
        if qpos is None or j.qpos_adr >= len(qpos):
            return
        name = j.name or f"joint{j.joint_id}"
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        selected_node = ctx.session.selected_node
        selected = bool(joint_node is not None and selected_node is joint_node)
        if joint_node is None:
            label_width = max(1.0, imgui.get_content_region_avail().x)
            imgui.text_disabled(name)
        else:
            label_width = max(1.0, imgui.get_content_region_avail().x)
            imgui.begin_disabled(not ctx.session.paused)
            clicked, _ = imgui.selectable(
                f"{name}##joint-select-{j.joint_id}",
                selected,
                imgui.SelectableFlags_.none.value,
                imgui.ImVec2(label_width, 0.0),
            )
            imgui.end_disabled()
            double_clicked = imgui.is_item_hovered(
                imgui.HoveredFlags_.allow_when_disabled.value
            ) and imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
            if clicked:
                ctx.submit(cmd.SelectNode(joint_node.node_id))
            if double_clicked and ctx.focus_joint is not None:
                if not clicked:
                    ctx.submit(cmd.SelectNode(joint_node.node_id))
                ctx.focus_joint(j.joint_id)
        copyable_name_item(ctx, name, label_width)
        imgui.table_next_column()
        if j.type not in _SCALAR_KINDS:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(f"{j.type} · {j.dof} {ctx.tr('dof')}")
            return

        value = float(qpos[j.qpos_adr])
        lo, hi = _joint_range(j)
        imgui.set_next_item_width(-1.0)
        imgui.begin_disabled(not ctx.session.adapter.caps.write_qpos or not ctx.session.paused)
        edit = value_slider(
            f"##joint-qpos-{j.qpos_adr}",
            value,
            lo,
            hi,
            initial=_initial_value(self._initial_qpos, j.qpos_adr, value),
            fmt="%.4f",
            more_hint="",
        )
        value_clicked = imgui.is_item_clicked(imgui.MouseButton_.left)
        imgui.end_disabled()
        if value_clicked and joint_node is not None and ctx.session.paused:
            ctx.submit(cmd.SelectNode(joint_node.node_id))
        if edit.changed:
            ctx.submit(cmd.SetQpos(j.qpos_adr, edit.value))

    def _snapshot(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        frame = ctx.session.frame
        if gen != self._snapshot_generation:
            self._snapshot_generation = gen
            self._initial_qpos = np.zeros(0, np.float64)
            self._joint_page = 0
            self._search = ""
            self._browse_cache_key = None
            self._browse_cache = ()
            self._joint_nodes_generation = -1
            self._joint_nodes = {}
        if frame.qpos is not None and len(self._initial_qpos) != len(frame.qpos):
            self._initial_qpos = np.asarray(frame.qpos, np.float64).copy()


def _page_controls(ctx: PanelContext, label: str, count: int, page: int) -> tuple[int, int, int]:
    page, pages, start, stop = page_span(count, page)
    if pages <= 1:
        return page, start, stop
    if imgui.button(f"{ctx.tr('Previous')}##{label}"):
        page -= 1
    imgui.same_line()
    if imgui.button(f"{ctx.tr('Next')}##{label}"):
        page += 1
    page, pages, start, stop = page_span(count, page)
    imgui.same_line()
    imgui.text_disabled(f"{ctx.tr('page')} {page + 1}/{pages} · {start + 1}-{stop}")
    return page, start, stop


def filter_joints(joints, query: str) -> tuple[JointInfo, ...]:
    """Return joints whose stable display name contains ``query``."""

    needle = str(query).strip().casefold()
    if not needle:
        return tuple(joints)
    return tuple(
        joint for joint in joints if needle in (joint.name or f"joint{joint.joint_id}").casefold()
    )


def sort_joints(joints, *, by_name: bool) -> tuple[JointInfo, ...]:
    """Order joints by simulation state address or stable display name."""

    if by_name:
        return tuple(
            sorted(
                joints,
                key=lambda joint: (
                    (joint.name or f"joint{joint.joint_id}").casefold(),
                    joint.qpos_adr,
                    joint.qvel_adr,
                    joint.joint_id,
                ),
            )
        )
    return tuple(sorted(joints, key=lambda joint: (joint.qpos_adr, joint.qvel_adr, joint.joint_id)))


def page_span(count: int, page: int, page_size: int = _PAGE_SIZE) -> tuple[int, int, int, int]:
    """Return a bounded page and half-open item span for a large editor list."""
    size = max(1, int(page_size))
    total = max(0, int(count))
    pages = max(1, (total + size - 1) // size)
    current = min(max(0, int(page)), pages - 1)
    start = current * size
    return current, pages, start, min(total, start + size)


def _initial_value(values: np.ndarray, index: int, fallback: float) -> float:
    return float(values[index]) if 0 <= index < len(values) else float(fallback)


def _joint_range(j: JointInfo) -> tuple[float, float]:
    if j.limited and j.range[1] > j.range[0]:
        return float(j.range[0]), float(j.range[1])
    return (-math.pi, math.pi) if j.type == "hinge" else (-1.0, 1.0)
