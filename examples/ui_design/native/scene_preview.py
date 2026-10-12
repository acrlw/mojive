"""Production viewer runtime hosted by the native design shell."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from mojive import CameraView, Scene, math3d
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, NodeType
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.config import ViewerConfig
from mojive.render.backend import RenderFlag
from mojive.render.geometry import GeometryView, set_geometry_view
from mojive.scene.assets import resolve
from mojive.scene.queries import node_geometry_indices, node_world_pose
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.input_bindings import InputAction
from mojive.ui.panels import PanelManager
from mojive.ui.viewcube import ViewCube


def rgba(value):
    return (*[int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)], 1.0)


PANEL_NAMES = {
    "hierarchy": "Scene",
    "joints": "Pose",
    "control": "Control",
    "assets": "Assets",
    "camera": "Cameras",
    "sensors": "Sensors",
    "layers": "Layers",
    "output": "Output",
    "stats": "Statistics",
    "inspector": "Inspector",
    "keyframes": "Timeline",
    "settings": "Settings",
}


class ReferencePanels(PanelManager):
    """Connect shared panel needs and shortcuts to the reference's own controls."""

    def __init__(self):
        super().__init__(builtin_ids=tuple(PANEL_NAMES))
        self.study = None
        for panel in self.panels:
            panel.name = PANEL_NAMES[panel.id]
            panel.open = False

    def set_open(self, panel_id, open):
        changed = super().set_open(panel_id, open)
        panel = self.get(panel_id)
        if changed and self.study is not None:
            if open:
                self.study.activate(panel.name)
            else:
                self.study.open[panel.name] = False
        return changed

    def draw(self, ctx):
        ctx.panels = self
        ctx.status_hints_by_panel.clear()
        for panel in self.panels:
            panel.open = self.study.open[panel.name]
            ctx.status_hints = ()
            if panel.open:
                ctx.status_hints_by_panel[panel.name] = ()
            if self.study.panel(panel.name, ctx):
                ctx.status_hints_by_panel[panel.name] = tuple(ctx.status_hints)
            if not self.study.open[panel.name]:
                ctx.status_hints_by_panel.pop(panel.name, None)
        self.study.menus.finish_color_edit()
        self.study.finish_numeric_frame(ctx)


class ReferenceViewCube(ViewCube):
    """Scale the orientation widget and its hit regions together in compact viewports."""

    def __init__(self, selection_padding):
        super().__init__(selection_padding)
        self.reference_scale = 1.0
        self.preview_active = False

    def scale_for(self, rect, style_scale):
        from mojive.ui import viewcube

        if self.preview_active and rect[2] < 180 * style_scale and rect[3] < 300 * style_scale:
            return 0.0
        base = style_scale * (
            0.7 if rect[2] < 260 * style_scale or rect[3] < 233 * style_scale else 1
        )
        # The right/top anchor includes the margin, spoke radius and full backdrop.
        extent = (
            viewcube.MARGIN_PT + viewcube.RADIUS_PT + viewcube.BALL_PT + viewcube.BACKDROP_RADIUS_PT
        )
        return min(
            base,
            # Reserve the compact controls' 10 pt margin, 36 pt shell and 8 pt gap.
            max(0, rect[2] - 54 * style_scale) / extent,
            max(0, rect[3] - 8 * style_scale) / extent,
        )

    def update(self, cam, rect, cursor, style_scale, *, enabled=True):
        self.reference_scale = self.scale_for(rect, style_scale)
        if self.reference_scale <= 0:
            self._balls = []
            self._hover = None
            self._origin_hovered = False
            return None
        return super().update(cam, rect, cursor, self.reference_scale, enabled=enabled)

    def draw(self, overlay, style_scale=1.0, **kwargs):
        super().draw(overlay, self.reference_scale, **kwargs)


class Preview(ViewerApp):
    """Override UI composition while retaining ViewerApp's complete frame pipeline."""

    def __init__(self, window, backend_name):
        if backend_name == "bgfx":
            from mojive.render.native.backend import NativeBackend

            backend = NativeBackend(800, 700, 4)
        else:
            from mojive.render.opengl.backend import OpenGLBackend

            window.make_current()
            backend = OpenGLBackend(None, 800, 700, 4)
        scene = Scene()
        scene.add_camera(
            "inspection",
            CameraView(
                eye=np.array([2.5, -1.7, 1.7]),
                target=np.array([0.6, 0, 0.7]),
                fov_y=np.radians(42),
            ),
        )
        session = Session(WorkspaceAdapter(MuJoCoAdapter(resolve("joint_types")), scene))
        session.submit(cmd.Pause())
        super().__init__(
            session,
            backend,
            window,
            config=ViewerConfig(debug_server=False, live_model_updates=True),
        )
        set_geometry_view(backend, GeometryView.VISUAL)
        self.panels = ReferencePanels()
        self.set_input_binding(InputAction.GIZMO_SPACE, "b", persist=False)
        self.study = None
        self.view_cube = ReferenceViewCube(self.view_cube.selection_padding)
        self.readback = False
        self.pixels = None
        window._layout_done = True
        self.gizmo.set_style("2d")
        self.gizmo.set_mode("rotate")
        self.selected = "hinge_body"
        self.camera.adopt(
            CameraView(
                eye=np.array([4.3, -5.9, 4.2]),
                target=np.array([0, 0, 0.55]),
                fov_y=np.radians(42),
                near=0.05,
                far=100,
            )
        )
        self._initial_camera_set = True
        self.document = json.loads((Path(__file__).parent / "document.json").read_text())
        self.entities = []
        self._entity_cache = {}
        self.refresh_document()

    @property
    def selected(self):
        node = self.session.selected_node
        return node.name if node is not None else None

    @selected.setter
    def selected(self, name):
        if name is None:
            self.session.submit(cmd.Select(0))
            return
        matches = [n for n in self.session.nodes if n.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one scene node named {name!r}, found {len(matches)}")
        self.session.submit(cmd.SelectNode(matches[0].node_id))

    @property
    def entity(self):
        node = self.session.selected_node
        if node is None:
            return None
        entry = next((e for e in self.entities if e["node_id"] == node.node_id), None)
        if entry is not None:
            entry["hidden"] = not node.visible
        return entry

    def refresh_document(self):
        """Project Session state into the reference controls; Session owns all edits."""
        from .navigation import node_accent, node_icon

        new_document = getattr(self, "_reference_document_id", None) != self.session.document_id
        if new_document:
            self._entity_cache.clear()
            if getattr(self, "study", None) is not None:
                self.study.pinned = None
        kinds = {kind: kind.value for kind in NodeType}
        source = self.session.source
        entities = []
        for node in self.session.nodes:
            if node.type not in kinds:
                continue
            entry = self._entity_cache.setdefault(node.node_id, {})
            position, rotation = node_world_pose(self.session, node)
            indices = node_geometry_indices(self.session, node)
            rgba = source.geom_rgba[indices[0]] if len(indices) else (0.6, 0.7, 0.8, 1)
            entry.update(
                id=str(node.node_id),
                node_id=node.node_id,
                object_id=node.object_id,
                parent=node.parent,
                name=node.name,
                type=kinds[node.type],
                identity=(node.model_id, node.type.value, node.source_name or node.name)
                if node.source_editable
                else ("object", node.object_id)
                if node.object_id
                else ("node", node.node_id, node.type.value, node.model_id),
                icon=node_icon(node, self.session),
                icon_color=node_accent(self.theme, node.type.value),
                position=position.tolist(),
                rotation=np.degrees(math3d.mat3_to_euler_xyz(rotation)).tolist(),
                scale=list(self.session.scale_factors(node.node_id)),
                color="#" + "".join(f"{round(np.clip(v, 0, 1) * 255):02x}" for v in rgba[:3]),
                rgba=[float(v) for v in rgba],
                hidden=not node.visible,
                roughness=0.5,
                posable=node.posable,
                scalable=self.session.scale_target(node.node_id) is not None,
                colorable=node.type is NodeType.GEOM
                or any(self.session.node(child).type is NodeType.GEOM for child in node.children),
            )
            entities.append(entry)
        self.entities[:] = entities
        self.document["entities"] = self.entities
        self._projected_entities = copy.deepcopy(self.entities)
        if new_document:
            self.initial_entities = copy.deepcopy(self.entities)
            self._reference_document_id = self.session.document_id
        for key, joint_name in (("hinge", "hinge_limited"), ("slide", "slide"), ("ball", "ball")):
            joint = next((j for j in self.session.joints if j.name == joint_name), None)
            if joint is not None and self.session.frame.qpos is not None:
                value = self.session.frame.qpos[joint.qpos_adr]
                if key == "ball":
                    value = math3d.mat3_to_euler_xyz(
                        math3d.quat_to_mat3(
                            self.session.frame.qpos[joint.qpos_adr : joint.qpos_adr + 4]
                        )
                    )[1]
                self.document["pose"][key] = float(value if key == "slide" else np.degrees(value))

    def apply_document_edits(self):
        """Route reference field changes through the same command boundary as gizmos."""
        previous = {e["node_id"]: e for e in self._projected_entities}
        for entry in self.entities:
            old = previous.get(entry["node_id"])
            if old is None:
                continue
            nid = entry["node_id"]
            edits = []
            if (entry["position"], entry["rotation"]) != (old["position"], old["rotation"]):
                edits.append(
                    cmd.SetPose(
                        nid,
                        np.array(entry["position"]),
                        math3d.euler_xyz_to_mat3(np.radians(entry["rotation"])),
                    )
                )
            if entry["scale"] != old["scale"]:
                edits.append(cmd.SetScale(nid, np.array(entry["scale"])))
            if entry["hidden"] != old["hidden"]:
                edits.append(cmd.SetVisible(nid, not entry["hidden"]))
            if entry["color"] != old["color"] or entry["rgba"] != old["rgba"]:
                node = self.session.node(nid)
                geometry = (
                    [node]
                    if node.type is NodeType.GEOM
                    else [
                        self.session.node(c)
                        for c in node.children
                        if self.session.node(c).type is NodeType.GEOM
                    ]
                )
                if entry["rgba"] == old["rgba"]:
                    entry["rgba"][:3] = [
                        int(entry["color"][i : i + 2], 16) / 255 for i in (1, 3, 5)
                    ]
                rgba = np.array(entry["rgba"])
                edits.extend(cmd.SetGeometryColor(g.node_id, rgba) for g in geometry)
            if edits:
                # Single fields retain the UI draft interceptor; color gestures
                # retain their existing transaction rather than nesting a batch.
                active = self.session.editing or self.model_edits._checkpoint is not None
                group = (
                    nullcontext()
                    if len(edits) == 1 or active
                    else self.session.edit("Edit " + entry["name"])
                )
                try:
                    with group:
                        for command in edits:
                            result = self.session.submit(command)
                            if not result.ok:
                                raise RuntimeError(result.message)
                except RuntimeError as error:
                    self.session.report_message(str(error), level="warning")
        self._projected_entities = copy.deepcopy(self.entities)

    def set_pose_value(self, name, value):
        joint_name = {"hinge": "hinge_limited", "slide": "slide", "ball": "ball"}[name]
        joint = next((j for j in self.session.joints if j.name == joint_name), None)
        if joint is None:
            return
        if name == "ball":
            indices = np.arange(joint.qpos_adr, joint.qpos_adr + 4)
            quat = math3d.mat3_to_quat(math3d.euler_xyz_to_mat3((0, np.radians(value), 0)))
            self.session.submit(cmd.SetQposBatch(indices, quat))
        else:
            self.session.submit(
                cmd.SetQpos(joint.qpos_adr, value if name == "slide" else np.radians(value))
            )

    def frame_needs(self):
        return super().frame_needs().merge(FrameNeeds(qpos=True, actuator=True))

    @property
    def tool(self):
        if not self.gizmo.enabled:
            return "select"
        return {"translate": "move", "rotate": "rotate", "dimensions": "scale"}[self.gizmo.mode]

    @tool.setter
    def tool(self, value):
        if value == "select":
            self.gizmo.cancel()
            self.gizmo.enabled = False
        else:
            self.gizmo.set_mode(
                {"move": "translate", "rotate": "rotate", "scale": "dimensions"}[value]
            )

    @property
    def gizmos(self):
        return self.viewport_layers.gizmos

    @gizmos.setter
    def gizmos(self, value):
        self.set_viewport_layers(replace(self.viewport_layers, gizmos=bool(value)), persist=False)

    @property
    def shadows(self):
        return self.backend.get_flag(RenderFlag.SHADOW)

    @shadows.setter
    def shadows(self, value):
        self.backend.set_flag(RenderFlag.SHADOW, bool(value))

    def bind(self, study):
        self.study = self.panels.study = study
        self._numeric_cancel_frame = -1
        from .output_panel import ReferenceOutput
        from .widgets import DIM, INPUT, LINE, PANEL, SAGE, TEXT

        def normalized(value):
            return tuple(channel / 255 for channel in value)

        self.theme = replace(
            self.theme,
            text=normalized(TEXT),
            text_disabled=normalized(DIM),
            bg_window=normalized(PANEL),
            bg_child=normalized(PANEL),
            bg_frame=normalized(INPUT),
            border=normalized(LINE),
            primary=normalized(SAGE),
        )
        settings = self.panels.get("settings")
        self.panels.panels[self.panels.panels.index(settings)] = study.settings
        output = self.panels.get("output")
        self.panels.panels[self.panels.panels.index(output)] = ReferenceOutput(study)
        for panel in self.panels.panels:
            panel.open = study.open[panel.name]

    def _panel_context(self):
        source, _ratio = self.study.ui.font_sources[0]
        fonts = replace(
            self.window.font_report,
            mono=source.label,
            mono_path=source.path,
            mono_index=source.index,
        )
        return replace(
            super()._panel_context(), translate=self.study.ui.translate, font_report=fonts
        )

    def _draw_main_menu(self):
        self.refresh_document()
        self.study.begin_frame()

    def _poll_application_shortcuts(self):
        # A numeric gesture owns Escape before application selection shortcuts.
        inspector = self.panels.get("inspector")
        numeric_edit = self.study._numeric_edit is not None or inspector._edit_transaction
        if numeric_edit and imgui.is_key_pressed(imgui.Key.escape, False):
            result = self.session.submit(cmd.CancelEditTransaction())
            if result.ok:
                self.study._numeric_edit = None
                inspector._edit_transaction = False
                self._numeric_cancel_frame = imgui.get_frame_count()
                imgui.internal.clear_active_id()
                self.refresh_document()
            return
        super()._poll_application_shortcuts()
        io = imgui.get_io()
        if self._scene_input_blocked() or imgui.is_any_item_active() or io.key_ctrl or io.key_super:
            return
        if imgui.is_key_pressed(imgui.Key.v, False):
            self.tool = "select"
        if imgui.is_key_pressed(imgui.Key.t, False):
            self.study.toggle_timeline()

    def _draw_application_status_bar(self, *, loading=False):
        super()._draw_application_status_bar(loading=loading)

    def _draw_collapsed_output(self):
        pass

    def _draw_viewport_status(self, overlay):
        if self.model_edits.active and self._viewport_rect[3] < 300 * self.study.s:
            self._status_path_bounds = self._status_notice_bounds = None
            return
        super()._draw_viewport_status(overlay)

    def _draw_model_drop_overlay(self, overlay):
        if (
            self.model_edits.active
            and self._has_scene_content()
            and not self.window.file_drag_active
        ):
            return
        super()._draw_model_drop_overlay(overlay)

    def _begin_viewport_panel(self):
        study = self.study
        if study.needs_layout:
            study.build()
        if study.pending:
            study.command(study.pending)
            study.pending = None
        # DockBuilder changes must finish before the affected window's Begin/End scope.
        study.apply_timeline_layout()
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            0 if study.show_viewport_tab else imgui.DockNodeFlags_.auto_hide_tab_bar
        )
        imgui.set_next_window_class(wc)
        self.viewport_surface.begin("Viewport")
        if not study.bottom_open:
            width, height = self.viewport_surface.size
            self.viewport_surface.size = (width, max(20 * study.s, height - 38 * study.s))

    def _scene_input_blocked(self):
        if self._numeric_cancel_frame == imgui.get_frame_count():
            return True
        if super()._scene_input_blocked():
            return True
        # A captured scene gesture retains ownership while crossing reference controls.
        if self.router.owns_scene_pointer or self.gizmo.using:
            return False
        cursor = imgui.get_io().mouse_pos
        return any(
            x <= cursor.x <= x + w and y <= cursor.y <= y + h
            for x, y, w, h in self.study.input_overlays
        )

    def _input_state(self):
        # A compact preview and the view cube must share drawing and hit geometry.
        self.view_cube.preview_active = bool(self.camera_preview.selected_camera(self.session)[1])
        return super()._input_state()

    def _draw_viewport_contents(self, preview_name="", *, session_busy=False):
        if session_busy:
            # The worker owns Session during a rebuild. Present its last image
            # without projecting entities or reading interaction targets.
            self._viewport_rect = self.viewport_surface.draw_image(
                self._viewport_image, self.window.viewport_texture_ref
            )
            imgui.end()
            return
        self.refresh_document()
        self.study.viewport(preview_name)
        if self.viewport_layers.gizmos:
            self._draw_joint_limit_controls()
            self._draw_joint_gizmo_picker()

    def _draw_reference_camera_preview(self, preview_name):
        from .camera_preview import ReferenceCameraPreview

        if not hasattr(self, "_reference_camera_preview"):
            self._reference_camera_preview = ReferenceCameraPreview(self.study)
        self._reference_camera_preview.draw(preview_name)

    def reset_layout(self, *, persist=True):
        self.study.pending = "reset"

    def _draw_playback_widget(self):
        pass

    def _draw_tool_column_widget(self):
        pass

    def _draw_context_hint_widget(self):
        pass

    def _needs_presented_readback(self, dt):
        return self.readback or super()._needs_presented_readback(dt)

    def _finish_capture_and_recording(self, presented, dt):
        self.pixels = presented
        super()._finish_capture_and_recording(presented, dt)

    def close(self):
        self.release()
