"""Native workspace frontend of the shared ViewerApp frame and command pipeline."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.render.backend import RenderFlag
from mojive.ui.app import ViewerApp

from .design import DESIGN, RAIL_STYLE
from .entities import EntityProjection
from .notices import draw_operation_notice
from .panel_manager import WorkspacePanels
from .shell import WorkspaceShell
from .viewcube import WorkspaceViewCube


class WorkspaceApp(ViewerApp):
    def __init__(self, session, backend, window, **kwargs):
        super().__init__(session, backend, window, **kwargs)
        config = kwargs.get("config")
        self.panels = WorkspacePanels(
            config=dict(config.panels) if config else {},
            builtin_ids=config.builtin_panels if config else None,
        )
        self.projection = EntityProjection(self)
        self.workspace = WorkspaceShell(self)
        self._numeric_cancel_frame = -1
        self._selection_status_revision = -1
        self._selection_notice_sequence = None
        self.view_cube = WorkspaceViewCube(self.view_cube.selection_padding)
        self.workspace.cube = self.view_cube
        self.window._layout_done = True
        self.gizmo.set_style("2d")

    @property
    def entity(self):
        return self.projection.entity

    @property
    def entities(self):
        return self.projection.entities

    @property
    def initial_entities(self):
        return self.projection.initial_entities

    @property
    def selected(self):
        return self.projection.selected

    @selected.setter
    def selected(self, name):
        self.projection.selected = name

    def refresh_document(self):
        self.projection.refresh_document()

    def apply_document_edits(self):
        self.projection.apply_document_edits()

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

    def _panel_context(self):
        source, _ratio = self.workspace.ui.font_sources[2]
        fonts = replace(
            self.window.font_report,
            mono=source.label,
            mono_path=source.path,
            mono_index=source.index,
        )
        return replace(
            super()._panel_context(),
            translate=self.workspace.ui.translate,
            font_report=fonts,
            rail_style=RAIL_STYLE,
        )

    def _draw_main_menu(self):
        self.projection.refresh_document()
        self.workspace.begin_frame()

    def frame(self):
        # DockSpace settings must be restored before ImGui starts the next frame.
        if self.workspace.restore_next:
            imgui.set_current_context(self.window._imgui_context)
            imgui.load_ini_settings_from_memory(self.workspace.saved)
            self.workspace.restore_next = False
        super().frame()

    def _poll_application_shortcuts(self):
        # A numeric gesture owns Escape before application selection shortcuts.
        inspector = self.panels.get("inspector")
        numeric_edit = self.workspace._numeric_edit is not None or (
            inspector is not None and inspector._edit_transaction
        )
        if numeric_edit and imgui.is_key_pressed(imgui.Key.escape, False):
            result = self.session.submit(cmd.CancelEditTransaction())
            if result.ok:
                self.workspace._numeric_edit = None
                if inspector is not None:
                    inspector._edit_transaction = False
                self._numeric_cancel_frame = imgui.get_frame_count()
                imgui.internal.clear_active_id()
                self.projection.refresh_document()
            return
        super()._poll_application_shortcuts()
        io = imgui.get_io()
        if self._scene_input_blocked() or imgui.is_any_item_active() or io.key_ctrl or io.key_super:
            return
        if imgui.is_key_pressed(imgui.Key.v, False):
            self.tool = "select"
        if imgui.is_key_pressed(imgui.Key.t, False):
            self.workspace.toggle_timeline()

    def _draw_collapsed_output(self):
        pass

    def _draw_viewport_status(self, overlay):
        from mojive.ui.files import message_path

        self._status_path_bounds = self._status_notice_bounds = None
        if self.model_edits.active and self._viewport_rect[3] < 300 * self.workspace.s:
            return
        message = self.output.active_status()
        if message is None or message.sequence == self._selection_notice_sequence:
            return
        path = message_path(message.text, message.copy_text)
        if path is not None:
            self._draw_file_status(overlay, message, path, padding_pt=DESIGN.file_receipt_padding)
        elif self.viewport_layers.viewport_ui:
            draw_operation_notice(self, overlay, message)

    def _sync_session_status(self):
        session = self.session
        pending = session.message_revision != self._seen_message_revision
        changed = session.selection_revision != self._selection_status_revision
        selected = session.selected_node
        selection_text = selected.name if selected is not None else "Selection cleared"
        redundant = (
            pending
            and changed
            and session.last_message == selection_text
            and session.last_message_level == "info"
            and session.last_message_copy_text is None
        )
        super()._sync_session_status()
        self._selection_status_revision = session.selection_revision
        if redundant:
            message = self.output.active_status()
            self._selection_notice_sequence = message.sequence if message is not None else None

    def _draw_model_drop_overlay(self, overlay):
        if (
            self.model_edits.active
            and self._has_scene_content()
            and not self.window.file_drag_active
        ):
            return
        super()._draw_model_drop_overlay(overlay)

    def _begin_viewport_panel(self):
        workspace = self.workspace
        if workspace.needs_layout:
            workspace.build()
        if workspace.pending:
            workspace.command(workspace.pending)
            workspace.pending = None
        # DockBuilder changes must finish before the affected window's Begin/End scope.
        workspace.apply_timeline_layout()
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            0 if workspace.show_viewport_tab else imgui.DockNodeFlags_.auto_hide_tab_bar
        )
        imgui.set_next_window_class(wc)
        self.viewport_surface.begin("Viewport")
        if not workspace.bottom_open:
            width, height = self.viewport_surface.size
            self.viewport_surface.size = (width, max(20 * workspace.s, height - 38 * workspace.s))

    def _scene_input_blocked(self):
        if self._numeric_cancel_frame == imgui.get_frame_count():
            return True
        if super()._scene_input_blocked():
            return True
        # A captured scene gesture retains ownership while crossing workspace controls.
        if self.router.owns_scene_pointer or self.gizmo.using:
            return False
        cursor = imgui.get_io().mouse_pos
        return any(
            x <= cursor.x <= x + w and y <= cursor.y <= y + h
            for x, y, w, h in self.workspace.input_overlays
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
        self.projection.refresh_document()
        self.workspace.viewport(preview_name)
        if self.viewport_layers.gizmos:
            self._draw_joint_limit_controls()
            self._draw_joint_gizmo_picker()

    def _draw_workspace_camera_preview(self, preview_name):
        from .camera_preview import WorkspaceCameraPreview

        if not hasattr(self, "_workspace_camera_preview"):
            self._workspace_camera_preview = WorkspaceCameraPreview(self.workspace)
        self._workspace_camera_preview.draw(preview_name)

    def reset_layout(self, *, persist=True):
        super().reset_layout(persist=persist)
        self.window._layout_done = True
        self.workspace.pending = "reset"

    def show_physics_options(self):
        super().show_physics_options()
        self.workspace.inspector_tab = "Physics"
        self.workspace.activate("Inspector")

    def _draw_playback_widget(self):
        pass

    def _draw_tool_column_widget(self):
        pass

    def _draw_context_hint_widget(self):
        pass
