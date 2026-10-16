"""Native workspace frontend of the shared ViewerApp frame and command pipeline."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.render.backend import RenderFlag
from mojive.ui.app import ViewerApp
from mojive.ui.scrollbar_visibility import (
    DEFAULT_HIDE_DELAY,
    NativeScrollbarVisibility,
    normalize_hide_delay,
)
from mojive.ui.workspace.design import DESIGN, rail_style
from mojive.ui.workspace.entities import EntityProjection
from mojive.ui.workspace.notices import draw_operation_notice
from mojive.ui.workspace.panel_manager import WorkspacePanels
from mojive.ui.workspace.shell import WorkspaceShell
from mojive.ui.workspace.viewcube import WorkspaceViewCube


class WorkspaceApp(ViewerApp):
    def release(self):
        try:
            workbench = getattr(self.design, "workbench", None)
            if workbench is not None:
                workbench.editor.close()
        finally:
            super().release()

    def _draw_center_notice(self, overlay, message, *, border=False):
        from mojive.ui.workspace.notices import draw_center_notice

        return draw_center_notice(self, overlay, message, border=border)

    def _draw_file_status(self, overlay, message, path, *, padding_pt=12):
        from mojive.ui.workspace.notices import draw_file_receipt

        return draw_file_receipt(self, overlay, message, path, padding_pt=padding_pt)

    def __init__(self, session, backend, window, **kwargs):
        self.design = kwargs.pop("design", None)
        if self.design is None:
            from mojive.design.runtime import DesignRuntime

            self.design = DesignRuntime()
        super().__init__(session, backend, window, **kwargs)
        config = kwargs.get("config")
        self.panels = WorkspacePanels(
            config=dict(config.panels) if config else {},
            builtin_ids=config.builtin_panels if config else None,
        )
        self.projection = EntityProjection(self)
        self.workspace = WorkspaceShell(self)
        from mojive.ui.workspace.typography import Typography

        self.workspace.ui.typography = Typography.resolve(self.preferences.get("typography"))
        self.scrollbar_visibility = NativeScrollbarVisibility(
            idle_seconds=self.preferences.get("scrollbar_hide_delay", DEFAULT_HIDE_DELAY)
        )
        if self.design is not None:
            from mojive.design.native.presentation import DesignPresentation

            self.workspace.ui.design = DesignPresentation(self.workspace.ui, self.design)
        self._numeric_cancel_frame = -1
        self._selection_status_revision = -1
        self._selection_notice_sequence = None
        self.view_cube = WorkspaceViewCube(self.view_cube.selection_padding)
        self.workspace.cube = self.view_cube
        self.window.set_layout_builder(self.workspace.ensure_layout)
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
            rail_style=rail_style(),
            scrollbar_hide_delay=self.scrollbar_visibility.idle_seconds,
            set_scrollbar_hide_delay=self.set_scrollbar_hide_delay,
        )

    def set_typography(self, value, *, persist=True):
        from dataclasses import asdict

        if persist:
            self.preferences.update({"typography": asdict(value)})
        self.workspace.ui.typography = value

    def set_scrollbar_hide_delay(self, value: float, *, persist: bool = True) -> None:
        value = normalize_hide_delay(value)
        if persist:
            self.preferences.update({"scrollbar_hide_delay": value})
        self.scrollbar_visibility.set_idle_seconds(value)

    def _draw_main_menu(self):
        self.scrollbar_visibility.update()
        self.scrollbar_visibility.sync_visible(self.workspace.open)
        self.projection.refresh_document()
        self.workspace.begin_frame()

    def frame(self):
        imgui.set_current_context(self.window._imgui_context)
        # DockSpace settings must be restored before ImGui starts the next frame.
        if self.workspace.restore_next:
            imgui.set_current_context(self.window._imgui_context)
            imgui.load_ini_settings_from_memory(self.workspace.saved)
            self.workspace.restore_next = False
        if self.design is None:
            with self.scrollbar_visibility.scope():
                super().frame()
        else:
            with self.design.scope(self.workspace):
                self.workspace.ui.design.begin_frame()
                with self.scrollbar_visibility.scope():
                    super().frame()

    def _present_frame(self, elapsed):
        if self.design is not None and self.design.workbench is not None:
            self.design.workbench.draw_overlay(self.workspace)
        super()._present_frame(elapsed)

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
        if workspace.pending:
            workspace.command(workspace.pending)
            workspace.pending = None
        workspace.ensure_layout()
        # DockBuilder changes must finish before the affected window's Begin/End scope.
        workspace.apply_timeline_layout()
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            0 if workspace.show_viewport_tab else imgui.DockNodeFlags_.auto_hide_tab_bar
        )
        imgui.set_next_window_class(wc)
        with self.scrollbar_visibility.panel_scope("Viewport", window_name="Viewport"):
            self.viewport_surface.begin("Viewport")
            self.scrollbar_visibility.bind_current_window()
        if not workspace.bottom_open:
            from mojive.ui.workspace.viewport import collapsed_bottom_height

            width, height = self.viewport_surface.size
            self.viewport_surface.size = (
                width,
                max(20 * workspace.s, height - collapsed_bottom_height(workspace)),
            )

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
        with self.scrollbar_visibility.panel_scope("Viewport", window_name="Viewport"):
            self._draw_owned_viewport_contents(preview_name, session_busy=session_busy)

    def _draw_owned_viewport_contents(self, preview_name="", *, session_busy=False):
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
        from mojive.ui.workspace.camera_preview import WorkspaceCameraPreview

        if not hasattr(self, "_workspace_camera_preview"):
            self._workspace_camera_preview = WorkspaceCameraPreview(self.workspace)
        self._workspace_camera_preview.draw(preview_name)

    def reset_layout(self, *, persist=True):
        super().reset_layout(persist=persist)
        from mojive.ui.workspace.docking import reset_visibility

        reset_visibility(self.workspace)

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
