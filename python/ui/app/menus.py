"""App: menus."""

from __future__ import annotations

import random
import sys
import webbrowser
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.capture import CaptureSurface, RecordingPhase
from mojive.types import Light, LightType, MeshShape
from mojive.ui.input_bindings import InputAction
from mojive.ui.panels import (
    button_width,
)
from mojive.ui.theme import THEME

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode


from .support import _equal_modal_buttons, _model_filters, _prepare_modal


class _Menus:
    """Private menus methods of ViewerApp; state belongs to its owner."""

    def _begin_main_menu(self, label: str, enabled: bool = True) -> bool:
        # Horizontal menu entries need less spacing than vertical popup rows.
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(8.0 * self.window.style_scale, 8.0 * self.window.style_scale),
        )
        opened = imgui.begin_menu(label, enabled)
        imgui.pop_style_var()
        return opened

    def _draw_main_menu(self) -> None:
        actions: list[Callable[[], object]] = []
        shortcut = "Cmd" if sys.platform == "darwin" else "Ctrl"
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(20.0 * self.window.style_scale, 8.0 * self.window.style_scale),
        )
        if imgui.begin_main_menu_bar():
            imgui.set_cursor_pos_x(4.0 * self.window.style_scale)
            self._draw_file_menu(actions, shortcut)
            self._draw_edit_menu(actions, shortcut)
            self._draw_entity_menu(
                actions, shortcut, bool(self.session.adapter.caps.scene_authoring)
            )
            self._draw_view_menu(actions, shortcut)
            self._draw_window_menu(actions)
            self._draw_help_menu(actions)
            self._draw_menu_document_name()
            imgui.end_main_menu_bar()
        imgui.pop_style_var()

        # Document replacement and panel creation must run outside the menu's ImGui scope.
        for action in actions:
            action()

    def _menu_action(
        self,
        actions: list[Callable[[], object]],
        label: str,
        action: Callable[..., object],
        *args: object,
        shortcut: str = "",
        enabled: bool = True,
        **kwargs: object,
    ) -> None:
        if imgui.menu_item(self.localizer.text(label), shortcut, False, enabled)[0]:
            actions.append(partial(action, *args, **kwargs))

    def _draw_file_menu(self, actions, shortcut: str) -> None:
        t = self.localizer.text
        if not self._begin_main_menu(t("File")):
            return
        caps = self.session.adapter.caps
        can_new = caps.supports("scene_new")
        can_open = caps.supports("scene_open")
        can_save = caps.supports("scene_save")
        if can_new:
            self._menu_action(
                actions,
                "New Scene",
                self._request_document_action,
                "new_scene",
                shortcut=f"{shortcut}+N",
            )
        if can_open:
            self._menu_action(
                actions,
                "Open Scene...",
                self._open_scene_dialog,
                "open",
                shortcut=f"{shortcut}+O",
                enabled=self._scene_dialog is None,
            )
        if can_save:
            self._menu_action(
                actions,
                "Save",
                self._save_current_scene,
                shortcut=f"{shortcut}+S",
                enabled=self.session.dirty,
            )
            self._menu_action(
                actions,
                "Save As...",
                self._open_scene_dialog,
                "save",
                shortcut=f"{shortcut}+Shift+S",
            )
        if _model_filters(caps):
            if can_new or can_open or can_save:
                imgui.separator()
            self._menu_action(
                actions,
                "Open Model...",
                self._open_model_dialog,
                shortcut=f"{shortcut}+O" if not can_open else "",
                enabled=self._model_dialog is None,
            )
            if caps.model_composition:
                self._menu_action(
                    actions,
                    "Add Models...",
                    self._open_model_dialog,
                    "add",
                    enabled=self._model_dialog is None,
                )
                removable = [item for item in self.session.scene_models if item.removable]
                if imgui.begin_menu(t("Remove Model"), bool(removable)):
                    for item in removable:
                        if imgui.menu_item(item.name, "", False)[0]:
                            actions.append(partial(self.remove_model, item.model_id))
                    imgui.end_menu()
            self._menu_action(
                actions,
                "Reload Model",
                self._queue_model_load,
                "reload",
                self.session.asset_path,
                shortcut=f"{shortcut}+Shift+O",
                enabled=caps.reload and self.session.asset_path is not None,
            )
        if caps.scene_authoring and imgui.begin_menu(t("Resource Directories")):
            self._menu_action(
                actions,
                "Add Directory...",
                self._open_resource_dialog,
                enabled=self._resource_dialog is None,
            )
            for root in self.session.adapter.resource_roots:
                if imgui.menu_item(f"{t('Remove')} {root}", "", False)[0]:
                    actions.append(partial(self.session.submit, cmd.RemoveResourceRoot(root)))
            imgui.end_menu()
        imgui.separator()
        self._menu_action(
            actions, "Quit", self._request_document_action, "quit", shortcut=f"{shortcut}+Q"
        )
        imgui.end_menu()

    def _save_current_scene(self) -> None:
        if self.session.asset_path is None:
            self._open_scene_dialog("save")
        else:
            self._request_scene_save(self.session.asset_path)

    def _draw_edit_menu(self, actions, shortcut: str) -> None:
        if not self._begin_main_menu(self.localizer.text("Edit")):
            return
        history = self.session.adapter.caps.edit_history
        self._menu_action(
            actions,
            "Undo",
            self.session.submit,
            cmd.Undo(),
            shortcut=f"{shortcut}+Z",
            enabled=history and self.session.can_undo,
        )
        self._menu_action(
            actions,
            "Redo",
            self.session.submit,
            cmd.Redo(),
            shortcut=f"{shortcut}+Shift+Z",
            enabled=history and self.session.can_redo,
        )
        imgui.separator()
        self._menu_action(
            actions, "Settings...", self.panels.open_panel, "Settings", shortcut=f"{shortcut}+,"
        )
        imgui.end_menu()

    def _draw_view_menu(self, actions, shortcut: str) -> None:
        t = self.localizer.text
        if not self._begin_main_menu(t("View")):
            return
        self._menu_action(
            actions,
            "Frame All",
            self._frame_all_from_menu,
            shortcut=self.input_bindings.label(InputAction.FRAME_SCENE),
            enabled=self._model_camera_id < 0,
        )
        if imgui.begin_menu(t("Capture")):
            self._menu_action(
                actions,
                "Scene Image",
                self.request_capture,
                surface=CaptureSurface.SCENE,
                shortcut=f"{shortcut}+Shift+P",
            )
            self._menu_action(
                actions,
                "Viewport with UI",
                self.request_capture,
                surface=CaptureSurface.VIEWPORT,
            )
            self._menu_action(
                actions,
                "Entire Window",
                self.request_capture,
                surface=CaptureSurface.WINDOW,
            )
            imgui.end_menu()
        self._draw_recording_menu(actions, shortcut)
        self._menu_action(actions, "Recording Settings...", self._open_recording_settings)
        imgui.separator()
        self._menu_action(actions, "Layers...", self.panels.open_panel, "Layers")
        self._draw_entity_visibility_menu()
        imgui.end_menu()

    def _frame_all_from_menu(self) -> None:
        self._leave_model_camera()
        self._frame_scene(animate=True)

    def _draw_recording_menu(self, actions, shortcut: str) -> None:
        t = self.localizer.text
        if self.recording.phase is RecordingPhase.COUNTDOWN:
            self._menu_action(
                actions,
                "Cancel Recording",
                self._request_recording_stop,
                shortcut=f"{shortcut}+Shift+R",
            )
        elif self.recording.phase is RecordingPhase.FINALIZING:
            imgui.menu_item(t("Finalizing recording"), "", False, False)
        elif self.recording.active:
            if self._viewport_recording_phase is RecordingPhase.PAUSED:
                self._menu_action(actions, "Resume Recording", self.resume_recording)
            else:
                self._menu_action(actions, "Pause Recording", self.pause_recording)
            self._menu_action(
                actions,
                "Stop Recording",
                self._request_recording_stop,
                shortcut=f"{shortcut}+Shift+R",
            )
        elif imgui.begin_menu(t("Record")):
            for label, surface in (
                ("Scene Only", CaptureSurface.SCENE),
                ("Viewport with UI", CaptureSurface.VIEWPORT),
                ("Entire Window", CaptureSurface.WINDOW),
            ):
                self._menu_action(actions, label, self._start_recording_from_menu, surface)
            imgui.end_menu()

    def _start_recording_from_menu(self, surface: CaptureSurface) -> None:
        try:
            self.start_recording(surface=surface)
        except Exception as exc:
            self.session.report_message(
                f"{self.localizer.text('Recording failed')}: {exc}", level="error"
            )

    def _open_recording_settings(self) -> None:
        self.panels.open_panel("Settings")
        self.panels.get("Settings").show_category("Recording")

    def _draw_entity_visibility_menu(self) -> None:
        t = self.localizer.text
        if imgui.menu_item(t("Camera & Light Helpers"), "", bool(self.scene_entities.visible))[0]:
            self.scene_entities.visible = not self.scene_entities.visible
        if imgui.is_item_hovered():
            imgui.set_tooltip(t("Show camera and light icons in the scene"))
        if imgui.menu_item(
            t("Selected Camera & Light Volumes"),
            "",
            bool(self.scene_entities.show_influence),
            bool(self.scene_entities.visible),
        )[0]:
            self.scene_entities.show_influence = not self.scene_entities.show_influence
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(t("Show the selected camera frustum or light range"))

    def _draw_window_menu(self, actions) -> None:
        t = self.localizer.text
        if not self._begin_main_menu(t("Window")):
            return
        for panel in self.panels:
            if (
                panel.enabled
                and not panel.modal
                and imgui.menu_item(t(panel.name), panel.shortcut, panel.open)[0]
            ):
                actions.append(panel.toggle)
        for panel_id, title in self.panels.unloaded_builtins():
            self._menu_action(actions, title, self.panels.open, panel_id)
        imgui.separator()
        self._menu_action(actions, "Reset Layout", self.reset_layout)
        caps = self.session.adapter.caps
        self._menu_action(
            actions,
            "Physics Options...",
            self.show_physics_options,
            enabled=caps.simulation or caps.supports("physics.options"),
        )
        imgui.end_menu()

    def _draw_help_menu(self, actions) -> None:
        if not self._begin_main_menu(self.localizer.text("Help")):
            return
        self._menu_action(
            actions, "Interaction Reference", self.panels.open_panel, "Help", shortcut="F1"
        )
        self._menu_action(
            actions, "Documentation", webbrowser.open, "https://github.com/acrlw/mojive#readme"
        )
        imgui.separator()
        self._menu_action(actions, "About", self.panels.open_panel, "Info")
        imgui.end_menu()

    def _draw_menu_document_name(self) -> None:
        path = self.session.asset_path
        if path is not None:
            document = path.name
        elif any(
            self.session.adapter.caps.supports(name)
            for name in ("scene_new", "scene_open", "scene_save")
        ):
            document = self.localizer.text("Untitled")
        else:
            return
        if self.session.dirty:
            document += " ●"
        target_x = imgui.get_window_width() - imgui.calc_text_size(document).x - 10.0
        imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), target_x))
        imgui.text_disabled(document)

    def show_physics_options(self) -> None:
        """Reveal the environment's physics controls from the Window menu."""
        node = next((n for n in self.session.nodes if n.type is NodeType.ENVIRONMENT), None)
        if node is not None:
            self.session.submit(cmd.SelectNode(node.node_id))
            self.panels.open_panel("Inspector")
            self.panels.get("Inspector").show_environment_physics()

    def _draw_entity_menu(self, actions, shortcut: str, enabled: bool) -> None:
        t = self.localizer.text
        if not self._begin_main_menu(t("Entity"), enabled):
            return
        if imgui.begin_menu(t("Create")):
            self._draw_create_entity_menu(actions)
            imgui.end_menu()
        selected = bool(self._selected_entity()) or self._selected_model_element() is not None
        self._menu_action(
            actions,
            "Duplicate",
            self._duplicate_selected,
            shortcut=f"{shortcut}+D",
            enabled=selected,
        )
        self._menu_action(
            actions, "Rename", self._request_selected_rename, shortcut="F2", enabled=selected
        )
        self._menu_action(
            actions, "Delete", self._remove_selected, shortcut="Delete", enabled=selected
        )
        imgui.end_menu()

    def _draw_create_entity_menu(self, actions) -> None:
        for label, shape in (
            ("Box", MeshShape.BOX),
            ("Sphere", MeshShape.SPHERE),
            ("Cylinder", MeshShape.CYLINDER),
            ("Cone", MeshShape.CONE),
            ("Plane", MeshShape.PLANE),
        ):
            self._menu_action(actions, label, self._add_scene_object, shape, label.lower())
        self._menu_action(
            actions,
            "Ellipsoid",
            self._add_scene_object,
            MeshShape.SPHERE,
            "ellipsoid",
            size=(0.65, 0.45, 0.35),
        )
        topology = self.session.adapter.caps.topology_editing
        self._menu_action(
            actions,
            "Capsule",
            self._add_model_primitive,
            "capsule",
            "capsule",
            enabled=topology,
        )
        imgui.separator()
        self._menu_action(actions, "Point Light", self._add_scene_light)
        self._menu_action(actions, "Camera", self._add_scene_camera)
        self._menu_action(actions, "Site", self._add_model_site, enabled=topology)

    def _entity_name(self, base: str) -> str:
        names = {node.name for node in self.session.nodes}
        if base not in names:
            return base
        index = 2
        while f"{base} {index}" in names:
            index += 1
        return f"{base} {index}"

    def _model_child_parent(self) -> SceneNode | None:
        """Resolve the owning MuJoCo body for a top-level create action."""
        node = self.session.selected_node
        while node is not None:
            if node.type in (NodeType.MODEL, NodeType.WORLD, NodeType.LINK, NodeType.ROBOT) and (
                node.type in (NodeType.WORLD, NodeType.MODEL) or node.source_editable
            ):
                return node
            node = self.session.node(node.parent)
        return next(
            (
                node
                for node in self.session.nodes
                if node.type is NodeType.WORLD and node.parent < 0
            ),
            None,
        )

    def _add_model_site(self) -> None:
        parent = self._model_child_parent()
        if parent is None:
            self.session.report_message(
                self.localizer.text("Select a model or body before creating a site")
            )
            return
        with self._entity_creation("Create site"):
            result = self.session.submit(
                cmd.AddModelElement(parent.node_id, "site", self._entity_name("site"))
            )
            if result.ok:
                self._submit_creation_style(
                    result,
                    cmd.SetGeometryColor(result.entity_id, self._next_entity_color()),
                    cmd.SelectNode(result.entity_id),
                )

    def _add_model_primitive(self, primitive: str, base_name: str) -> None:
        parent = self._model_child_parent()
        if parent is None:
            self.session.report_message(
                self.localizer.text("Select a model or body before creating geometry")
            )
            return
        with self._entity_creation(f"Create {base_name}"):
            result = self.session.submit(
                cmd.AddModelElement(
                    parent.node_id,
                    f"geom:{primitive}",
                    self._entity_name(base_name),
                )
            )
            if result.ok:
                self._submit_creation_style(
                    result,
                    cmd.SetGeometryColor(result.entity_id, self._next_entity_color()),
                    cmd.SelectNode(result.entity_id),
                )

    def _submit_creation_style(self, result, *commands) -> None:
        """Submit the styling of a created element and select it.

        A deferred edit returns a batch-local key instead of a node ID, so these
        commands bind to the element when the pending batch applies.
        """

        key = getattr(result, "entity_key", "")
        for command in commands:
            self.session.submit(replace(command, node_id=-1, node_key=key) if key else command)

    @contextmanager
    def _entity_creation(self, label: str):
        """Coalesce creation and initial styling into one undoable edit."""

        opened = False
        if not self.session.editing and self.session.adapter.caps.edit_history:
            opened = self.session.submit(cmd.BeginEditTransaction(label)).ok
        try:
            yield
        finally:
            if opened:
                self.session.submit(cmd.EndEditTransaction())

    def _next_entity_color(self) -> tuple[float, float, float, float]:
        """Choose one authored-object color from the active theme palette."""

        theme = getattr(self, "theme", THEME)
        palette = theme.entity_palette or THEME.entity_palette
        return random.choice(palette)

    def _add_scene_object(
        self,
        shape: MeshShape,
        base_name: str,
        *,
        size: tuple[float, float, float] | None = None,
    ) -> None:
        name = self._entity_name(base_name)
        color = self._next_entity_color()
        with self._entity_creation(f"Create {base_name}"):
            if shape is MeshShape.PLANE and self.session.adapter.caps.topology_editing:
                world = next(
                    (
                        node
                        for node in self.session.nodes
                        if node.type is NodeType.WORLD and node.parent < 0
                    ),
                    None,
                )
                if world is not None:
                    result = self.session.submit(
                        cmd.AddModelElement(
                            world.node_id,
                            "geom:plane",
                            name,
                        )
                    )
                    if result.ok:
                        self._submit_creation_style(
                            result,
                            cmd.SetGeometryColor(result.entity_id, color),
                            cmd.SelectNode(result.entity_id),
                        )
                        return
            position = tuple(float(value) for value in self._camera_view().target)
            size = size or ((4.0, 4.0, 0.02) if shape is MeshShape.PLANE else (0.5, 0.5, 0.5))
            result = self.session.submit(
                cmd.AddSceneObject(shape, name, size=size, position=position, color=color)
            )
            if result.ok:
                self.session.submit(cmd.Select(result.entity_id))

    def _add_scene_light(self) -> None:
        view = self._camera_view()
        name = self._entity_name("point light")
        result = self.session.submit(
            cmd.AddSceneLight(
                name,
                Light(type=LightType.POINT, position=np.asarray(view.eye, np.float32).copy()),
            )
        )
        if result.ok:
            node = next(
                (
                    node
                    for node in reversed(self.session.nodes)
                    if node.type is NodeType.LIGHT and node.name == name
                ),
                None,
            )
            if node is not None:
                self.session.submit(cmd.Select(node.object_id))

    def _add_scene_camera(self) -> None:
        name = self._entity_name("camera")
        result = self.session.submit(cmd.AddSceneCamera(name, self._camera_view()))
        if result.ok:
            node = next(
                (
                    node
                    for node in reversed(self.session.nodes)
                    if node.type is NodeType.CAMERA and node.name == name
                ),
                None,
            )
            if node is not None:
                self.session.submit(cmd.Select(node.object_id))

    def _duplicate_selected(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            result = self.session.submit(cmd.DuplicateModelElement(node.node_id))
            if result.ok:
                self._submit_creation_style(result, cmd.SelectNode(result.entity_id))
            return
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.DuplicateSceneEntity(object_id))

    def _remove_selected(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            self.session.submit(cmd.RemoveModelElement(node.node_id))
            return
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.RemoveSceneEntity(object_id))

    def _selected_model_element(self) -> SceneNode | None:
        node = self.session.selected_node
        if (
            node is None
            or node.model_id < 0
            or not node.source_editable
            or node.type
            not in {
                NodeType.ROBOT,
                NodeType.LINK,
                NodeType.GEOM,
                NodeType.JOINT,
                NodeType.SITE,
                NodeType.CAMERA,
                NodeType.LIGHT,
            }
        ):
            return None
        return node

    def _selected_entity(self) -> int:
        node = self.session.selected_node
        if (
            node is None
            or node.model_id >= 0
            or node.type not in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
        ):
            return 0
        return int(node.object_id)

    def request_rename(self, object_id: int) -> None:
        node = self.session.node_by_object_id(object_id)
        if (
            node is None
            or node.model_id >= 0
            or node.type not in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
        ):
            return
        self._rename_object_id = int(object_id)
        self._rename_model_node_id = -1
        self._rename_value = node.name
        self._open_rename_popup = True

    def request_model_rename(self, node_id: int) -> None:
        node = self.session.node(node_id)
        if (
            node is None
            or node.model_id < 0
            or not node.source_editable
            or node.type
            not in {
                NodeType.ROBOT,
                NodeType.LINK,
                NodeType.GEOM,
                NodeType.JOINT,
                NodeType.SITE,
                NodeType.CAMERA,
                NodeType.LIGHT,
            }
        ):
            return
        self._rename_object_id = 0
        self._rename_model_node_id = int(node_id)
        self._rename_value = node.name
        self._open_rename_popup = True

    def _request_selected_rename(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            self.request_model_rename(node.node_id)
        else:
            self.request_rename(self.session.selected)

    def _report_model_error(self, message: str) -> None:
        self._model_load_error = message
        self._show_model_load_error = True
        self.session.report_message(message, level="error", duration=10.0)

    def _draw_model_load_error(self) -> None:
        t = self.localizer.text
        popup_title = f"{t('File operation failed')}###File operation failed"
        if self._show_model_load_error:
            imgui.open_popup(popup_title)
            self._show_model_load_error = False
        if imgui.is_popup_open(popup_title):
            _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(self._model_load_error)
        imgui.spacing()
        copy_error = t("Copy error")
        if imgui.button(copy_error, imgui.ImVec2(button_width(copy_error, 110.0), 0.0)):
            imgui.set_clipboard_text(self._model_load_error)
        imgui.same_line()
        ok = t("OK")
        if imgui.button(ok, imgui.ImVec2(button_width(ok, 88.0), 0.0)) or imgui.is_key_pressed(
            imgui.Key.escape, False
        ):
            imgui.close_current_popup()
        imgui.end_popup()

    def _request_document_action(self, action: str, path: Path | None = None) -> None:
        if self.session.dirty:
            self._pending_document_action = (action, path)
            return
        self._execute_document_action(action, path)

    def _execute_document_action(self, action: str, path: Path | None = None) -> None:
        if action == "new_scene":
            result = self.session.submit(cmd.NewScene())
            if result.ok:
                self._after_model_change()
                self._set_model_drop_notice(self.localizer.text("New Mojive scene"))
            else:
                self._report_model_error(result.message)
        elif action == "open_scene" and path is not None:
            self._queue_scene_open(path)
        elif action == "quit":
            self._closing_without_save = True
            self.window.request_close()

    def _draw_unsaved_changes(self) -> None:
        pending = self._pending_document_action
        if pending is None:
            return
        t = self.localizer.text
        popup_title = f"{t('Unsaved changes')}###Unsaved changes"
        imgui.open_popup(popup_title)
        _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        name = (
            self.session.asset_path.name if self.session.asset_path is not None else t("Untitled")
        )
        imgui.text(t("Save changes to this file?"))
        imgui.text_wrapped(name)
        imgui.spacing()
        cancel, discard, save = _equal_modal_buttons(
            (t("Cancel"), t("Discard"), t("Save")), self.theme, primary=2
        )
        if save:
            if self.session.asset_path is None:
                self._after_save_action = pending
                self._pending_document_action = None
                self._open_scene_dialog("save")
            else:
                self._pending_document_action = None
                self._request_scene_save(self.session.asset_path, pending)
            imgui.close_current_popup()
        elif discard:
            self.model_edits.clear()
            self._pending_document_action = None
            self._execute_document_action(*pending)
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._pending_document_action = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_pose_save_prompt(self) -> None:
        pending = self._pending_pose_save
        if pending is None:
            return
        t = self.localizer.text
        labels = (t("Cancel"), t("Save without keyframe"), t("Save as key0"))
        popup_title = f"{t('Save current pose')}###Save current pose"
        imgui.open_popup(popup_title)
        _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(
            t(
                "The current pose differs from the model default. Add the current qpos as "
                "keyframe key0 in the exported MJCF?"
            )
        )
        imgui.spacing()
        target, after = pending
        cancel, without_key, save_key = _equal_modal_buttons(labels, self.theme, primary=2)
        if save_key:
            self._pending_pose_save = None
            if self.save_scene(target, current_pose_keyframe="key0").ok and after is not None:
                self._execute_document_action(*after)
            imgui.close_current_popup()
        elif without_key:
            self._pending_pose_save = None
            if self.save_scene(target).ok and after is not None:
                self._execute_document_action(*after)
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._pending_pose_save = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_rename_popup(self) -> None:
        popup_title = f"{self.localizer.text('Rename Entity')}###Rename Entity"
        if self._open_rename_popup:
            imgui.open_popup(popup_title)
            self._open_rename_popup = False
        width = min(220.0, max(1.0, float(imgui.get_main_viewport().work_size.x) - 32.0))
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(width, 0.0),
            imgui.ImVec2(width, float(np.finfo(np.float32).max)),
        )
        visible = imgui.begin_popup(popup_title, imgui.WindowFlags_.always_auto_resize.value)
        if not visible:
            return
        imgui.text_disabled(self.localizer.text("Rename"))
        imgui.separator()
        imgui.set_next_item_width(-1.0)
        submitted, self._rename_value = imgui.input_text(
            "##entity_name",
            self._rename_value,
            imgui.InputTextFlags_.enter_returns_true.value
            | imgui.InputTextFlags_.auto_select_all.value,
        )
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here(-1)
        if submitted and self._rename_value.strip():
            value = self._rename_value.strip()
            command = (
                cmd.RenameModelElement(self._rename_model_node_id, value)
                if self._rename_model_node_id >= 0
                else cmd.RenameSceneEntity(self._rename_object_id, value)
            )
            self.session.submit(command)
            imgui.close_current_popup()
        elif imgui.is_key_pressed(imgui.Key.escape, False):
            imgui.close_current_popup()
        imgui.end_popup()
