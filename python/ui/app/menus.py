"""App: menus."""

from __future__ import annotations

import random
import sys
import webbrowser
from contextlib import contextmanager
from dataclasses import replace
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
        t = self.localizer.text
        caps = self.session.adapter.caps
        can_load = bool(_model_filters(caps))
        can_edit = bool(caps.scene_authoring)
        can_scene_files = bool(caps.scene_files)
        shortcut = "Cmd" if sys.platform == "darwin" else "Ctrl"
        new_scene = False
        open_scene = False
        save_scene = False
        save_scene_as = False
        open_model = False
        add_model = False
        remove_model_id = -1
        add_resource_root = False
        remove_resource_root: Path | None = None
        reload_model = False
        undo = False
        redo = False
        open_settings = False
        open_recording_settings = False
        frame_scene = False
        capture_surface: CaptureSurface | None = None
        start_recording_surface: CaptureSurface | None = None
        pause_recording = False
        stop_recording = False
        reset_layout = False
        open_help = False
        open_documentation = False
        open_about = False
        quit_viewer = False
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(20.0 * self.window.style_scale, 8.0 * self.window.style_scale),
        )
        if imgui.begin_main_menu_bar():
            imgui.set_cursor_pos_x(4.0 * self.window.style_scale)
            if self._begin_main_menu(t("File")):
                if can_scene_files:
                    new_scene, _ = imgui.menu_item(t("New Scene"), f"{shortcut}+N", False)
                    open_scene, _ = imgui.menu_item(
                        t("Open Scene..."),
                        f"{shortcut}+O",
                        False,
                        self._scene_dialog is None,
                    )
                    save_scene, _ = imgui.menu_item(
                        t("Save"), f"{shortcut}+S", False, self.session.dirty
                    )
                    save_scene_as, _ = imgui.menu_item(
                        t("Save As..."), f"{shortcut}+Shift+S", False
                    )
                if can_load:
                    if can_scene_files:
                        imgui.separator()
                    open_model, _ = imgui.menu_item(
                        t("Open Model..."),
                        f"{shortcut}+O" if not can_scene_files else "",
                        False,
                        self._model_dialog is None,
                    )
                    if caps.model_composition:
                        add_model, _ = imgui.menu_item(
                            t("Add Models..."),
                            "",
                            False,
                            self._model_dialog is None,
                        )
                        removable = [item for item in self.session.scene_models if item.removable]
                        if imgui.begin_menu(t("Remove Model"), bool(removable)):
                            for item in removable:
                                clicked, _ = imgui.menu_item(item.name, "", False)
                                if clicked:
                                    remove_model_id = item.model_id
                            imgui.end_menu()
                    reload_model, _ = imgui.menu_item(
                        t("Reload Model"),
                        f"{shortcut}+Shift+O",
                        False,
                        caps.reload and self.session.asset_path is not None,
                    )
                if can_scene_files and imgui.begin_menu(t("Resource Directories")):
                    add_resource_root, _ = imgui.menu_item(
                        t("Add Directory..."), "", False, self._resource_dialog is None
                    )
                    for root in self.session.adapter.resource_roots:
                        clicked, _ = imgui.menu_item(f"{t('Remove')} {root}", "", False)
                        if clicked:
                            remove_resource_root = root
                    imgui.end_menu()
                imgui.separator()
                quit_viewer, _ = imgui.menu_item(t("Quit"), f"{shortcut}+Q", False, True)
                imgui.end_menu()
            if self._begin_main_menu(t("Edit")):
                undo, _ = imgui.menu_item(
                    t("Undo"),
                    f"{shortcut}+Z",
                    False,
                    caps.edit_history and self.session.can_undo,
                )
                redo, _ = imgui.menu_item(
                    t("Redo"),
                    f"{shortcut}+Shift+Z",
                    False,
                    caps.edit_history and self.session.can_redo,
                )
                imgui.separator()
                open_settings, _ = imgui.menu_item(t("Settings..."), f"{shortcut}+,", False)
                imgui.end_menu()
            self._draw_entity_menu(shortcut, can_edit)
            if self._begin_main_menu(t("View")):
                frame_scene, _ = imgui.menu_item(
                    t("Frame All"), self.input_bindings.label(InputAction.FRAME_SCENE), False
                )
                if imgui.begin_menu(t("Capture")):
                    clicked, _ = imgui.menu_item(t("Scene Image"), f"{shortcut}+Shift+P", False)
                    if clicked:
                        capture_surface = CaptureSurface.SCENE
                    clicked, _ = imgui.menu_item(t("Viewport with UI"), "", False)
                    if clicked:
                        capture_surface = CaptureSurface.VIEWPORT
                    clicked, _ = imgui.menu_item(t("Entire Window"), "", False)
                    if clicked:
                        capture_surface = CaptureSurface.WINDOW
                    imgui.end_menu()
                if self.recording.phase is RecordingPhase.COUNTDOWN:
                    stop_recording, _ = imgui.menu_item(
                        t("Cancel Recording"), f"{shortcut}+Shift+R", False
                    )
                elif self.recording.active:
                    if self._viewport_recording_phase is RecordingPhase.PAUSED:
                        pause_recording, _ = imgui.menu_item(t("Resume Recording"), "", False)
                    else:
                        pause_recording, _ = imgui.menu_item(t("Pause Recording"), "", False)
                    stop_recording, _ = imgui.menu_item(
                        t("Stop Recording"), f"{shortcut}+Shift+R", False
                    )
                elif imgui.begin_menu(t("Record")):
                    clicked, _ = imgui.menu_item(t("Scene Only"), "", False)
                    if clicked:
                        start_recording_surface = CaptureSurface.SCENE
                    clicked, _ = imgui.menu_item(t("Viewport with UI"), "", False)
                    if clicked:
                        start_recording_surface = CaptureSurface.VIEWPORT
                    clicked, _ = imgui.menu_item(t("Entire Window"), "", False)
                    if clicked:
                        start_recording_surface = CaptureSurface.WINDOW
                    imgui.end_menu()
                open_recording_settings, _ = imgui.menu_item(t("Recording Settings..."), "", False)
                imgui.separator()
                layers, _ = imgui.menu_item(t("Layers..."), "", False)
                if layers:
                    self.panels.open_panel("Layers")
                helpers, _ = imgui.menu_item(
                    t("Camera & Light Helpers"), "", bool(self.scene_entities.visible)
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(t("Show camera and light icons in the scene"))
                if helpers:
                    self.scene_entities.visible = not self.scene_entities.visible
                influence, _ = imgui.menu_item(
                    t("Selected Camera & Light Volumes"),
                    "",
                    bool(self.scene_entities.show_influence),
                    bool(self.scene_entities.visible),
                )
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                    imgui.set_tooltip(t("Show the selected camera frustum or light range"))
                if influence:
                    self.scene_entities.show_influence = not self.scene_entities.show_influence
                imgui.end_menu()
            if self._begin_main_menu(t("Window")):
                for panel in self.panels:
                    if not panel.enabled or panel.modal:
                        continue
                    label = t(panel.name)
                    clicked, _ = imgui.menu_item(
                        label,
                        panel.shortcut,
                        panel.open,
                    )
                    if clicked:
                        panel.toggle()
                imgui.separator()
                reset_layout, _ = imgui.menu_item(t("Reset Layout"), "", False)
                imgui.end_menu()
            if self._begin_main_menu(t("Help")):
                open_help, _ = imgui.menu_item(t("Interaction Reference"), "F1", False)
                open_documentation, _ = imgui.menu_item(t("Documentation"), "", False)
                imgui.separator()
                open_about, _ = imgui.menu_item(t("About"), "", False)
                imgui.end_menu()
            path = self.session.asset_path
            document = ""
            if path is not None:
                document = path.name + (" ●" if self.session.dirty else "")
            elif can_scene_files:
                document = t("Untitled") + (" ●" if self.session.dirty else "")
            if document:
                target_x = imgui.get_window_width() - imgui.calc_text_size(document).x - 10.0
                imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), target_x))
                imgui.text_disabled(document)
            imgui.end_main_menu_bar()
        imgui.pop_style_var()

        if new_scene:
            self._request_document_action("new_scene")
        if undo:
            self.session.submit(cmd.Undo())
        if redo:
            self.session.submit(cmd.Redo())
        if open_settings or open_recording_settings:
            self.panels.open_panel("Settings")
            if open_recording_settings:
                self.panels.get("Settings").show_category("Recording")
        if frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)
        if capture_surface is not None:
            self.request_capture(surface=capture_surface)
        if start_recording_surface is not None:
            try:
                self.start_recording(surface=start_recording_surface)
            except Exception as exc:
                self.session.report_message(
                    f"{self.localizer.text('Recording failed')}: {exc}", level="error"
                )
        if pause_recording:
            if self._viewport_recording_phase is RecordingPhase.PAUSED:
                self.resume_recording()
            else:
                self.pause_recording()
        if stop_recording:
            self.stop_recording()
        if reset_layout:
            self.reset_layout()
        if open_help:
            self.panels.open_panel("Help")
        if open_documentation:
            webbrowser.open("https://github.com/acrlw/mojive#readme")
        if open_about:
            self.panels.open_panel("Info")
        if open_scene:
            self._open_scene_dialog("open")
        if save_scene:
            if self.session.asset_path is None:
                self._open_scene_dialog("save")
            else:
                self._request_scene_save(self.session.asset_path)
        if save_scene_as:
            self._open_scene_dialog("save")
        if open_model:
            self._open_model_dialog()
        if add_model:
            self._open_model_dialog("add")
        if remove_model_id >= 0:
            self.remove_model(remove_model_id)
        if add_resource_root:
            self._open_resource_dialog()
        if remove_resource_root is not None:
            self.session.submit(cmd.RemoveResourceRoot(remove_resource_root))
        if reload_model:
            self._queue_model_load("reload", self.session.asset_path)
        if quit_viewer:
            self._request_document_action("quit")

    def _draw_entity_menu(self, shortcut: str, enabled: bool) -> None:
        t = self.localizer.text
        if not self._begin_main_menu(t("Entity"), enabled):
            return
        if imgui.begin_menu(t("Create")):
            for label, shape in (
                ("Box", MeshShape.BOX),
                ("Sphere", MeshShape.SPHERE),
                ("Cylinder", MeshShape.CYLINDER),
                ("Cone", MeshShape.CONE),
                ("Plane", MeshShape.PLANE),
            ):
                clicked, _ = imgui.menu_item(t(label), "", False)
                if clicked:
                    self._add_scene_object(shape, label.lower())
            ellipsoid, _ = imgui.menu_item(t("Ellipsoid"), "", False)
            capsule, _ = imgui.menu_item(
                t("Capsule"), "", False, self.session.adapter.caps.topology_editing
            )
            if ellipsoid:
                self._add_scene_object(
                    MeshShape.SPHERE,
                    "ellipsoid",
                    size=(0.65, 0.45, 0.35),
                )
            if capsule:
                self._add_model_primitive("capsule", "capsule")
            imgui.separator()
            point_light, _ = imgui.menu_item(t("Point Light"), "", False)
            camera, _ = imgui.menu_item(t("Camera"), "", False)
            site, _ = imgui.menu_item(
                t("Site"), "", False, self.session.adapter.caps.topology_editing
            )
            if point_light:
                self._add_scene_light()
            if camera:
                self._add_scene_camera()
            if site:
                self._add_model_site()
            imgui.end_menu()
        scene_selected = bool(self._selected_entity())
        model_selected = self._selected_model_element() is not None
        duplicate, _ = imgui.menu_item(
            t("Duplicate"), f"{shortcut}+D", False, scene_selected or model_selected
        )
        rename, _ = imgui.menu_item(t("Rename"), "F2", False, scene_selected or model_selected)
        remove, _ = imgui.menu_item(t("Delete"), "Delete", False, scene_selected or model_selected)
        if duplicate:
            self._duplicate_selected()
        if rename:
            self._request_selected_rename()
        if remove:
            self._remove_selected()
        imgui.end_menu()

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
