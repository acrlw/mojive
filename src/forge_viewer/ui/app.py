"""Main viewer UI loop and panel coordination."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs

from .. import commands as cmd
from ..adapters.base import FrameNeeds, NodeKind
from ..render.backend import FrameMode, LabelMode, RenderFlag
from ..types import Light, LightKind, MeshShape, ViewportImage
from ..workspace_io import (
    MissingResource,
    missing_resource_entries,
    relocate_workspace_resource,
    repair_workspace_resources,
)
from . import gestures as gs
from .camera import CameraOut, OrbitCamera, ndc_from_viewport, unproject
from .camera_preview import CameraPreview
from .draw2d import ImguiDraw2D
from .gizmo import ObjectGizmo
from .panels import PanelContext, PanelSet
from .perturb import PerturbController, draw_fallback
from .scene_entities import SceneEntityHelpers
from .theme import THEME, Theme
from .viewcube import ViewCube
from .window import Window, WindowConfig

if TYPE_CHECKING:
    from ..commands import CommandResult
    from ..session import Session

CLICK_SLOP_PT = 4.0


PICK_SCREEN_RADIUS_PT = 40.0

MODEL_EXTENSIONS = frozenset((".xml", ".mjcf", ".urdf"))
MODEL_FILTERS = [
    "All supported models (*.xml, *.mjcf, *.urdf)",
    "*.xml *.mjcf *.urdf",
    "MuJoCo XML / MJCF (*.xml, *.mjcf)",
    "*.xml *.mjcf",
    "URDF (*.urdf)",
    "*.urdf",
    "All files",
    "*",
]
SCENE_SUFFIX = ".forge.json"
SCENE_FILTERS = ["Forge scenes", "*.forge.json", "All files", "*"]


@dataclass
class Keys:
    fly: tuple[float, float, float] = (0.0, 0.0, 0.0)
    toggle_pause: bool = False
    frame_scene: bool = False
    gizmo_translate: bool = False
    gizmo_rotate: bool = False
    gizmo_space: bool = False
    gizmo_axis: int = -1


class ViewerApp:
    def __init__(
        self,
        session: Session,
        backend: Any,
        window: Window | None = None,
        *,
        title: str = "forge-viewer",
        theme: Theme | None = None,
        debug_bridge: Any | None = None,
    ) -> None:
        self.session = session
        self.backend = backend
        self.window = window
        self.title = title
        self.theme = theme or THEME
        self.debug_bridge = debug_bridge
        self.camera = OrbitCamera()

        self.camera_out = CameraOut(backend=backend, session=session)
        self.camera.attach(self.camera_out)
        self.camera_preview = CameraPreview()
        self.gizmo = ObjectGizmo()
        self.view_cube = ViewCube()
        self.perturb = PerturbController()
        self.scene_entities = SceneEntityHelpers()
        self.router = gs.GestureRouter()
        self.panels = PanelSet()
        self._started = False
        self._frame_index = 0
        self._last_time = time.perf_counter()
        self._viewport_rect = (0.0, 0.0, 640.0, 480.0)
        self._viewport_image: ViewportImage | None = None
        self._dt = 0.0
        self._structure_generation = -1
        self._state = gs.InputState()
        self._model_camera_id = -1
        self._model_camera_view = None
        self._fixed_render_size: tuple[int, int] | None = None
        self._model_dialog: Any | None = None
        self._model_dialog_action = ""
        self._scene_dialog: Any | None = None
        self._scene_dialog_action = ""
        self._resource_dialog: Any | None = None
        self._resource_repair_dialog: Any | None = None
        self._resource_repair_dialog_action = ""
        self._resource_repair_model_index = -1
        self._resource_repair_path: Path | None = None
        self._missing_resources: tuple[MissingResource, ...] = ()
        self._resource_repair_status = ""
        self._open_resource_repair_popup = False
        self._pending_document_action: tuple[str, Path | None] | None = None
        self._after_save_action: tuple[str, Path | None] | None = None
        self._rename_object_id = 0
        self._rename_value = ""
        self._open_rename_popup = False
        self._window_title = ""
        self._closing_without_save = False
        self._model_load_error = ""
        self._show_model_load_error = False
        self._model_drop_notice = ""
        self._model_drop_notice_until = 0.0
        self._display_scale_generation = -1

    def set_fixed_render_size(self, width: int, height: int) -> None:
        self._fixed_render_size = (max(1, int(width)), max(1, int(height)))
        self.backend.resize(*self._fixed_render_size)
        self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])

    def _startup(self) -> None:
        if self._started:
            return
        if self.window is None:
            self.window = Window(WindowConfig(title=self.title))
        self._sync_structure()
        self._frame_scene(animate=False)
        self.window.show()
        self._started = True
        self._last_time = time.perf_counter()

    def run(self, max_frames: int | None = None) -> None:
        self._startup()
        while not self._should_close():
            if max_frames is not None and self._frame_index >= max_frames:
                break
            self.frame()
        self.release()

    def sync(self) -> None:
        self._startup()
        self.frame()

    def _should_close(self) -> bool:
        closing = bool(self.window.should_close())
        if closing and self._closing_without_save:
            return True
        if closing and self.session.dirty and self._pending_document_action is None:
            self.window.cancel_close()
            self._pending_document_action = ("quit", None)
            return False
        return closing

    def release(self) -> None:
        if self._model_dialog is not None:
            self._model_dialog.kill()
            self._model_dialog = None
        if self._scene_dialog is not None:
            self._scene_dialog.kill()
            self._scene_dialog = None
        if self._resource_dialog is not None:
            self._resource_dialog.kill()
            self._resource_dialog = None
        if self._resource_repair_dialog is not None:
            self._resource_repair_dialog.kill()
            self._resource_repair_dialog = None
        if self.debug_bridge is not None:
            self.debug_bridge.close()
        self.camera_preview.release()
        self.backend.release()
        self.session.release()

    def load_model(self, path: str | Path) -> CommandResult:
        result = self.session.submit(cmd.LoadAsset(Path(path)))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(f"Loaded {self.session.asset_path.name}")
        else:
            self._report_model_error(result.message)
        return result

    def add_model(
        self, path: str | Path, position: tuple[float, float, float] | None = None
    ) -> CommandResult:
        location = position or tuple(float(value) for value in self.camera.pivot)
        result = self.session.submit(cmd.AddSceneModel(Path(path), location))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def remove_model(self, model_id: int) -> CommandResult:
        result = self.session.submit(cmd.RemoveSceneModel(model_id))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def open_scene(self, path: str | Path) -> CommandResult:
        target = Path(path).expanduser().resolve()
        try:
            missing = missing_resource_entries(target)
        except Exception:
            missing = ()
        if missing:
            self._begin_resource_repair(target, missing)
            return cmd.CommandResult.bad(
                f"{len(missing)} workspace resource(s) must be located before opening"
            )
        result = self.session.submit(cmd.OpenScene(target))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(f"Opened {self.session.asset_path.name}")
        else:
            self._report_model_error(result.message)
        return result

    def save_scene(self, path: str | Path) -> CommandResult:
        target = Path(path).expanduser()
        if not target.name.endswith(SCENE_SUFFIX):
            target = target.with_name(target.name + SCENE_SUFFIX)
        result = self.session.submit(cmd.SaveScene(target))
        if result.ok:
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def _after_model_change(self) -> None:
        self.router.abort()
        self.gizmo.cancel()
        self._model_camera_id = -1
        self._model_camera_view = None
        self._structure_generation = -1
        self._sync_structure()
        self._frame_scene(animate=False)

    def _open_model_dialog(self, action: str = "open") -> None:
        if self._model_dialog is not None:
            return
        current = self.session.asset_path
        default_path = str(current.parent if current is not None else Path.cwd())
        self._model_dialog = portable_file_dialogs.open_file(
            "Add MJCF or URDF models" if action == "add" else "Open an MJCF or URDF model",
            default_path,
            MODEL_FILTERS,
            portable_file_dialogs.opt.multiselect
            if action == "add"
            else portable_file_dialogs.opt.none,
        )
        self._model_dialog_action = action
        self._set_model_drop_notice(
            "Choose a model to add" if action == "add" else "Choose an MJCF or URDF model"
        )

    def _open_scene_dialog(self, action: str) -> None:
        if self._scene_dialog is not None:
            return
        current = self.session.asset_path
        if action == "save":
            default = current or (Path.cwd() / f"scene{SCENE_SUFFIX}")
            self._scene_dialog = portable_file_dialogs.save_file(
                "Save Forge scene", str(default), SCENE_FILTERS
            )
        else:
            default = current.parent if current is not None else Path.cwd()
            self._scene_dialog = portable_file_dialogs.open_file(
                "Open Forge scene", str(default), SCENE_FILTERS
            )
        self._scene_dialog_action = action

    def _open_resource_dialog(self) -> None:
        if self._resource_dialog is not None:
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        self._resource_dialog = portable_file_dialogs.select_folder(
            "Add Forge resource directory", str(default)
        )

    def _begin_resource_repair(self, path: Path, missing: tuple[MissingResource, ...]) -> None:
        self._resource_repair_path = path
        self._missing_resources = missing
        self._resource_repair_status = ""
        self._open_resource_repair_popup = True

    def _open_resource_repair_dialog(self, action: str, model_index: int = -1) -> None:
        if self._resource_repair_dialog is not None or self._resource_repair_path is None:
            return
        default = self._resource_repair_path.parent
        if action == "locate":
            missing = next(
                (item for item in self._missing_resources if item.model_index == model_index), None
            )
            if missing is None:
                return
            self._resource_repair_dialog = portable_file_dialogs.open_file(
                f"Locate {missing.model_name}", str(default), MODEL_FILTERS
            )
        else:
            self._resource_repair_dialog = portable_file_dialogs.select_folder(
                "Search a directory for missing resources", str(default)
            )
        self._resource_repair_dialog_action = action
        self._resource_repair_model_index = model_index

    def _poll_resource_repair_dialog(self) -> None:
        dialog = self._resource_repair_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._resource_repair_dialog_action
        model_index = self._resource_repair_model_index
        self._resource_repair_dialog = None
        self._resource_repair_dialog_action = ""
        self._resource_repair_model_index = -1
        try:
            selected = dialog.result()
        except Exception as exc:
            self._resource_repair_status = str(exc)
            self._open_resource_repair_popup = True
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            self._open_resource_repair_popup = True
            return
        path = self._resource_repair_path
        if path is None:
            return
        try:
            if action == "locate":
                repair = relocate_workspace_resource(path, model_index, selected)
            else:
                repair = repair_workspace_resources(path, selected)
        except Exception as exc:
            self._resource_repair_status = str(exc)
            self._open_resource_repair_popup = True
            return
        self._missing_resources = repair.missing
        if repair.missing:
            self._resource_repair_status = (
                f"Repaired {repair.repaired}; {len(repair.missing)} resource(s) still missing."
            )
            self._open_resource_repair_popup = True
            return
        self._resource_repair_path = None
        self._resource_repair_status = ""
        self._set_model_drop_notice(f"Repaired {repair.repaired} resource path(s)")
        self.open_scene(path)

    def _poll_resource_dialog(self) -> None:
        dialog = self._resource_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._resource_dialog = None
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if selected:
            result = self.session.submit(cmd.AddResourceRoot(Path(selected)))
            if not result.ok:
                self._report_model_error(result.message)

    def _draw_resource_repair(self) -> None:
        if self._open_resource_repair_popup:
            imgui.open_popup("Missing Resources")
            self._open_resource_repair_popup = False
        visible, _ = imgui.begin_popup_modal(
            "Missing Resources", None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(
            "This Forge scene references model files that are no longer available. "
            "Locate files individually or search one directory to repair every unambiguous path."
        )
        imgui.spacing()
        locate = -1
        for missing in self._missing_resources:
            imgui.text(f"{missing.model_name}: {missing.reference}")
            imgui.same_line()
            if imgui.small_button(f"Locate...##missing-resource-{missing.model_index}"):
                locate = missing.model_index
        if self._resource_repair_status:
            imgui.spacing()
            imgui.text_wrapped(self._resource_repair_status)
        imgui.spacing()
        search = imgui.button("Search Directory...", imgui.ImVec2(160.0, 0.0))
        imgui.same_line()
        cancel = imgui.button("Cancel", imgui.ImVec2(100.0, 0.0))
        if locate >= 0:
            self._open_resource_repair_dialog("locate", locate)
            imgui.close_current_popup()
        elif search:
            self._open_resource_repair_dialog("search")
            imgui.close_current_popup()
        elif cancel:
            self._resource_repair_path = None
            self._missing_resources = ()
            self._resource_repair_status = ""
            imgui.close_current_popup()
        imgui.end_popup()

    def _poll_scene_dialog(self) -> None:
        dialog = self._scene_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._scene_dialog_action
        self._scene_dialog = None
        self._scene_dialog_action = ""
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            self._after_save_action = None
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            self._after_save_action = None
            return
        if action == "save":
            pending = self._after_save_action
            self._after_save_action = None
            if self.save_scene(selected).ok and pending is not None:
                self._execute_document_action(*pending)
        else:
            self._request_document_action("open_scene", Path(selected))

    def _poll_model_dialog(self) -> None:
        dialog = self._model_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._model_dialog_action
        self._model_dialog = None
        self._model_dialog_action = ""
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if not selected:
            return
        if action == "add":
            for path in selected:
                if not self.add_model(path).ok:
                    break
        else:
            self.load_model(selected[0])

    def _poll_model_drop(self) -> None:
        paths = self.window.consume_file_drops()
        if not paths:
            return
        if len(paths) == 1 and paths[0].name.endswith(SCENE_SUFFIX):
            path = paths[0]
            if not self.session.adapter.caps.scene_files:
                self._report_model_error("The current workspace cannot open Forge scene files")
                return
            self._request_document_action("open_scene", path)
            return
        unsupported = next(
            (path for path in paths if path.suffix.lower() not in MODEL_EXTENSIONS), None
        )
        if unsupported is not None:
            self._report_model_error(f"Unsupported file: {unsupported.name}")
            return
        can_add = self.session.adapter.caps.model_composition
        for path in paths:
            source = self.session.source
            has_scene_content = source is not None and source.instance_count > 0
            if has_scene_content and can_add:
                result = self.add_model(path)
            else:
                result = self.load_model(path)
            if not result.ok:
                break

    def _set_model_drop_notice(self, message: str) -> None:
        self._model_drop_notice = message
        self._model_drop_notice_until = time.monotonic() + 1.8

    def _draw_main_menu(self) -> None:
        caps = self.session.adapter.caps
        can_load = bool(caps.asset_loading)
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
        quit_viewer = False
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if can_scene_files:
                    new_scene, _ = imgui.menu_item("New Scene", f"{shortcut}+N", False)
                    open_scene, _ = imgui.menu_item(
                        "Open Scene...", f"{shortcut}+O", False, self._scene_dialog is None
                    )
                    save_scene, _ = imgui.menu_item(
                        "Save", f"{shortcut}+S", False, self.session.dirty
                    )
                    save_scene_as, _ = imgui.menu_item("Save As...", f"{shortcut}+Shift+S", False)
                if can_load:
                    if can_scene_files:
                        imgui.separator()
                    open_model, _ = imgui.menu_item(
                        "Open Model (MJCF / URDF)...",
                        f"{shortcut}+O" if not can_scene_files else "",
                        False,
                        self._model_dialog is None,
                    )
                    if caps.model_composition:
                        add_model, _ = imgui.menu_item(
                            "Add Models (MJCF / URDF)...",
                            "",
                            False,
                            self._model_dialog is None,
                        )
                        removable = [item for item in self.session.scene_models if item.removable]
                        if imgui.begin_menu("Remove Model", bool(removable)):
                            for item in removable:
                                clicked, _ = imgui.menu_item(item.name, "", False)
                                if clicked:
                                    remove_model_id = item.model_id
                            imgui.end_menu()
                    reload_model, _ = imgui.menu_item(
                        "Reload Model",
                        f"{shortcut}+Shift+O",
                        False,
                        self.session.asset_path is not None,
                    )
                if can_scene_files and imgui.begin_menu("Resource Directories"):
                    add_resource_root, _ = imgui.menu_item(
                        "Add Directory...", "", False, self._resource_dialog is None
                    )
                    for root in self.session.adapter.resource_roots:
                        clicked, _ = imgui.menu_item(f"Remove {root}", "", False)
                        if clicked:
                            remove_resource_root = root
                    imgui.end_menu()
                imgui.separator()
                quit_viewer, _ = imgui.menu_item("Quit", f"{shortcut}+Q", False, True)
                imgui.end_menu()
            if imgui.begin_menu("Edit", caps.edit_history):
                undo, _ = imgui.menu_item("Undo", f"{shortcut}+Z", False, self.session.can_undo)
                redo, _ = imgui.menu_item(
                    "Redo", f"{shortcut}+Shift+Z", False, self.session.can_redo
                )
                imgui.end_menu()
            self._draw_entity_menu(shortcut, can_edit)
            path = self.session.asset_path
            if path is not None:
                imgui.text_disabled(path.name + (" *" if self.session.dirty else ""))
            elif can_scene_files:
                imgui.text_disabled("Untitled" + (" *" if self.session.dirty else ""))
            imgui.end_main_menu_bar()

        io = imgui.get_io()
        modifier = bool(io.key_ctrl or io.key_super)
        keyboard_free = not io.want_text_input and not imgui.is_any_item_active()
        if modifier and keyboard_free:
            if caps.edit_history:
                undo |= imgui.is_key_pressed(imgui.Key.z, False) and not io.key_shift
                redo |= imgui.is_key_pressed(imgui.Key.z, False) and bool(io.key_shift)
            if can_scene_files:
                new_scene |= imgui.is_key_pressed(imgui.Key.n, False)
                open_scene |= imgui.is_key_pressed(imgui.Key.o, False) and not io.key_shift
                save_scene |= imgui.is_key_pressed(imgui.Key.s, False) and not io.key_shift
                save_scene_as |= imgui.is_key_pressed(imgui.Key.s, False) and bool(io.key_shift)
            elif can_load:
                open_model |= imgui.is_key_pressed(imgui.Key.o, False) and not io.key_shift
            if can_load:
                reload_model |= imgui.is_key_pressed(imgui.Key.o, False) and bool(io.key_shift)
            if can_edit and self._selected_entity() and imgui.is_key_pressed(imgui.Key.d, False):
                self._duplicate_selected()
        quit_viewer |= modifier and imgui.is_key_pressed(imgui.Key.q, False)
        if can_edit and keyboard_free and self._selected_entity():
            if imgui.is_key_pressed(imgui.Key.delete, False) or imgui.is_key_pressed(
                imgui.Key.backspace, False
            ):
                self._remove_selected()
            if imgui.is_key_pressed(imgui.Key.f2, False):
                self.request_rename(self.session.selected)

        if new_scene:
            self._request_document_action("new_scene")
        if undo:
            self.session.submit(cmd.Undo())
        if redo:
            self.session.submit(cmd.Redo())
        if open_scene:
            self._open_scene_dialog("open")
        if save_scene:
            if self.session.asset_path is None:
                self._open_scene_dialog("save")
            else:
                self.save_scene(self.session.asset_path)
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
            result = self.session.submit(cmd.Reload())
            if result.ok:
                self._after_model_change()
                self._set_model_drop_notice(f"Reloaded {self.session.asset_path.name}")
            else:
                self._report_model_error(result.message)
        if quit_viewer:
            self._request_document_action("quit")

    def _draw_entity_menu(self, shortcut: str, enabled: bool) -> None:
        if not imgui.begin_menu("Entity", enabled):
            return
        if imgui.begin_menu("Create"):
            for label, shape in (
                ("Box", MeshShape.BOX),
                ("Sphere", MeshShape.SPHERE),
                ("Cylinder", MeshShape.CYLINDER),
                ("Cone", MeshShape.CONE),
                ("Plane", MeshShape.PLANE),
            ):
                clicked, _ = imgui.menu_item(label, "", False)
                if clicked:
                    self._add_scene_object(shape, label.lower())
            imgui.separator()
            point_light, _ = imgui.menu_item("Point Light", "", False)
            camera, _ = imgui.menu_item("Camera", "", False)
            if point_light:
                self._add_scene_light()
            if camera:
                self._add_scene_camera()
            imgui.end_menu()
        selected = bool(self._selected_entity())
        duplicate, _ = imgui.menu_item("Duplicate", f"{shortcut}+D", False, selected)
        rename, _ = imgui.menu_item("Rename", "F2", False, selected)
        remove, _ = imgui.menu_item("Delete", "Delete", False, selected)
        if duplicate:
            self._duplicate_selected()
        if rename:
            self.request_rename(self.session.selected)
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

    def _add_scene_object(self, shape: MeshShape, base_name: str) -> None:
        position = tuple(float(value) for value in self._camera_view().target)
        size = (4.0, 4.0, 0.02) if shape is MeshShape.PLANE else (0.5, 0.5, 0.5)
        result = self.session.submit(
            cmd.AddSceneObject(shape, self._entity_name(base_name), size=size, position=position)
        )
        if result.ok:
            self.session.submit(cmd.Select(result.entity_id))

    def _add_scene_light(self) -> None:
        view = self._camera_view()
        name = self._entity_name("point light")
        result = self.session.submit(
            cmd.AddSceneLight(
                name,
                Light(kind=LightKind.POINT, position=np.asarray(view.eye, np.float32).copy()),
            )
        )
        if result.ok:
            node = next(
                (
                    node
                    for node in reversed(self.session.nodes)
                    if node.kind is NodeKind.LIGHT and node.name == name
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
                    if node.kind is NodeKind.CAMERA and node.name == name
                ),
                None,
            )
            if node is not None:
                self.session.submit(cmd.Select(node.object_id))

    def _duplicate_selected(self) -> None:
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.DuplicateSceneEntity(object_id))

    def _remove_selected(self) -> None:
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.RemoveSceneEntity(object_id))

    def _selected_entity(self) -> int:
        node = self.session.selected_node
        if (
            node is None
            or node.model_id >= 0
            or node.kind not in (NodeKind.LINK, NodeKind.LIGHT, NodeKind.CAMERA)
        ):
            return 0
        return int(node.object_id)

    def request_rename(self, object_id: int) -> None:
        node = self.session.node_by_object_id(object_id)
        if (
            node is None
            or node.model_id >= 0
            or node.kind not in (NodeKind.LINK, NodeKind.LIGHT, NodeKind.CAMERA)
        ):
            return
        self._rename_object_id = int(object_id)
        self._rename_value = node.name
        self._open_rename_popup = True

    def _report_model_error(self, message: str) -> None:
        self._model_load_error = message
        self._show_model_load_error = True

    def _draw_model_load_error(self) -> None:
        if self._show_model_load_error:
            imgui.open_popup("File operation failed")
            self._show_model_load_error = False
        visible, _ = imgui.begin_popup_modal(
            "File operation failed", None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(self._model_load_error)
        imgui.spacing()
        if imgui.button("OK", imgui.ImVec2(100.0, 0.0)):
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
                self._set_model_drop_notice("New Forge scene")
            else:
                self._report_model_error(result.message)
        elif action == "open_scene" and path is not None:
            self.open_scene(path)
        elif action == "quit":
            self._closing_without_save = True
            self.window.request_close()

    def _draw_unsaved_changes(self) -> None:
        pending = self._pending_document_action
        if pending is None:
            return
        imgui.open_popup("Unsaved changes")
        visible, _ = imgui.begin_popup_modal(
            "Unsaved changes", None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        name = self.session.asset_path.name if self.session.asset_path is not None else "Untitled"
        imgui.text(f"Save changes to {name}?")
        imgui.spacing()
        if imgui.button("Save", imgui.ImVec2(100.0, 0.0)):
            if self.session.asset_path is None:
                self._after_save_action = pending
                self._pending_document_action = None
                self._open_scene_dialog("save")
            elif self.save_scene(self.session.asset_path).ok:
                self._pending_document_action = None
                self._execute_document_action(*pending)
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Discard", imgui.ImVec2(100.0, 0.0)):
            self._pending_document_action = None
            self._execute_document_action(*pending)
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100.0, 0.0)):
            self._pending_document_action = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_rename_popup(self) -> None:
        if self._open_rename_popup:
            imgui.open_popup("Rename Entity")
            self._open_rename_popup = False
        visible, _ = imgui.begin_popup_modal(
            "Rename Entity", None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.set_next_item_width(320.0 * self.window.style_scale)
        submitted, self._rename_value = imgui.input_text(
            "##entity_name",
            self._rename_value,
            imgui.InputTextFlags_.enter_returns_true.value,
        )
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here(-1)
        accept = submitted or imgui.button("Rename", imgui.ImVec2(100.0, 0.0))
        imgui.same_line()
        cancel = imgui.button("Cancel", imgui.ImVec2(100.0, 0.0)) or imgui.is_key_pressed(
            imgui.Key.escape, False
        )
        if accept and self._rename_value.strip():
            self.session.submit(
                cmd.RenameSceneEntity(self._rename_object_id, self._rename_value.strip())
            )
            imgui.close_current_popup()
        elif cancel:
            imgui.close_current_popup()
        imgui.end_popup()

    def _sync_window_title(self) -> None:
        path = self.session.asset_path
        document = path.name if path is not None else "Untitled"
        title = f"{document}{' *' if self.session.dirty else ''} — {self.title}"
        if title != self._window_title:
            self.window.set_title(title)
            self._window_title = title

    def frame(self) -> None:
        window = self.window
        now = time.perf_counter()
        dt = self._dt = min(0.1, now - self._last_time)
        self._last_time = now

        window.begin_frame()
        self._sync_display_scale()
        self._poll_model_dialog()
        self._poll_scene_dialog()
        self._poll_resource_dialog()
        self._poll_resource_repair_dialog()
        self._poll_model_drop()
        self._draw_main_menu()
        window.begin_dockspace()
        keys = self._poll_keys()
        self.apply_keys(keys)

        state = self._state = self._input_state()

        self._claim_gesture(state)

        self._poll_gizmo(state, keys)
        self._poll_camera(state, keys, dt)
        self._poll_perturb(state)
        self._poll_pick(state)
        self._advance_camera(dt)

        frame = self.session.tick(self.frame_needs(), wall_dt=dt)
        self._sync_structure()
        self._sync_model_camera()
        self.backend.update(frame)

        self.backend.highlight(self.session.selected)
        self._sync_viewport_size()

        if self.debug_bridge is not None:
            self.debug_bridge.pump()
            if frame.debug_commands:
                self.debug_bridge.apply_batch(frame.debug_commands)

        self._publish_perturb_marks()
        self.scene_entities.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect[3],
            self.window.ui_scale,
            self._model_camera_id >= 0,
        )
        self._publish_gizmo()

        self._viewport_image = self.backend.render(frame)
        preview_name, preview_camera = self.camera_preview.selected_camera(self.session)
        preview_width = min(
            1024, max(320, int(self.window.points_to_pixels(340.0 * self.window.style_scale)))
        )
        self.camera_preview.update(
            self.backend,
            self.session.source,
            self.session.structure_generation,
            frame,
            preview_camera,
            (preview_width, max(1, preview_width * 9 // 16)),
        )

        ctx = self._panel_context()
        self._draw_viewport(ctx, preview_name)
        self.panels.draw(ctx)
        self._draw_rename_popup()
        self._draw_unsaved_changes()
        self._draw_resource_repair()
        self._draw_model_load_error()
        self._sync_window_title()
        window.end_frame()
        self._frame_index += 1

    def _poll_keys(self) -> Keys:
        k = imgui.Key
        io = imgui.get_io()
        self.panels.poll_shortcuts()

        if io.want_text_input:
            return Keys()
        if io.key_ctrl or io.key_super:
            return Keys()

        def down(key) -> float:
            return 1.0 if imgui.is_key_down(key) else 0.0

        axis = next((i for i, key in enumerate((k.x, k.y, k.z)) if imgui.is_key_down(key)), -1)

        return Keys(
            fly=(
                down(k.w) - down(k.s),
                down(k.d) - down(k.a),
                down(k.q) - down(k.e),
            ),
            toggle_pause=imgui.is_key_pressed(k.space, False),
            frame_scene=imgui.is_key_pressed(k.f, False),
            gizmo_translate=imgui.is_key_pressed(k.g, False),
            gizmo_rotate=imgui.is_key_pressed(k.r, False),
            gizmo_space=imgui.is_key_pressed(k.t, False),
            gizmo_axis=axis,
        )

    def _input_state(self) -> gs.InputState:
        io = imgui.get_io()
        cursor = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        rect = self._viewport_rect
        inside = (
            rect[0] <= cursor[0] <= rect[0] + rect[2] and rect[1] <= cursor[1] <= rect[1] + rect[3]
        )
        hovered_window = imgui.get_current_context().hovered_window
        hovered_name = None if hovered_window is None else str(hovered_window.name)
        over_viewport = gs.viewport_input_allowed(inside, hovered_name)
        view = self._camera_view()
        hovered_ball = self.view_cube.update(view, rect, cursor, self.window.style_scale)
        self.gizmo.update_hover(
            self.session,
            view,
            rect,
            cursor,
            enabled=over_viewport and not self._viewing_selected_camera(),
            style_scale=self.window.style_scale,
        )
        node = self.session.selected_node
        return gs.InputState(
            left=imgui.is_mouse_down(0),
            right=imgui.is_mouse_down(1),
            middle=imgui.is_mouse_down(2),
            ctrl=io.key_ctrl,
            shift=io.key_shift,
            alt=io.key_alt,
            wheel=float(io.mouse_wheel),
            cursor=cursor,
            delta=(float(io.mouse_delta.x), float(io.mouse_delta.y)),
            over_viewport=over_viewport,
            over_view_cube=over_viewport and hovered_ball is not None,
            gizmo_available=(self.gizmo.style == "2d" or self.backend.caps.gizmo)
            and self.gizmo.last_verdict.ok,
            gizmo_hovered=over_viewport and self.gizmo.hovered,
            has_selection=node is not None,
            perturbing=self.session.perturb.active,
            ui_wants_mouse=io.want_capture_mouse and not over_viewport,
        )

    def _claim_gesture(self, state: gs.InputState) -> gs.Claim:
        return self.router.update(state)

    def _poll_gizmo(self, state: gs.InputState, keys: Keys) -> None:
        if self._viewing_selected_camera():
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                -1,
                style_scale=self.window.style_scale,
            )
            self.gizmo.interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                claimed=False,
                left_down=state.left,
                released=self.router.released,
                style_scale=self.window.style_scale,
            )
            return
        keyboard_was_active = self.gizmo.keyboard_using
        axis = keys.gizmo_axis
        if not keyboard_was_active and (not state.over_viewport or state.any_button):
            axis = -1
        if keyboard_was_active or axis >= 0:
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                axis,
                snap=state.shift,
                style_scale=self.window.style_scale,
            )
            return
        self.gizmo.interact(
            self.session,
            self._camera_view(),
            self._viewport_rect,
            state.cursor,
            claimed=self.router.wants_gizmo(),
            left_down=state.left,
            released=self.router.released,
            snap=state.shift,
            style_scale=self.window.style_scale,
        )

    def _publish_gizmo(self) -> None:
        self.gizmo.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
            yielding=gs.gizmo_yields(self._state) or self._viewing_selected_camera(),
            interactive=self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO),
        )

    def _poll_camera(self, state: gs.InputState, keys: Keys, dt: float) -> None:
        fwd, right, up = keys.fly
        if fwd or right or up:
            self._leave_model_camera()
            self.camera.fly(dt, forward=fwd, right=right, up=up)
        if keys.frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)

        if self.router.wants_view_cube():
            ball = self.view_cube.hovered

            if self.router.travel >= CLICK_SLOP_PT and state.delta != (0.0, 0.0):
                self._leave_model_camera()
                self.view_cube.drag(self.camera, *state.delta)
            elif ball is not None and self.router.released and self.router.travel < CLICK_SLOP_PT:
                self._leave_model_camera()
                self.view_cube.click(self.camera, ball, self.camera_out)
            return

        if not self.router.wants_camera():
            return
        gesture = gs.camera_gesture(state)

        settled = self.router.travel >= CLICK_SLOP_PT
        if gesture is gs.CameraGesture.ORBIT and settled:
            self._leave_model_camera()
            self.camera.orbit(*state.delta)
        elif gesture is gs.CameraGesture.PAN and settled:
            self._leave_model_camera()
            self.camera.pan(state.delta[0], state.delta[1], self._viewport_rect[3])
        elif gesture is gs.CameraGesture.DOLLY:
            self._leave_model_camera()
            self.camera.dolly(state.wheel)

    def _advance_camera(self, dt: float) -> None:
        if self._model_camera_id >= 0:
            return
        self.camera.advance(dt, self.camera_out)

    def _camera_view(self):
        return self._model_camera_view or self.camera.view()

    def select_model_camera(self, camera_id: int) -> None:
        i = int(camera_id)
        if i >= 0 and not any(c.camera_id == i for c in self.session.cameras):
            return
        if i < 0:
            self._leave_model_camera(publish=True)
            return
        self._model_camera_id = i

    def _viewing_selected_camera(self) -> bool:
        node = self.session.selected_node
        if node is None or node.kind is not NodeKind.CAMERA:
            return False
        index = int(node.camera_index)
        if not 0 <= index < len(self.session.cameras):
            return False
        return int(self.session.cameras[index].camera_id) == self._model_camera_id

    def _sync_model_camera(self) -> None:
        if self._model_camera_id < 0:
            return
        view = self.session.camera_view(self._model_camera_id)
        if view is None:
            self._leave_model_camera(publish=True)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        view = view.with_aspect(aspect)
        self._model_camera_view = view
        self.backend.set_camera(view)
        self.session.submit(cmd.SetCamera(view))

    def _leave_model_camera(self, *, publish: bool = False) -> None:
        if self._model_camera_id < 0:
            return
        if self._model_camera_view is not None:
            self.camera.adopt(self._model_camera_view)
        self._model_camera_id = -1
        self._model_camera_view = None
        if publish:
            self.camera.publish(self.camera_out)

    def _frame_scene(self, *, animate: bool = True) -> None:
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.frame_scene(self.session.bounds(), self.camera_out, animate=animate)

    def _poll_perturb(self, state: gs.InputState) -> None:
        st = self.session.perturb
        if not self.router.wants_perturb():
            if st.active:
                self.perturb.end(self.session)
            return

        node = self.session.selected_node
        if node is None:
            return
        cam = self._camera_view()
        if not st.active:
            pos, _ = self._node_pose(node)
            self.perturb.begin(
                self.session, cam, node, pos, self.router.mode, body_radius=self._body_radius(node)
            )
        if st.mode == "translate":
            origin, direction = self._cursor_ray(state.cursor)
            self.perturb.drag_translate(self.session, cam, origin, direction)
        else:
            self.perturb.drag_rotate(self.session, cam, state.delta[0], state.delta[1])
        self.perturb.apply(self.session)

    def _publish_perturb_marks(self) -> None:
        self.perturb.publish_marks(
            self.backend,
            self.session,
            self._camera_view(),
            rect=self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
        )

    def _poll_pick(self, state: gs.InputState) -> None:
        if not self.router.wants_camera():
            return
        if not self.router.released or self.router.travel > CLICK_SLOP_PT:
            return
        if not state.over_viewport:
            return
        object_id = self._pick_at(state.cursor)
        self.session.submit(cmd.Select(object_id))

    def _pick_at(self, cursor: tuple[float, float]) -> int:
        rect = self._viewport_rect

        helper = self.scene_entities.pick(
            self.session,
            self._camera_view(),
            rect,
            cursor,
            self.window.style_scale,
            self._model_camera_id >= 0,
        )
        if self._selectable(helper):
            return helper

        img = self._viewport_image
        if self.backend.caps.gpu_pick and img is not None:
            hit = img.pixel_from_viewport_point(cursor, rect)
            if hit is not None:
                object_id = int(self.backend.pick(*hit))
                if self._selectable(object_id):
                    return object_id

        if self.session.adapter.caps.raycast:
            origin, direction = self._cursor_ray(cursor)
            object_id, _dist = self.session.query(cmd.Pick(origin=origin, direction=direction))
            if self._selectable(int(object_id)):
                return int(object_id)

        return self._nearest_link(cursor)

    def _selectable(self, object_id: int) -> bool:
        if object_id <= 0:
            return False
        node = self.session.node_by_object_id(object_id)
        if node is None:
            return False
        return node.kind is not NodeKind.WORLD and node.parent >= 0

    def _nearest_link(self, cursor: tuple[float, float]) -> int:
        frame = self.session.frame
        if frame.body_xpos is None or len(frame.body_xpos) == 0:
            return 0
        cam = self._camera_view()
        mvp = cam.proj_matrix() @ cam.view_matrix()
        pts = np.asarray(frame.body_xpos, np.float64)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ mvp.T
        w = np.where(np.abs(h[:, 3]) < 1e-9, 1e-9, h[:, 3])
        rect = self._viewport_rect
        sx = rect[0] + (h[:, 0] / w * 0.5 + 0.5) * rect[2]
        sy = rect[1] + (0.5 - h[:, 1] / w * 0.5) * rect[3]
        d2 = (sx - cursor[0]) ** 2 + (sy - cursor[1]) ** 2
        d2[w <= 0.0] = np.inf
        best_body = int(np.argmin(d2))
        limit = (PICK_SCREEN_RADIUS_PT * self.window.style_scale) ** 2
        if not np.isfinite(d2[best_body]) or d2[best_body] > limit:
            return 0
        for node in self.session.nodes:
            if node.body_index == best_body and self._selectable(node.object_id):
                return int(node.object_id)
        return 0

    def _draw_viewport(self, ctx: PanelContext, preview_name: str = "") -> None:
        imgui.begin("Viewport", None, imgui.WindowFlags_.no_scrollbar.value)
        pos = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        self._viewport_rect = (
            float(pos.x),
            float(pos.y),
            max(float(size.x), 1.0),
            max(float(size.y), 1.0),
        )
        image = self._viewport_image
        if image is None:
            imgui.text_disabled("No viewport image is available")
        else:
            uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
            uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
            imgui.image(self.window.viewport_texture_ref(image), size, uv0, uv1)
        x, y, w, h = self._viewport_rect
        imgui.push_clip_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), True)
        try:
            overlay = ImguiDraw2D()
            st = self.session.perturb
            if st.active and not self.backend.caps.debug_draw:
                node = self.session.node(st.node_id)
                center = self._node_pose(node)[0] if node is not None else st.target_pos
                draw_fallback(
                    self._camera_view(),
                    st,
                    self._viewport_rect,
                    (imgui.get_io().mouse_pos.x, imgui.get_io().mouse_pos.y),
                    center,
                    overlay,
                    self.window.style_scale,
                )
            self.gizmo.draw_overlay(
                self._camera_view(),
                self._viewport_rect,
                overlay,
                style_scale=self.window.style_scale,
            )
            self.view_cube.draw(overlay, self.window.style_scale)
            self._draw_model_drop_overlay(overlay)
        finally:
            imgui.pop_clip_rect()
        self.camera_preview.draw(self.window, self._viewport_rect, preview_name)
        imgui.end()

    def _draw_model_drop_overlay(self, overlay: ImguiDraw2D) -> None:
        source = self.session.source
        empty = source is not None and source.instance_count == 0
        notice = self._model_drop_notice if time.monotonic() < self._model_drop_notice_until else ""
        caps = self.session.adapter.caps
        dragging = self.window.file_drag_active and (caps.asset_loading or caps.scene_files)
        if not empty and not notice and not dragging:
            return
        empty_hint = (
            "Drop a .forge.json scene here\nFile > Open Scene...  ·  Entity > Create"
            if caps.scene_files
            else "Drop an MJCF or URDF model here\nFile > Open Model...  ·  Add Model..."
        )
        if dragging:
            message = (
                "Release to add model(s)"
                if caps.model_composition and self.session.scene_models
                else "Release to open model"
            )
        else:
            message = notice or empty_hint
        lines = message.splitlines()
        sizes = [overlay.text_size(line) for line in lines]
        scale = self.window.style_scale
        pad_x, pad_y = 18.0 * scale, 12.0 * scale
        width = max(size[0] for size in sizes) + 2.0 * pad_x
        height = sum(size[1] for size in sizes) + 2.0 * pad_y + (len(lines) - 1) * 3.0 * scale
        x, y, w, h = self._viewport_rect
        left = x + (w - width) * 0.5
        top = y + (h - height) * 0.5
        if dragging:
            overlay.rect(
                (x + 3.0 * scale, y + 3.0 * scale),
                (x + w - 3.0 * scale, y + h - 3.0 * scale),
                (0.95, 0.68, 0.24, 0.95),
                2.0 * scale,
                rounding=8.0 * scale,
            )
        overlay.rect_filled(
            (left, top),
            (left + width, top + height),
            (0.08, 0.09, 0.11, 0.88),
            rounding=7.0 * scale,
        )
        cursor_y = top + pad_y
        for line, size in zip(lines, sizes, strict=True):
            overlay.text(
                (left + (width - size[0]) * 0.5, cursor_y),
                (0.93, 0.94, 0.95, 1.0),
                line,
            )
            cursor_y += size[1] + 3.0 * scale

    def frame_needs(self) -> FrameNeeds:
        needs = FrameNeeds(poses=True).merge(self.panels.frame_needs())
        label_mode = self.backend.get_label_mode()
        frame_mode = self.backend.get_frame_mode()
        needs.contacts = (
            self.backend.get_flag(RenderFlag.CONTACTPOINT)
            or self.backend.get_flag(RenderFlag.CONTACTFORCE)
            or label_mode in (LabelMode.CONTACT_POINT, LabelMode.CONTACT_FORCE)
            or frame_mode is FrameMode.CONTACT
        )
        needs.tendons = (
            self.backend.get_flag(RenderFlag.TENDON)
            or self.backend.get_flag(RenderFlag.ACTUATOR)
            or label_mode is LabelMode.TENDON
        )
        needs.actuator = (
            self.backend.get_flag(RenderFlag.ACTUATOR) or label_mode is LabelMode.ACTUATOR
        )
        needs.deformables = bool(
            (self.session.source and self.session.source.dynamic_meshes)
            or self.backend.get_flag(RenderFlag.FLEXVERT)
            or self.backend.get_flag(RenderFlag.FLEXEDGE)
            or label_mode is LabelMode.FLEX
        )
        needs.islands = self.backend.get_flag(RenderFlag.ISLAND)
        needs.bvh = self.backend.get_flag(RenderFlag.BODYBVH) or self.backend.get_flag(
            RenderFlag.MESHBVH
        )
        needs.diagnostics = (
            needs.bvh
            or any(
                self.backend.get_flag(flag)
                for flag in (
                    RenderFlag.ACTUATOR,
                    RenderFlag.JOINT,
                    RenderFlag.COM,
                    RenderFlag.INERTIA,
                    RenderFlag.CAMERA,
                    RenderFlag.LIGHT,
                    RenderFlag.RANGEFINDER,
                    RenderFlag.CONSTRAINT,
                    RenderFlag.AUTOCONNECT,
                )
            )
            or label_mode
            in (
                LabelMode.JOINT,
                LabelMode.ACTUATOR,
                LabelMode.CONSTRAINT,
                LabelMode.CAMERA,
                LabelMode.LIGHT,
            )
            or frame_mode in (FrameMode.CAMERA, FrameMode.LIGHT)
        )
        return needs

    def _sync_structure(self) -> None:
        gen = self.session.structure_generation
        if gen != self._structure_generation:
            self._structure_generation = gen
            self.backend.set_scene(self.session.source)

    def _sync_viewport_size(self) -> None:
        if self._fixed_render_size is not None:
            self.backend.resize(*self._fixed_render_size)
            self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])
            return
        settled = self.window.poll_render_size(self._viewport_rect[2:])
        if settled is None:
            return
        sw, sh = settled
        self.backend.resize(sw, sh)
        self.camera.set_aspect(max(sw, 1) / max(sh, 1))

    def _sync_display_scale(self) -> None:
        generation = self.window.scale_generation
        if generation == self._display_scale_generation:
            return
        configure_text = getattr(self.backend, "configure_text", None)
        if configure_text is not None:
            font = self.window.font_report
            configure_text(
                font.mono_path,
                font.mono_index,
                font.cjk_path,
                font.cjk_index,
                self.window.config.font_size_pt * self.window.ui_scale,
            )
        self._display_scale_generation = generation

    def _panel_context(self) -> PanelContext:
        return PanelContext(
            session=self.session,
            backend=self.backend,
            camera=self.camera,
            model_camera_id=self._model_camera_id,
            model_camera_view=self._model_camera_view,
            select_model_camera=self.select_model_camera,
            request_rename=self.request_rename,
            gizmo=self.gizmo,
            perturb=self.perturb,
            scene_entities=self.scene_entities,
            theme=self.theme,
            style_scale=self.window.style_scale,
            viewport_rect=self._viewport_rect,
            dt=self._dt,
            status=self.session.last_message,
        )

    def _cursor_ray(self, cursor: tuple[float, float]):
        ndc = ndc_from_viewport(cursor[0], cursor[1], self._viewport_rect)
        return unproject(self._camera_view(), *ndc)

    def _node_pose(self, node) -> tuple[np.ndarray, np.ndarray]:
        from .perturb import current_pose

        return current_pose(self.session, node)

    def _body_radius(self, node) -> float:
        src = self.session.source
        if src is None or len(src.geom_size) == 0:
            return 0.1
        sizes = src.geom_size[np.asarray(src.geom_body) == node.body_index]
        return float(np.max(sizes)) if len(sizes) else 0.1

    def apply_keys(self, keys: Keys) -> None:
        if keys.toggle_pause:
            self.session.submit(cmd.Play() if self.session.paused else cmd.Pause())
        if keys.gizmo_translate:
            self.gizmo.set_mode("translate")
        if keys.gizmo_rotate:
            self.gizmo.set_mode("rotate")
        if keys.gizmo_space:
            self.gizmo.toggle_space()
