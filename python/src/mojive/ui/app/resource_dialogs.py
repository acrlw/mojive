"""App: resource dialogs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui, portable_file_dialogs

from mojive import commands as cmd
from mojive.scene.workspace import (
    MissingResource,
    relocate_workspace_resource,
    repair_workspace_resources,
)

if TYPE_CHECKING:
    from mojive.commands import CommandResult


from .support import (
    IMAGE_FILTERS,
    MESH_FILTERS,
    SCENE_SUFFIX,
    _model_filters,
    _prepare_modal,
    _scene_filters,
    _scene_save_target,
    _translated_file_filters,
)


class _ResourceDialogs:
    """Private resource dialogs methods of ViewerApp; state belongs to its owner."""

    def save_scene(
        self, path: str | Path, *, current_pose_keyframe: str | None = None
    ) -> CommandResult:
        if self.model_edits.active:
            result = cmd.CommandResult.bad(
                self.localizer.text("Apply or discard pending model edits first")
            )
            self.session._record_result(result)
            return result
        target = _scene_save_target(path)
        result = self.session.submit(cmd.SaveScene(target, current_pose_keyframe))
        if result.ok:
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def _request_scene_save(
        self,
        path: str | Path,
        pending: tuple[str, Path | None] | None = None,
    ) -> None:
        target = _scene_save_target(path)
        if target.suffix.lower() in {".xml", ".mjcf"} and self.session.current_pose_modified:
            self._pending_pose_save = (target, pending)
            return
        if self.save_scene(target).ok and pending is not None:
            self._execute_document_action(*pending)

    def _open_model_dialog(self, action: str = "open") -> None:
        if self._model_dialog is not None or not _model_filters(self.session.adapter.caps):
            return
        t = self.localizer.text
        current = self.session.asset_path
        default_path = str(current.parent if current is not None else Path.cwd())
        self._model_dialog = portable_file_dialogs.open_file(
            t("Add models") if action == "add" else t("Open a model"),
            default_path,
            _translated_file_filters(_model_filters(self.session.adapter.caps), t),
            portable_file_dialogs.opt.multiselect
            if action == "add"
            else portable_file_dialogs.opt.none,
        )
        self._model_dialog_action = action
        self._set_model_drop_notice(
            t("Choose a model to add") if action == "add" else t("Choose a model")
        )

    def _open_scene_dialog(self, action: str) -> None:
        if self._scene_dialog is not None:
            return
        t = self.localizer.text
        current = self.session.asset_path
        if action == "save":
            default = current or (Path.cwd() / f"scene{SCENE_SUFFIX}")
            self._scene_dialog = portable_file_dialogs.save_file(
                t("Save scene"),
                str(default),
                _translated_file_filters(
                    _scene_filters(self.session.adapter.caps, saving=action == "save"), t
                ),
            )
        else:
            default = current.parent if current is not None else Path.cwd()
            self._scene_dialog = portable_file_dialogs.open_file(
                t("Open Mojive scene"),
                str(default),
                _translated_file_filters(
                    _scene_filters(self.session.adapter.caps, saving=action == "save"), t
                ),
            )
        self._scene_dialog_action = action

    def _open_resource_dialog(self) -> None:
        if self._resource_dialog is not None:
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        self._resource_dialog = portable_file_dialogs.select_folder(
            self.localizer.text("Add Mojive resource directory"), str(default)
        )

    def _open_texture_dialog(
        self, model_id: int, material_index: int = -1, texture_type: str = "2d"
    ) -> None:
        if self._texture_dialog is not None:
            return
        kind = str(texture_type).strip().lower()
        if kind not in ("2d", "cube", "skybox"):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        label = "2D" if kind == "2d" else kind
        self._texture_dialog = portable_file_dialogs.open_file(
            f"{self.localizer.text('Import')} "
            f"{self.localizer.text(label)} {self.localizer.text('texture')}",
            str(default),
            _translated_file_filters(IMAGE_FILTERS, self.localizer.text),
        )
        self._texture_import_target = (int(model_id), int(material_index), kind)

    def _poll_texture_dialog(self) -> None:
        dialog = self._texture_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._texture_dialog = None
        model_id, material_index, texture_type = self._texture_import_target
        self._texture_import_target = (-1, -1, "2d")
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, list | tuple):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or "texture"
        prefix = f"opengl_{model_id}_"
        existing = {
            name.removeprefix(prefix) for name in self.session.model_texture_names(model_id)
        }
        name = base
        suffix = 2
        while name in existing:
            name = f"{base}{suffix}"
            suffix += 1
        result = self.session.submit(
            cmd.ImportModelTexture(model_id, path, name, material_index, texture_type)
        )
        if not result.ok:
            self._report_model_error(result.message)

    def _open_geometry_resource_dialog(self, node_id: int, resource_type: str) -> None:
        if self._geometry_resource_dialog is not None:
            return
        kind = str(resource_type).strip().lower()
        if kind not in ("mesh", "hfield"):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        filters = MESH_FILTERS if kind == "mesh" else IMAGE_FILTERS
        title = self.localizer.text("Import mesh" if kind == "mesh" else "Import PNG height field")
        self._geometry_resource_dialog = portable_file_dialogs.open_file(
            title,
            str(default),
            _translated_file_filters(filters, self.localizer.text),
        )
        self._geometry_resource_import_target = (int(node_id), kind)

    def _poll_geometry_resource_dialog(self) -> None:
        dialog = self._geometry_resource_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._geometry_resource_dialog = None
        node_id, resource_type = self._geometry_resource_import_target
        self._geometry_resource_import_target = (-1, "")
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, list | tuple):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or resource_type
        properties = self.session.geometry_shape_properties(node_id)
        if properties is None:
            self._report_model_error(
                self.localizer.text("The target geometry is no longer available")
            )
            return
        existing = set(
            properties.mesh_names if resource_type == "mesh" else properties.height_field_names
        )
        name = base
        suffix = 2
        while name in existing:
            name = f"{base}{suffix}"
            suffix += 1
        result = self.session.submit(
            cmd.ImportModelGeometryResource(node_id, resource_type, path, name)
        )
        if not result.ok:
            self._report_model_error(result.message)

    def _open_model_asset_import_dialog(
        self,
        model_id: int,
        asset_type: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._open_model_asset_dialog("import", model_id, asset_type, "", fields)

    def _open_model_asset_replace_dialog(self, model_id: int, asset_type: str, name: str) -> None:
        self._open_model_asset_dialog("replace", model_id, asset_type, name, ())

    def _open_model_asset_dialog(
        self,
        action: str,
        model_id: int,
        asset_type: str,
        name: str,
        fields: tuple[tuple[str, str], ...],
    ) -> None:
        if self._model_asset_dialog is not None:
            return
        kind = str(asset_type).strip().lower()
        if kind not in ("mesh", "hfield", "texture") or action not in (
            "import",
            "replace",
        ):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        filters = MESH_FILTERS if kind == "mesh" else IMAGE_FILTERS
        verb = self.localizer.text("Import" if action == "import" else "Replace")
        label = (
            "mesh" if kind == "mesh" else "PNG height field" if kind == "hfield" else "PNG texture"
        )
        self._model_asset_dialog = portable_file_dialogs.open_file(
            f"{verb} {self.localizer.text(label)}",
            str(default),
            _translated_file_filters(filters, self.localizer.text),
        )
        self._model_asset_dialog_target = (
            action,
            int(model_id),
            kind,
            str(name),
            tuple(fields),
        )

    def _poll_model_asset_dialog(self) -> None:
        dialog = self._model_asset_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._model_asset_dialog = None
        action, model_id, asset_type, name, fields = self._model_asset_dialog_target
        self._model_asset_dialog_target = ("", -1, "", "", ())
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, list | tuple):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if action == "replace":
            result = self.session.submit(
                cmd.ReplaceModelAssetFile(model_id, asset_type, name, path)
            )
        else:
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or asset_type
            existing = {
                item.name for item in self.session.model_assets(model_id) if item.type == asset_type
            }
            name = base
            suffix = 2
            while name in existing:
                name = f"{base}{suffix}"
                suffix += 1
            result = self.session.submit(
                cmd.ImportModelAsset(model_id, asset_type, path, name, fields)
            )
        if not result.ok:
            self._report_model_error(result.message)

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
                f"{self.localizer.text('Locate')} {missing.model_name}",
                str(default),
                _translated_file_filters(
                    _model_filters(self.session.adapter.caps), self.localizer.text
                ),
            )
        else:
            self._resource_repair_dialog = portable_file_dialogs.select_folder(
                self.localizer.text("Search a directory for missing resources"), str(default)
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
        if isinstance(selected, list | tuple):
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
        self._queue_scene_open(path)

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
        t = self.localizer.text
        popup_title = f"{t('Missing Resources')}###Missing Resources"
        if self._open_resource_repair_popup:
            imgui.open_popup(popup_title)
            self._open_resource_repair_popup = False
        if imgui.is_popup_open(popup_title):
            _prepare_modal(480.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(
            t(
                "This Mojive scene references model files that are no longer available. Locate "
                "files individually or search one directory to repair every unambiguous path."
            )
        )
        imgui.spacing()
        locate = -1
        for missing in self._missing_resources:
            imgui.text(f"{missing.model_name}: {missing.reference}")
            imgui.same_line()
            if imgui.small_button(f"{t('Locate...')}##missing-resource-{missing.model_index}"):
                locate = missing.model_index
        if self._resource_repair_status:
            imgui.spacing()
            imgui.text_wrapped(self._resource_repair_status)
            if imgui.small_button(f"{t('Copy details')}##resource-repair"):
                imgui.set_clipboard_text(self._resource_repair_status)
        imgui.spacing()
        search = imgui.button(t("Search Directory..."), imgui.ImVec2(160.0, 0.0))
        imgui.same_line()
        cancel = imgui.button(t("Cancel"), imgui.ImVec2(100.0, 0.0))
        if locate >= 0:
            self._open_resource_repair_dialog("locate", locate)
            imgui.close_current_popup()
        elif search:
            self._open_resource_repair_dialog("search")
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._resource_repair_path = None
            self._missing_resources = ()
            self._resource_repair_status = ""
            imgui.close_current_popup()
        imgui.end_popup()
