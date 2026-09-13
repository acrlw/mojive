"""App: loading."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from mojive import commands as cmd
from mojive.scene.workspace import missing_resource_entries
from mojive.session.model_edits import MODEL_REBUILD_COMMANDS

if TYPE_CHECKING:
    from mojive.commands import CommandResult


from .support import SCENE_SUFFIXES, _ApplyModelEdits, _ModelLoadCompletion, _ModelLoadJob, log


class _Loading:
    """Private loading methods of ViewerApp; state belongs to its owner."""

    def load_model(self, path: str | Path) -> CommandResult:
        result = self.session.submit(cmd.LoadAsset(Path(path)))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(
                f"{self.localizer.text('Loaded')} {self.session.asset_path.name}"
            )
        else:
            self._report_model_error(result.message)
        return result

    def reload_model(self) -> CommandResult:
        """Reload the active source and refresh viewer-owned derived state."""

        result = self.session.submit(cmd.Reload())
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(self.localizer.text("Reloaded"))
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
        if self.session.adapter.caps.accepts_model(target):
            return self.load_model(target)
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
            self._set_model_drop_notice(
                f"{self.localizer.text('Opened')} {self.session.asset_path.name}"
            )
        else:
            self._report_model_error(result.message)
        return result

    def _queue_scene_open(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        if self.session.adapter.caps.accepts_model(target):
            self._queue_model_load("load", target)
            return
        try:
            missing = missing_resource_entries(target)
        except Exception:
            missing = ()
        if missing:
            self._begin_resource_repair(target, missing)
            return
        self._queue_model_load("open", target)

    def _queue_model_load(
        self,
        action: str,
        path: str | Path,
        position: tuple[float, float, float] | None = None,
    ) -> None:
        if self.model_edits.active:
            self._pending_edits_blocked()
            return
        target = Path(path).expanduser().resolve()
        if action == "add":
            command = cmd.AddSceneModel(target, position or tuple(self.camera.pivot))
        elif action == "open":
            command = cmd.OpenScene(target)
        elif action == "reload":
            command = cmd.Reload()
        else:
            command = cmd.LoadAsset(target)
        self._model_load_queue.append(_ModelLoadJob(action, target, command))

    def _queue_model_edit(self, command, completed=None) -> None:
        """Serialize expensive UI edits with loads; notify on the UI thread."""
        if not self.live_model_updates and isinstance(command, MODEL_REBUILD_COMMANDS):
            result = self.model_edits.stage(command)
            self.session._record_result(result)
            if completed is not None:
                completed(result)
            return
        path = self.session.asset_path or Path("Untitled")
        self._model_load_queue.append(_ModelLoadJob("edit", path, command, completed))

    def _start_model_load(self) -> bool:
        if self._model_load_completion is not None:
            return False
        if self._model_load_future is not None or not self._model_load_queue:
            return self._model_load_future is not None
        if getattr(self, "_take_video", None) is not None:
            self.stop_recording()
        if self.session.adapter.caps.simulation and not self.session.paused:
            paused = self.session.submit(cmd.Pause())
            if not paused.ok:
                job = self._model_load_queue.pop(0)
                self._model_load_queue.clear()
                self._report_model_error(
                    f"Cannot {self._model_load_verb(job.action).lower()} while physics is running: "
                    f"{paused.message}"
                )
                return False
        if self._model_load_executor is None:
            self._model_load_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mojive-model-loader",
            )
        job = self._model_load_queue.pop(0)
        self._model_load_job = job
        self._model_load_started = time.monotonic()
        log.info("{} {}", self._model_load_verb(job.action), job.path)
        self._model_prepared_resources = None
        self._model_source_prepared = 0.0
        self._model_load_future = self._model_load_executor.submit(self._load_model, job.command)
        return True

    def _load_model(self, command):
        result = (
            self.session.apply_model_edits(command.commands)
            if isinstance(command, _ApplyModelEdits)
            else self.session.submit(command)
        )
        self._model_source_prepared = time.monotonic()
        prepare = getattr(getattr(self, "backend", None), "prepare_scene", None)
        if result.ok and prepare is not None:
            self._model_prepared_resources = prepare(self.session.source)
        return result

    def _poll_model_load(self) -> bool:
        future = self._model_load_future
        job = self._model_load_job
        if future is None or job is None:
            return False
        if not future.done():
            return True
        prepared = getattr(self, "_model_source_prepared", 0.0) or time.monotonic()
        try:
            result = future.result()
        except Exception as exc:
            result = cmd.CommandResult.bad(str(exc))
        self._model_load_future = None
        self._model_load_job = None
        if result.ok:
            if job.action == "edit":
                self._sync_structure()
            else:
                self._after_model_change()
            self._model_load_completion = _ModelLoadCompletion(
                job, result.message, self._model_load_started, prepared, time.monotonic()
            )
            self._set_model_drop_notice(result.message)
        else:
            log.info(
                "{} {} after {:.3f}s", result.message, job.path, prepared - self._model_load_started
            )
            self._model_load_queue.clear()
            self._report_model_error(result.message)
        self._model_prepared_resources = None
        if job.completed is not None:
            job.completed(result)
        if self._close_after_model_load:
            self._close_after_model_load = False
            self._request_document_action("quit")
        return False

    @staticmethod
    def _model_load_verb(action: str) -> str:
        return {
            "add": "Adding model",
            "open": "Opening scene",
            "reload": "Reloading model",
            "edit": "Applying model edit",
        }.get(action, "Loading model")

    def _after_model_change(self) -> None:
        self.router.abort()
        self.gizmo.cancel()
        self.gizmo.cancel_model_placement(self.session)
        self._model_camera_id = -1
        self._model_camera_view = None
        self._camera_transition = None
        self._model_camera_projection_target = None
        self._pending_node_focus_id = None
        self._pending_joint_focus_id = None
        self._last_viewport_click = None
        self._gizmo_hint_hover.reset()
        self._structure_generation = -1
        self._sync_structure()
        self._reset_source_camera()

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
        if isinstance(selected, list | tuple):
            selected = selected[0] if selected else ""
        if not selected:
            self._after_save_action = None
            return
        if action == "save":
            pending = self._after_save_action
            self._after_save_action = None
            self._request_scene_save(selected, pending)
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
            position = tuple(float(value) for value in self.camera.pivot)
            for path in selected:
                self._queue_model_load("add", path, position)
        else:
            self._queue_model_load("load", selected[0])

    def _poll_model_drop(self) -> None:
        paths = self.window.consume_file_drops()
        if not paths:
            return
        if len(paths) == 1 and paths[0].name.endswith(SCENE_SUFFIXES):
            path = paths[0]
            if not self.session.adapter.caps.scene_files:
                self._report_model_error(
                    self.localizer.text("The current workspace cannot open Mojive scene files")
                )
                return
            self._request_document_action("open_scene", path)
            return
        unsupported = next(
            (path for path in paths if not self.session.adapter.caps.accepts_model(path)), None
        )
        if unsupported is not None:
            self._report_model_error(
                f"{self.localizer.text('Unsupported file')}: {unsupported.name}"
            )
            return
        can_add = self.session.adapter.caps.model_composition
        source = self.session.source
        has_scene_content = source is not None and source.instance_count > 0
        position = tuple(float(value) for value in self.camera.pivot)
        for path in paths:
            if has_scene_content and can_add:
                self._queue_model_load("add", path, position)
            else:
                self._queue_model_load("load", path)
            has_scene_content = True

    def _set_model_drop_notice(self, message: str) -> None:
        self._model_drop_notice = message
        self._model_drop_notice_until = time.monotonic() + 1.8
