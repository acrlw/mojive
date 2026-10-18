"""Native editing history for the shared, atomically stored specification."""

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from mojive.design.specification import (
    DEFAULT,
    SCHEMA,
    component_catalog,
    resolve_part_value,
    validate_spec,
)
from mojive.design.store import SpecificationStore, atomic_write

MISSING = object()
FAMILY_LABELS = dict(
    zip(
        SCHEMA["componentFamilies"],
        ("检查器", "时间轴", "设置", "菜单", "输出", "姿态", "场景导航", "视口", "应用框架"),
        strict=True,
    )
)


def path_value(spec, path):
    value = spec
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return deepcopy(value)


def assign(spec, path, value):
    parent = spec
    for key in path[:-1]:
        parent = parent.setdefault(key, {})
    if value is MISSING:
        parent.pop(path[-1], None)
        # Removing the final override restores shared inheritance.
        if len(path) == 3 and not parent:
            spec[path[0]].pop(path[1], None)
    else:
        parent[path[-1]] = deepcopy(value)


@dataclass
class Edit:
    path: tuple
    before: object
    after: object


class SpecificationEditor:
    """Keep editing history and external changes in one validated document."""

    def __init__(self, saved_path, live_path, reader):
        self.saved_path, self.live_path = Path(saved_path), Path(live_path)
        self.reader = reader
        self.store = SpecificationStore(saved_path, live_path)
        self.catalog = component_catalog()
        self.undo_stack, self.redo_stack = [], []
        self.error = ""
        self.last_change = ""
        self.writes = 0
        self._stamps = None
        self._last_gesture = None
        self.pending_sync = False
        self._draft_sync = None
        self._poll_at = 0.0
        self.spec = deepcopy(DEFAULT)
        self.saved_spec = deepcopy(DEFAULT)
        self.poll(force=True)

    def stamps(self):
        stamps = []
        for path in (self.saved_path, self.live_path):
            try:
                stat = path.stat()
            except FileNotFoundError:
                stamps.append(None)
            else:
                # Atomic replacement can preserve both modification time and size.
                stamps.append((stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns))
        return tuple(stamps)

    def poll(self, *, force=False):
        if not force and time.monotonic() - self._poll_at < 0.12:
            return False
        self._poll_at = time.monotonic()
        stamps = self.stamps()
        if stamps == self._stamps:
            return False
        envelope = self.store.latest()
        errors = "; ".join(f"{item['source']}: {item['error']}" for item in envelope["errors"])
        if envelope["source"] is None:
            if errors or self.spec != DEFAULT:
                self.error = "规范读取失败：" + (
                    errors or "The saved specification and draft are missing"
                )
            return False
        spec = envelope["spec"]
        if self._last_gesture is not None and path_value(
            self.spec, self._last_gesture
        ) != path_value(spec, self._last_gesture):
            self._last_gesture = None
        self.spec = spec
        if envelope["savedSpec"] is not None:
            self.saved_spec = envelope["savedSpec"]
        self._stamps = stamps
        self.error = "规范读取失败：" + errors if errors else ""
        return True

    def effective(self, part, *, family=None):
        return {
            field: resolve_part_value(self.spec, part, field, family=family, catalog=self.catalog)
            for field in self.catalog[part]["fields"]
        }

    def write(self, candidate, *, gesture=False):
        envelope = self.store.merge(
            self.spec, candidate, durable=not gesture and self._draft_sync is None
        )
        self._accept_write(envelope["spec"], gesture=gesture)

    def _accept_write(self, spec, *, gesture=False):
        self.pending_sync = gesture
        if self._draft_sync is not None and not gesture:
            self._draft_sync.dirty = True
        self.spec = spec
        # Another editor can publish after merge releases the store lock. Its
        # file stamps must never be cached alongside this earlier snapshot.
        self._stamps = None
        self.writes += 1
        self.error = ""

    def change(self, path, value, *, gesture=False):
        self.poll(force=True)
        path = tuple(path)
        before = path_value(self.spec, path)
        if before is not MISSING and value is not MISSING and before == value:
            return False
        if before is MISSING and value is MISSING:
            return False
        candidate = deepcopy(self.spec)
        assign(candidate, path, value)
        self.write(candidate, gesture=gesture)
        edit = Edit(path, before, deepcopy(value) if value is not MISSING else MISSING)
        if gesture and self._last_gesture == path and self.undo_stack:
            self.undo_stack[-1].after = edit.after
        else:
            self.undo_stack.append(edit)
        self._last_gesture = path if gesture else None
        self.redo_stack.clear()
        self.last_change = self.label(path)
        return True

    def end_gesture(self):
        if self.pending_sync:
            if self._draft_sync is None:
                self.store.sync_live()
            else:
                self._draft_sync.dirty = True
            self.pending_sync = False
            self.error = ""
        self._last_gesture = None

    def enable_background_sync(self):
        """Native workbenches flush drafts off the UI thread and close this owner."""
        from mojive.design.draft_sync import DraftSync

        if self._draft_sync is None:
            self._draft_sync = DraftSync(self.live_path)

    def poll_sync(self):
        if self._draft_sync is not None:
            previous_error = self._draft_sync.error
            self._draft_sync.poll()
            if (
                previous_error is not None
                and self._draft_sync.error is None
                and self.error == str(previous_error)
            ):
                self.error = ""

    def close(self):
        self.end_gesture()
        if self._draft_sync is not None:
            self._draft_sync.close()

    def undo(self):
        if not self.undo_stack:
            return False
        self.poll(force=True)
        edit = self.undo_stack[-1]
        # An external edit to this same field must not silently be overwritten.
        current = path_value(self.spec, edit.path)
        if current is not edit.after and current != edit.after:
            self.error = "此项已在另一工作台修改，请先核对当前值。"
            return False
        candidate = deepcopy(self.spec)
        assign(candidate, edit.path, edit.before)
        self.write(candidate)
        self.undo_stack.pop()
        self.redo_stack.append(edit)
        self.end_gesture()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.poll(force=True)
        edit = self.redo_stack[-1]
        current = path_value(self.spec, edit.path)
        if current is not edit.before and current != edit.before:
            self.error = "此项已在另一工作台修改，请先核对当前值。"
            return False
        candidate = deepcopy(self.spec)
        assign(candidate, edit.path, edit.after)
        self.write(candidate)
        self.redo_stack.pop()
        self.undo_stack.append(edit)
        self.end_gesture()
        return True

    def save(self):
        self.end_gesture()
        if self._draft_sync is not None:
            self._draft_sync.flush()
        self.poll(force=True)
        self.spec = self.store.merge(self.spec, self.spec, save=True)["spec"]
        self.saved_spec = deepcopy(self.spec)
        self.pending_sync = False
        self._stamps = None
        self.last_change = "规范已保存；下次原生启动继续读取。"

    def export(self, path):
        self.poll(force=True)
        atomic_write(Path(path).expanduser(), self.spec)
        self.last_change = "规范已导出。"

    def import_file(self, path):
        self.poll(force=True)
        value = validate_spec(self.reader(Path(path).expanduser()))
        self._accept_write(self.store.write(value)["spec"])
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.last_change = "规范已导入并应用。"

    def label(self, path):
        if path[0] == "parts":
            part = self.catalog[path[1]]
            return (
                part["label"] + " / " + part["fields"][path[2]]["label"]
                if len(path) == 3
                else part["label"]
            )
        if path[0] in ("tokens", "components"):
            return SCHEMA["tokens"][path[-1]]["label"]
        if path[0] == "layout":
            return SCHEMA["layout"][path[-1]]["label"]
        return {"g3": "G3 连续圆角", "g3_smoothing": "圆角平滑程度"}.get(path[-1], path[-1])

    def overrides(self):
        return [
            {
                "part": part,
                "field": field,
                "label": self.label(("parts", part, field)),
                "value": value,
            }
            for part, fields in self.spec.get("parts", {}).items()
            for field, value in fields.items()
        ]
