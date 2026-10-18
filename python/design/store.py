"""Atomically share design drafts between native editors and running workspaces."""

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from mojive.design.specification import DEFAULT, validate_spec

_LOCK = threading.RLock()
_SURFACE_TOKENS = ("floating_color", "popup_color")


def _validate_loaded_spec(value):
    """Read the known v1 shape from before independent surface colors were added."""
    if (
        isinstance(value, dict)
        and type(value.get("version")) is int
        and value["version"] == 1
        and isinstance(value.get("tokens"), dict)
        and set(value["tokens"]) == set(DEFAULT["tokens"]).difference(_SURFACE_TOKENS)
    ):
        value = {
            **value,
            "tokens": {
                **value["tokens"],
                **{name: DEFAULT["tokens"][name] for name in _SURFACE_TOKENS},
            },
        }
    return validate_spec(value)


def read_spec(path):
    return _validate_loaded_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def default_spec_path():
    return (
        Path(os.environ.get("MOJIVE_DESIGN_SPEC", "output/design-workbench/current-spec.json"))
        .expanduser()
        .resolve()
    )


def atomic_write(path, spec, *, durable=True):
    spec = validate_spec(deepcopy(spec))
    data = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + "-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        else:
            import msvcrt

            stream.write(b"\0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class SpecificationConflict(ValueError):
    pass


def changes(before, after, path=()):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in before.keys() | after.keys():
            if key not in after:
                if isinstance(before[key], dict):
                    result.extend(changes(before[key], {}, (*path, key)))
                else:
                    result.append(((*path, key), None, True))
            elif key not in before:
                if isinstance(after[key], dict) and after[key]:
                    result.extend(changes({}, after[key], (*path, key)))
                else:
                    result.append(((*path, key), deepcopy(after[key]), False))
            else:
                result.extend(changes(before[key], after[key], (*path, key)))
        return result
    return [(path, deepcopy(after), False)]


class SpecificationStore:
    def __init__(self, saved_path=None, live_path=None):
        self.saved_path = Path(saved_path or default_spec_path()).resolve()
        draft_name = (
            "live-spec.json"
            if self.saved_path.name == "current-spec.json"
            else self.saved_path.stem + ".live.json"
        )
        self.live_path = Path(live_path or self.saved_path.with_name(draft_name)).resolve()
        self.lock_path = self.saved_path.with_name(".specification.lock")

    def latest(self):
        candidates, errors = [], []
        for order, path in enumerate((self.saved_path, self.live_path)):
            try:
                with path.open(encoding="utf-8") as stream:
                    stat = os.fstat(stream.fileno())
                    value = _validate_loaded_spec(json.load(stream))
                candidates.append(
                    (stat.st_mtime_ns, order, path.name, value, stat.st_ino, stat.st_ctime_ns)
                )
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append({"source": path.name, "error": str(error)})
        if candidates:
            stamp, _, name, value, inode, changed = max(candidates, key=lambda item: item[:2])
        else:
            stamp, name, value, inode, changed = 0, None, deepcopy(DEFAULT), 0, 0
        return {
            "spec": value,
            "savedSpec": next((item[3] for item in candidates if item[1] == 0), None),
            "source": name,
            # Rapid atomic replacements can share a filesystem modification time.
            # Include the replaced file identity so a new draft cannot be skipped.
            "revision": f"{name}:{stamp}:{changed}:{inode}",
            "mtimeNs": str(stamp),
            "modifiedAt": stamp / 1_000_000,
            "errors": errors,
        }

    def write(self, spec, *, save=False, expected_revision=None):
        spec = validate_spec(deepcopy(spec))
        with file_lock(self.lock_path):
            latest = self.latest()
            if expected_revision is not None and expected_revision != latest["revision"]:
                raise SpecificationConflict("规范已在另一工作台更新，请读取最新值后合并。")
            atomic_write(self.saved_path if save else self.live_path, spec)
            return self.latest()

    def sync_live(self):
        """Finish durability once a continuous native editing gesture ends."""
        with file_lock(self.lock_path), self.live_path.open("rb") as stream:
            os.fsync(stream.fileno())

    def merge(self, before, after, *, save=False, durable=True):
        delta = changes(before, after)
        absent = object()
        with file_lock(self.lock_path):
            latest = self.latest()
            if latest["source"] is None and (latest["errors"] or before != DEFAULT):
                errors = (
                    "; ".join(f"{item['source']}: {item['error']}" for item in latest["errors"])
                    or "The saved specification and draft are missing"
                )
                raise SpecificationConflict("No valid design specification is available: " + errors)
            result = deepcopy(latest["spec"])
            for path, value, remove in delta:
                base, remote, target = before, latest["spec"], result
                parents = []
                for key in path[:-1]:
                    base = base.get(key, {}) if isinstance(base, dict) else {}
                    remote = remote.get(key, {}) if isinstance(remote, dict) else {}
                    parents.append((target, key))
                    target = target.setdefault(key, {})
                key = path[-1]
                old = base.get(key, absent)
                current = remote.get(key, absent)
                if current != old and not (remove and current is absent) and current != value:
                    raise SpecificationConflict("此项已在另一工作台修改，请核对当前值。")
                if remove:
                    target.pop(key, None)
                    for parent, container in reversed(parents[1:]):
                        if parent[container]:
                            break
                        del parent[container]
                else:
                    target[key] = value
            atomic_write(self.saved_path if save else self.live_path, result, durable=durable)
            return self.latest()
