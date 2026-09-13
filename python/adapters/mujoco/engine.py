"""Optional MuJoCo dependency boundary."""

from __future__ import annotations

try:
    import mujoco
except ImportError as exc:
    mujoco = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None
