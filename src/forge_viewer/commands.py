from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import CameraView, Light


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    message: str = ""

    def __bool__(self) -> bool:
        return self.ok

    @staticmethod
    def good(message: str = "") -> CommandResult:
        return CommandResult(True, message)

    @staticmethod
    def bad(message: str) -> CommandResult:

        assert message
        return CommandResult(False, message)


class Command:
    pass


@dataclass(frozen=True)
class Pause(Command):
    pass


@dataclass(frozen=True)
class Play(Command):
    pass


@dataclass(frozen=True)
class Step(Command):
    count: int = 1


@dataclass(frozen=True)
class Reset(Command):
    pass


@dataclass(frozen=True)
class Reload(Command):
    pass


@dataclass(frozen=True)
class LoadAsset(Command):
    path: Path


@dataclass(frozen=True)
class LoadKeyframe(Command):
    keyframe_id: int


@dataclass(frozen=True)
class Select(Command):
    object_id: int


@dataclass(frozen=True)
class SetPose(Command):
    node_id: int
    position: np.ndarray
    rotation: np.ndarray  # 3×3


@dataclass(frozen=True)
class SetLight(Command):
    light_id: int
    light: Light


@dataclass(frozen=True)
class SetSceneCamera(Command):
    camera_id: int
    camera: CameraView


@dataclass(frozen=True)
class SetQpos(Command):
    index: int
    value: float


@dataclass(frozen=True)
class SetCtrl(Command):
    index: int
    value: float


@dataclass(frozen=True)
class Perturb(Command):
    node_id: int
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    target_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    mode: str = "translate"


@dataclass(frozen=True)
class ClearPerturb(Command):
    pass


@dataclass(frozen=True)
class SetCamera(Command):
    camera: CameraView


@dataclass(frozen=True)
class SetVisible(Command):
    node_id: int
    visible: bool


@dataclass(frozen=True)
class SetVisualGroup(Command):
    category: str
    group: int
    visible: bool


@dataclass(frozen=True)
class SetSpeed(Command):
    factor: float


class Query:
    pass


@dataclass(frozen=True)
class Pick(Query):
    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class NodeAt(Query):
    object_id: int


@dataclass(frozen=True)
class Bounds(Query):
    pass
