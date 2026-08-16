from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Rect = tuple[float, float, float, float]


IdReader = Callable[[int, int], int]


ScreenPicker = Callable[[float, float], int]


@dataclass(frozen=True)
class PickResult:
    object_id: int = 0
    level: int = 0

    pixel: tuple[int, int] | None = None

    ndc: tuple[float, float] | None = None

    @property
    def hit(self) -> bool:
        return self.object_id != 0


def viewport_point_to_target_pixel(
    point: tuple[float, float], rect: Rect, target_size: tuple[int, int]
) -> tuple[int, int] | None:

    rx, ry, rw, rh = rect
    tw, th = int(target_size[0]), int(target_size[1])
    if rw <= 0.0 or rh <= 0.0 or tw <= 0 or th <= 0:
        return None
    u = (float(point[0]) - rx) / rw
    v = (float(point[1]) - ry) / rh
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None

    scale_x = tw / rw
    scale_y = th / rh
    col = math.floor((float(point[0]) - rx) * scale_x)
    row_from_top = math.floor((float(point[1]) - ry) * scale_y)

    col = min(max(col, 0), tw - 1)
    row_from_top = min(max(row_from_top, 0), th - 1)
    return col, th - 1 - row_from_top


def viewport_point_to_ndc(point: tuple[float, float], rect: Rect) -> tuple[float, float]:

    rx, ry, rw, rh = rect
    nx = (float(point[0]) - rx) / max(rw, 1e-9) * 2.0 - 1.0
    ny = 1.0 - (float(point[1]) - ry) / max(rh, 1e-9) * 2.0
    return nx, ny


def pick(
    point: tuple[float, float],
    rect: Rect,
    target_size: tuple[int, int],
    read_id: IdReader,
    ray_pick: ScreenPicker | None = None,
    nearest_pick: ScreenPicker | None = None,
    root_id: int = 0,
) -> PickResult:

    px = viewport_point_to_target_pixel(point, rect, target_size)
    if px is None:
        return PickResult()
    ndc = viewport_point_to_ndc(point, rect)

    oid = int(read_id(px[0], px[1]))
    if oid != 0:
        return PickResult(_selectable(oid, root_id), 1, px, ndc)

    if ray_pick is not None:
        oid = int(ray_pick(*ndc))
        if oid != 0:
            return PickResult(_selectable(oid, root_id), 2, px, ndc)

    if nearest_pick is not None:
        oid = int(nearest_pick(*ndc))
        if oid != 0:
            return PickResult(_selectable(oid, root_id), 3, px, ndc)

    return PickResult(0, 0, px, ndc)


def _selectable(object_id: int, root_id: int) -> int:

    return 0 if object_id == root_id else object_id
