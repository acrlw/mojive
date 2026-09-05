"""Retained debug primitives and world-space labels."""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..gizmo import ARROW_CORNER_RADIUS_PT, AXIS_HEAD_LENGTH_PT, AXIS_SHAFT_HALF_PT
from ..log import get_logger
from ..types import MeshKey, MeshShape

log = get_logger("debugdraw")


class Occlusion(enum.StrEnum):
    """Choose how a debug layer interacts with scene depth."""

    DEPTH = "depth"

    ALWAYS = "always"

    GHOST = "ghost"


OCCLUSION_ORDER: tuple[Occlusion, ...] = (Occlusion.DEPTH, Occlusion.ALWAYS, Occlusion.GHOST)


class PrimitiveType(enum.IntEnum):
    """Internal primitive layout identifiers exposed for diagnostics and tests."""

    LINE = 0
    ARROW = 1
    POINT = 2
    FRAME = 3
    BOX = 4
    SPHERE = 5
    SECTOR = 6
    STROKE = 7
    DRAG_LINK = 8
    SOLID_ARROW = 9
    SOLID_DOUBLE_ARROW = 10
    CYLINDER = 11


VERTEX_COUNT: dict[PrimitiveType, int] = {
    PrimitiveType.LINE: 2,
    PrimitiveType.ARROW: 2,
    PrimitiveType.POINT: 1,
    PrimitiveType.FRAME: 6,
    PrimitiveType.BOX: 1,
    PrimitiveType.SPHERE: 1,
    PrimitiveType.SECTOR: 3,
    PrimitiveType.STROKE: 3,
    PrimitiveType.DRAG_LINK: 2,
    PrimitiveType.SOLID_ARROW: 1,
    PrimitiveType.SOLID_DOUBLE_ARROW: 1,
    PrimitiveType.CYLINDER: 1,
}


class DrawPath(enum.StrEnum):
    """Packed GPU stream used by a debug primitive family."""

    SEGMENT = "segment"
    ARROW = "arrow"

    POINT = "point"
    STROKE = "stroke"

    SOLID = "solid"

    SECTOR = "sector"
    DRAG_LINK = "drag_link"


PRIMITIVE_PATH: dict[PrimitiveType, DrawPath] = {
    PrimitiveType.LINE: DrawPath.SEGMENT,
    PrimitiveType.ARROW: DrawPath.ARROW,
    PrimitiveType.FRAME: DrawPath.SEGMENT,
    PrimitiveType.POINT: DrawPath.POINT,
    PrimitiveType.BOX: DrawPath.SOLID,
    PrimitiveType.SPHERE: DrawPath.SOLID,
    PrimitiveType.SECTOR: DrawPath.SECTOR,
    PrimitiveType.STROKE: DrawPath.STROKE,
    PrimitiveType.DRAG_LINK: DrawPath.DRAG_LINK,
    PrimitiveType.SOLID_ARROW: DrawPath.SOLID,
    PrimitiveType.SOLID_DOUBLE_ARROW: DrawPath.SOLID,
    PrimitiveType.CYLINDER: DrawPath.SOLID,
}

PRIMITIVE_MESH: dict[PrimitiveType, MeshKey] = {
    PrimitiveType.BOX: MeshKey(MeshShape.BOX),
    PrimitiveType.SPHERE: MeshKey(MeshShape.SPHERE),
    PrimitiveType.SOLID_ARROW: MeshKey(MeshShape.ARROW),
    PrimitiveType.SOLID_DOUBLE_ARROW: MeshKey(MeshShape.DOUBLE_ARROW),
    PrimitiveType.CYLINDER: MeshKey(MeshShape.CYLINDER),
}


RECORD_FLOATS: dict[DrawPath, int] = {
    DrawPath.SEGMENT: 13,  # a(3) b(3) rgba(4) width_px(1) unused(2)
    DrawPath.ARROW: 13,  # a(3) b(3) rgba(4) width_px(1) head_px(1) start_mask_px(1)
    DrawPath.POINT: 8,  # p(3) rgba(4) radius_px(1)
    DrawPath.SOLID: 20,
    DrawPath.SECTOR: 14,  # center(3) rotvec_end(3) ref_end(3) rgba(4) radius_px(1)
    DrawPath.STROKE: 14,  # prev/a/b(9) rgba(4) width_px(1)
    # a(3) b(3) core_rgba(4) edge_rgba(4) width/radius/edge_px(3)
    DrawPath.DRAG_LINK: 17,
}

NEVER = -1.0


AXIS_COLORS = np.array(
    [[0.90, 0.25, 0.22, 1.0], [0.35, 0.78, 0.30, 1.0], [0.30, 0.50, 0.92, 1.0]], np.float32
)


FRAME_WIDTH_PX = 2.0


ARROW_HEAD_RATIO = AXIS_HEAD_LENGTH_PT / (2.0 * AXIS_SHAFT_HALF_PT)


ARROW_CORNER_RADIUS_RATIO = ARROW_CORNER_RADIUS_PT / (2.0 * AXIS_SHAFT_HALF_PT)


DEFAULT_LIMIT = 500_000


def _grow(arr: np.ndarray, cap: int) -> np.ndarray:
    out = np.zeros((cap, *arr.shape[1:]), arr.dtype)
    out[: len(arr)] = arr
    return out


class _Store:
    __slots__ = (
        "colors",
        "count",
        "edge_colors",
        "extras",
        "outlines",
        "positions",
        "primitive_type",
        "sizes",
        "transforms",
        "verts",
    )

    def __init__(self, primitive_type: PrimitiveType) -> None:
        self.primitive_type = primitive_type
        self.verts = VERTEX_COUNT[primitive_type]
        self.count = 0
        self.positions = np.zeros((0, self.verts, 3), np.float32)
        self.colors = np.zeros((0, 4), np.float32)
        self.edge_colors = np.zeros((0, 4), np.float32)
        self.sizes = np.zeros(0, np.float32)

        self.extras = np.zeros(0, np.float32)

        self.outlines = np.zeros(0, np.float32)
        self.transforms = (
            np.zeros((0, 4, 4), np.float32)
            if primitive_type in PRIMITIVE_MESH
            else np.zeros((0, 0, 0), np.float32)
        )

    @property
    def capacity(self) -> int:
        return len(self.positions)

    def reserve(self, need: int) -> None:
        if need <= self.capacity:
            return
        cap = max(need, self.capacity * 2, 16)
        self.positions = _grow(self.positions, cap)
        self.colors = _grow(self.colors, cap)
        self.edge_colors = _grow(self.edge_colors, cap)
        self.sizes = _grow(self.sizes, cap)
        self.extras = _grow(self.extras, cap)
        self.outlines = _grow(self.outlines, cap)
        if self.transforms.ndim == 3:
            self.transforms = _grow(self.transforms, cap)

    def shift_left(self, start: int, count: int) -> int:
        n = self.count
        tail = n - (start + count)
        if tail > 0:
            src, dst = slice(start + count, n), slice(start, start + tail)
            self.positions[dst] = self.positions[src]
            self.colors[dst] = self.colors[src]
            self.edge_colors[dst] = self.edge_colors[src]
            self.sizes[dst] = self.sizes[src]
            self.extras[dst] = self.extras[src]
            self.outlines[dst] = self.outlines[src]
            if self.transforms.ndim == 3:
                self.transforms[dst] = self.transforms[src]
        self.count = n - count
        return count + tail


@dataclass
class _Entry:
    primitive_type: PrimitiveType
    start: int
    count: int
    expires: float


@dataclass
class TextLabel:
    """One retained world-space text label with a screen-space offset."""

    text: str
    anchor: np.ndarray
    color: np.ndarray
    offset_px: np.ndarray
    align: np.ndarray
    occlusion: Occlusion
    expires: float


@dataclass
class DebugStats:
    """Current storage and lifecycle counters for a :class:`DebugDraw`."""

    primitives: int = 0
    layers: int = 0
    dropped: int = 0
    vertices: int = 0
    expiring: int = 0

    moves: int = 0

    expired: int = 0


class Layer:
    """Named collection of retained debug primitives sharing one occlusion mode."""

    __slots__ = (
        "_finite",
        "_index",
        "_owner",
        "_stores",
        "_text_finite",
        "_texts",
        "name",
        "occlusion",
        "visible",
    )

    def __init__(self, name: str, occlusion: Occlusion, owner: DebugDraw) -> None:
        self.name = name
        self.occlusion = occlusion
        self.visible = True
        self._owner = owner
        self._stores: dict[PrimitiveType, _Store] = {}
        self._index: dict[str, _Entry] = {}
        self._texts: dict[str, TextLabel] = {}
        self._finite: set[str] = set()
        self._text_finite: set[str] = set()

    def _store(self, primitive_type: PrimitiveType) -> _Store:
        st = self._stores.get(primitive_type)
        if st is None:
            st = _Store(primitive_type)
            self._stores[primitive_type] = st
        return st

    def _alloc(self, primitive_type: PrimitiveType, ident: str, count: int, duration: float) -> int:
        now = self._owner.now
        expires = math.inf if duration < 0 else now + float(duration)
        if ident in self._texts:
            self._remove_text(ident)
        old = self._index.get(ident)
        if old is not None and old.primitive_type is primitive_type and old.count == count:
            old.expires = expires
            self._track(ident, expires)
            return old.start
        if old is not None:
            self._owner.moves += self._remove(ident)
        if self._owner.primitives + count > self._owner.limit:
            self._owner.drop(
                count, f"Primitive limit {self._owner.limit} exceeded; dropped {ident!r}"
            )
            return -1
        st = self._store(primitive_type)
        start = st.count
        st.reserve(start + count)
        st.count = start + count
        self._owner._primitives += count
        self._index[ident] = _Entry(primitive_type, start, count, expires)
        self._track(ident, expires)
        return start

    def _track(self, ident: str, expires: float) -> None:
        if math.isinf(expires):
            self._finite.discard(ident)
        else:
            self._finite.add(ident)

    def _remove(self, ident: str) -> int:
        entry = self._index.pop(ident, None)
        if entry is None:
            return 0
        self._finite.discard(ident)
        moved = self._stores[entry.primitive_type].shift_left(entry.start, entry.count)
        self._owner._primitives -= entry.count
        for other in self._index.values():
            if other.primitive_type is entry.primitive_type and other.start > entry.start:
                other.start -= entry.count
        return moved

    def _remove_text(self, ident: str) -> None:
        if self._texts.pop(ident, None) is not None:
            self._text_finite.discard(ident)
            self._owner._primitives -= 1

    def line(self, ident: str, a, b, color, width_px: float = 1.5, duration: float = NEVER) -> None:
        """Create or replace one screen-width world-space line segment."""
        i = self._alloc(PrimitiveType.LINE, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.LINE]
        st.positions[i, 0] = a
        st.positions[i, 1] = b
        _write_color(st.colors, i, color)
        st.sizes[i] = width_px

    def lines(
        self, ident: str, pts_a, pts_b, color, width_px: float = 1.5, duration: float = NEVER
    ) -> None:
        """Create or replace a batch of independent line segments."""
        self._many_segments(PrimitiveType.LINE, ident, pts_a, pts_b, color, width_px, duration)

    def polyline(
        self,
        ident: str,
        points,
        color,
        width_px: float = 1.5,
        *,
        closed: bool = False,
        duration: float = NEVER,
    ) -> None:
        """Create or replace a joined polyline through world-space points."""
        p = np.asarray(points, np.float32).reshape(-1, 3)
        count = len(p) if closed else len(p) - 1
        if count <= 0:
            self._remove(ident)
            return
        i = self._alloc(PrimitiveType.STROKE, ident, count, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.STROKE]
        dst = st.positions[i : i + count]
        if closed:
            dst[:, 0] = np.roll(p, 1, axis=0)
            dst[:, 1] = p
            dst[:, 2] = np.roll(p, -1, axis=0)
        else:
            dst[:, 1] = p[:-1]
            dst[:, 2] = p[1:]
            dst[:, 0] = np.concatenate((p[:1], p[:-2]), axis=0)
        colors = np.asarray(color, np.float32)
        st.colors[i : i + count] = colors[:count] if colors.ndim == 2 else _rgba(color)
        st.sizes[i : i + count] = width_px

    def arrow(
        self,
        ident: str,
        a,
        b,
        color,
        width_px: float = 2.0,
        duration: float = NEVER,
        *,
        start_mask_px: float = 0.0,
    ) -> None:
        """Create or replace one line arrow with a screen-space arrow head."""
        i = self._alloc(PrimitiveType.ARROW, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.ARROW]
        st.positions[i, 0] = a
        st.positions[i, 1] = b
        _write_color(st.colors, i, color)
        st.sizes[i] = width_px
        st.extras[i] = start_mask_px

    def arrows(
        self,
        ident: str,
        pts_a,
        pts_b,
        color,
        width_px: float = 2.0,
        duration: float = NEVER,
        *,
        start_mask_px: float = 0.0,
    ) -> None:
        """Create or replace a batch of independent line arrows."""
        self._many_segments(
            PrimitiveType.ARROW, ident, pts_a, pts_b, color, width_px, duration, start_mask_px
        )

    def point(self, ident: str, p, color, radius_px: float = 4.0, duration: float = NEVER) -> None:
        """Create or replace one world-anchored screen-space point."""
        i = self._alloc(PrimitiveType.POINT, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.POINT]
        st.positions[i, 0] = p
        _write_color(st.colors, i, color)
        st.sizes[i] = radius_px

    def drag_link(
        self,
        ident: str,
        start,
        target,
        core_color,
        edge_color,
        *,
        width_px: float = 2.0,
        radius_px: float = 6.0,
        edge_px: float = 0.75,
        duration: float = NEVER,
    ) -> None:
        """Draw a connected drag origin, line, and target with one outline."""
        i = self._alloc(PrimitiveType.DRAG_LINK, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.DRAG_LINK]
        st.positions[i, 0] = start
        st.positions[i, 1] = target
        _write_color(st.colors, i, core_color)
        _write_color(st.edge_colors, i, edge_color)
        st.sizes[i] = width_px
        st.extras[i] = radius_px
        st.outlines[i] = edge_px

    def points(
        self, ident: str, positions, color, radius_px: float = 4.0, duration: float = NEVER
    ) -> None:
        """Create or replace a batch of world-anchored points."""
        p = np.asarray(positions, np.float32).reshape(-1, 3)
        if not len(p):
            self._remove(ident)
            return
        i = self._alloc(PrimitiveType.POINT, ident, len(p), duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.POINT]
        st.positions[i : i + len(p), 0] = p
        colors = np.asarray(color, np.float32)
        st.colors[i : i + len(p)] = colors[: len(p)] if colors.ndim == 2 else _rgba(color)
        st.sizes[i : i + len(p)] = radius_px

    def _many_segments(
        self,
        primitive_type: PrimitiveType,
        ident: str,
        pts_a,
        pts_b,
        color,
        width_px: float,
        duration: float,
        extra: float = 0.0,
    ) -> None:
        a = np.asarray(pts_a, np.float32).reshape(-1, 3)
        b = np.asarray(pts_b, np.float32).reshape(-1, 3)
        n = min(len(a), len(b))
        if n == 0:
            self._remove(ident)
            return
        i = self._alloc(primitive_type, ident, n, duration)
        if i < 0:
            return
        st = self._stores[primitive_type]
        st.positions[i : i + n, 0] = a[:n]
        st.positions[i : i + n, 1] = b[:n]
        colors = np.asarray(color, np.float32)
        if colors.ndim == 2:
            st.colors[i : i + n] = colors[:n]
        else:
            st.colors[i : i + n] = _rgba(color)
        st.sizes[i : i + n] = width_px
        st.extras[i : i + n] = extra if primitive_type is PrimitiveType.ARROW else 0.0

    def frame(
        self, ident: str, transform4x4, axis_len: float = 0.1, duration: float = NEVER
    ) -> None:
        """Draw RGB axes from a row-major world transform."""
        i = self._alloc(PrimitiveType.FRAME, ident, 1, duration)
        if i < 0:
            return
        m = np.asarray(transform4x4, np.float32).reshape(4, 4)
        st = self._stores[PrimitiveType.FRAME]
        origin = m[:3, 3]
        for k in range(3):
            axis = m[:3, k]
            n = float(np.linalg.norm(axis))
            direction = axis / n if n > 1e-12 else np.zeros(3, np.float32)
            st.positions[i, 2 * k] = origin
            st.positions[i, 2 * k + 1] = origin + direction * float(axis_len)
        st.colors[i] = 1.0
        st.sizes[i] = FRAME_WIDTH_PX

    def frames(
        self,
        ident: str,
        positions,
        rotations,
        axis_len: float = 0.1,
        duration: float = NEVER,
    ) -> None:
        """Create or replace a batch of RGB coordinate frames."""
        origins = np.asarray(positions, np.float32).reshape(-1, 3)
        matrices = np.asarray(rotations, np.float32).reshape(-1, 3, 3)
        count = min(len(origins), len(matrices))
        if count == 0:
            self._remove(ident)
            return
        i = self._alloc(PrimitiveType.FRAME, ident, count, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.FRAME]
        dst = st.positions[i : i + count]
        dst[:, 0::2] = origins[:count, None, :]
        ends = dst[:, 1::2]
        ends[:] = matrices[:count].transpose(0, 2, 1)
        norms = np.linalg.norm(ends, axis=2, keepdims=True)
        np.divide(ends, norms, out=ends, where=norms > 1e-12)
        ends[norms[..., 0] <= 1e-12] = 0.0
        ends *= float(axis_len)
        ends += origins[:count, None, :]
        st.colors[i : i + count] = 1.0
        st.sizes[i : i + count] = FRAME_WIDTH_PX

    def box(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:
        """Create or replace a solid unit box transformed into world space."""
        self._solid(PrimitiveType.BOX, ident, transform4x4, color, duration)

    def sphere(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:
        """Create or replace a solid unit sphere transformed into world space."""
        self._solid(PrimitiveType.SPHERE, ident, transform4x4, color, duration)

    def cylinder(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:
        """Create or replace a solid unit cylinder transformed into world space."""
        self._solid(PrimitiveType.CYLINDER, ident, transform4x4, color, duration)

    def solid_arrow(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:
        """Create or replace a solid arrow transformed into world space."""
        self._solid(PrimitiveType.SOLID_ARROW, ident, transform4x4, color, duration)

    def solid_double_arrow(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:
        """Create or replace a solid double arrow transformed into world space."""
        self._solid(PrimitiveType.SOLID_DOUBLE_ARROW, ident, transform4x4, color, duration)

    def _solid(
        self, primitive_type: PrimitiveType, ident: str, transform4x4, color, duration: float
    ) -> None:
        i = self._alloc(primitive_type, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[primitive_type]
        m = np.asarray(transform4x4, np.float32).reshape(4, 4)
        st.transforms[i] = m
        st.positions[i, 0] = m[:3, 3]
        _write_color(st.colors, i, color)

    def sector(
        self,
        ident: str,
        center,
        rotvec_end,
        ref_end,
        color,
        duration: float = NEVER,
        *,
        radius_px: float = 0.0,
    ) -> None:
        """Draw a rotation sector from reference and rotated radius endpoints."""
        i = self._alloc(PrimitiveType.SECTOR, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[PrimitiveType.SECTOR]
        st.positions[i, 0] = center
        st.positions[i, 1] = rotvec_end
        st.positions[i, 2] = ref_end
        _write_color(st.colors, i, color)
        st.sizes[i] = radius_px

    def text(
        self,
        ident: str,
        anchor,
        text: str,
        color=(1.0, 1.0, 1.0, 1.0),
        offset_px=(0.0, 0.0),
        align=(0.0, 0.5),
        duration: float = NEVER,
    ) -> None:
        """Create or replace a world-anchored text label.

        ``offset_px`` and ``align`` operate in screen space. A negative duration
        retains the label until :meth:`erase` or :meth:`clear` is called.
        """
        if not text:
            self.erase(ident)
            return
        if ident in self._index:
            self._owner.moves += self._remove(ident)
        expires = math.inf if duration < 0 else self._owner.now + float(duration)
        label = self._texts.get(ident)
        if label is None:
            if self._owner.primitives >= self._owner.limit:
                self._owner.drop(
                    1, f"Primitive limit {self._owner.limit} exceeded; dropped {ident!r}"
                )
                return
            label = TextLabel(
                str(text),
                np.zeros(3, np.float32),
                np.ones(4, np.float32),
                np.zeros(2, np.float32),
                np.zeros(2, np.float32),
                self.occlusion,
                expires,
            )
            self._texts[ident] = label
            self._owner._primitives += 1
        label.text = str(text)
        label.anchor[:] = np.asarray(anchor, np.float32).reshape(3)
        label.color[:] = _rgba(color)
        label.offset_px[:] = np.asarray(offset_px, np.float32).reshape(2)
        label.align[:] = np.clip(np.asarray(align, np.float32).reshape(2), 0.0, 1.0)
        label.expires = expires
        if math.isinf(expires):
            self._text_finite.discard(ident)
        else:
            self._text_finite.add(ident)

    def clear(self) -> None:
        """Remove every primitive and label from this layer."""
        if not self._index and not self._texts:
            return
        for st in self._stores.values():
            self._owner._primitives -= st.count
            st.count = 0
        self._index.clear()
        self._finite.clear()
        self._owner._primitives -= len(self._texts)
        self._texts.clear()
        self._text_finite.clear()

    def erase(self, ident: str) -> None:
        """Remove one retained primitive or label by identifier."""
        self._owner.moves += self._remove(ident)
        self._remove_text(ident)

    @property
    def primitives(self) -> int:
        """Return the number of primitives and labels in this layer."""
        return sum(st.count for st in self._stores.values()) + len(self._texts)

    def count_of(self, primitive_type: PrimitiveType) -> int:
        st = self._stores.get(primitive_type)
        return st.count if st is not None else 0

    def positions_of(self, primitive_type: PrimitiveType) -> np.ndarray:
        st = self._stores.get(primitive_type)
        return (
            st.positions[: st.count]
            if st is not None
            else np.zeros((0, VERTEX_COUNT[primitive_type], 3), np.float32)
        )

    def expiry_of(self, ident: str) -> float:
        entry = self._index.get(ident)
        return entry.expires if entry is not None else math.nan


@dataclass
class Batch:
    """One packed draw range sharing occlusion, path, and optional mesh."""

    occlusion: Occlusion = Occlusion.DEPTH
    path: DrawPath = DrawPath.SEGMENT
    mesh: MeshKey | None = None
    start: int = 0
    count: int = 0


@dataclass
class PackedFrame:
    """Reusable packed debug streams consumed by renderer debug passes."""

    streams: dict[DrawPath, np.ndarray] = field(default_factory=dict)

    counts: dict[DrawPath, int] = field(default_factory=dict)
    batches: list[Batch] = field(default_factory=list)
    batch_count: int = 0
    texts: list[TextLabel] = field(default_factory=list)
    text_count: int = 0

    def stream(self, path: DrawPath) -> np.ndarray:
        return self.streams[path][: self.counts[path]]

    def active(self) -> list[Batch]:
        return self.batches[: self.batch_count]

    def active_texts(self) -> list[TextLabel]:
        return self.texts[: self.text_count]


class DebugDraw:
    """Build retained and transient debug primitives into reusable GPU batches."""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = int(limit)
        self.dropped = 0
        self.moves = 0
        self.expired = 0
        self.now = 0.0

        self._layers: dict[str, Layer] = {}
        self._by_occlusion: dict[Occlusion, list[Layer]] = {o: [] for o in OCCLUSION_ORDER}
        self._frame = PackedFrame(
            streams={p: np.zeros((0, RECORD_FLOATS[p]), np.float32) for p in DrawPath},
            counts=dict.fromkeys(DrawPath, 0),
        )
        self._need = dict.fromkeys(DrawPath, 0)
        self._primitives = 0

        self._last_drop_note = ""

    def layer(self, name: str, occlusion: Occlusion = Occlusion.DEPTH) -> Layer:
        """Return a persistent named layer, creating it on first use.

        A layer keeps the occlusion mode from its first creation. Reusing stable
        layer and primitive names updates storage in place on hot frame paths.
        """
        layer = self._layers.get(name)
        if layer is None:
            layer = Layer(name, occlusion, self)
            self._layers[name] = layer
            self._by_occlusion[occlusion].append(layer)
        elif layer.occlusion is not occlusion:
            log.warning(
                "Layer {} already uses {} occlusion; ignoring requested {} mode",
                name,
                layer.occlusion,
                occlusion,
            )
        return layer

    def layers(self) -> tuple[Layer, ...]:
        """Return all layers in creation order."""
        return tuple(self._layers.values())

    def drop(self, count: int, note: str) -> None:
        self.dropped += int(count)
        if note and note != self._last_drop_note:
            self._last_drop_note = note
            log.warning("Debug draw dropped primitives: {}", note)

    def render_frame(self, emit, now: float | None = None) -> PackedFrame:
        """Pack active primitives, pass them to ``emit``, then expire old entries."""
        self.now = time.monotonic() if now is None else float(now)
        frame = self.build()
        emit(frame)
        self.expire(self.now)
        return frame

    def expire(self, now: float | None = None) -> int:
        """Remove entries whose finite duration elapsed and return the count."""
        when = self.now if now is None else float(now)
        total = 0
        for layer in self._layers.values():
            if not layer._finite and not layer._text_finite:
                continue
            for ident in [i for i in layer._finite if layer._index[i].expires <= when]:
                self.moves += layer._remove(ident)
                total += 1
            for ident in [i for i in layer._text_finite if layer._texts[i].expires <= when]:
                layer._remove_text(ident)
                total += 1
        self.expired += total
        return total

    def clear(self) -> None:
        """Clear every debug layer while retaining allocated storage."""
        for layer in self._layers.values():
            layer.clear()

    def stats(self) -> DebugStats:
        """Measure active primitives, vertices, layers, and lifecycle counters."""
        prims = 0
        verts = 0
        expiring = 0
        for layer in self._layers.values():
            expiring += len(layer._finite) + len(layer._text_finite)
            prims += len(layer._texts)
            verts += sum(6 * len(label.text) for label in layer._texts.values())
            for primitive_type, st in layer._stores.items():
                prims += st.count
                verts += st.count * VERTEX_COUNT[primitive_type]
        return DebugStats(
            primitives=prims,
            layers=len(self._layers),
            dropped=self.dropped,
            vertices=verts,
            expiring=expiring,
            moves=self.moves,
            expired=self.expired,
        )

    @property
    def primitives(self) -> int:
        return self._primitives

    def build(self) -> PackedFrame:
        frame = self._frame
        for path in DrawPath:
            frame.counts[path] = 0
        frame.batch_count = 0
        frame.text_count = 0
        self._reserve(frame)

        for occ in OCCLUSION_ORDER:
            layers = self._by_occlusion[occ]
            if not layers:
                continue

            self._batch_solid(frame, occ, layers)
            self._batch(
                frame, occ, layers, DrawPath.SECTOR, (PrimitiveType.SECTOR,), self._pack_sector
            )
            self._batch(
                frame, occ, layers, DrawPath.STROKE, (PrimitiveType.STROKE,), self._pack_stroke
            )
            self._batch(
                frame,
                occ,
                layers,
                DrawPath.DRAG_LINK,
                (PrimitiveType.DRAG_LINK,),
                self._pack_drag_link,
            )
            self._batch(
                frame,
                occ,
                layers,
                DrawPath.SEGMENT,
                (PrimitiveType.LINE, PrimitiveType.FRAME),
                self._pack_segment,
            )
            self._batch(
                frame,
                occ,
                layers,
                DrawPath.ARROW,
                (PrimitiveType.ARROW,),
                self._pack_segment,
            )
            self._batch(
                frame, occ, layers, DrawPath.POINT, (PrimitiveType.POINT,), self._pack_point
            )
            for layer in layers:
                if not layer.visible:
                    continue
                for label in layer._texts.values():
                    if frame.text_count == len(frame.texts):
                        frame.texts.append(label)
                    else:
                        frame.texts[frame.text_count] = label
                    frame.text_count += 1
        return frame

    def _reserve(self, frame: PackedFrame) -> None:
        need = self._need
        for path in DrawPath:
            need[path] = 0
        for layer in self._layers.values():
            if not layer.visible:
                continue
            for primitive_type, st in layer._stores.items():
                if st.count == 0:
                    continue

                factor = 3 if primitive_type is PrimitiveType.FRAME else 1
                need[PRIMITIVE_PATH[primitive_type]] += st.count * factor
        for path, n in need.items():
            buf = frame.streams[path]
            if n > len(buf):
                frame.streams[path] = np.zeros(
                    (max(n, len(buf) * 2, 64), RECORD_FLOATS[path]), np.float32
                )

    def _new_batch(self, frame: PackedFrame) -> Batch:
        if frame.batch_count == len(frame.batches):
            frame.batches.append(Batch())
        b = frame.batches[frame.batch_count]
        frame.batch_count += 1
        return b

    def _batch(self, frame, occ, layers, path, primitive_types, pack) -> None:
        start = frame.counts[path]
        dst = frame.streams[path]
        at = start
        for layer in layers:
            if not layer.visible:
                continue
            for primitive_type in primitive_types:
                st = layer._stores.get(primitive_type)
                if st is not None and st.count:
                    at = pack(dst, at, st)
        if at == start:
            return
        frame.counts[path] = at
        b = self._new_batch(frame)
        b.occlusion, b.path, b.mesh, b.start, b.count = occ, path, None, start, at - start

    def _batch_solid(self, frame, occ, layers) -> None:
        for primitive_type, mesh in PRIMITIVE_MESH.items():
            start = frame.counts[DrawPath.SOLID]
            dst = frame.streams[DrawPath.SOLID]
            at = start
            for layer in layers:
                if not layer.visible:
                    continue
                st = layer._stores.get(primitive_type)
                if st is not None and st.count:
                    at = self._pack_solid(dst, at, st)
            if at == start:
                continue
            frame.counts[DrawPath.SOLID] = at
            b = self._new_batch(frame)
            b.occlusion, b.path, b.mesh, b.start, b.count = (
                occ,
                DrawPath.SOLID,
                mesh,
                start,
                at - start,
            )

    @staticmethod
    def _pack_segment(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        if st.primitive_type is PrimitiveType.FRAME:
            for k in range(3):
                s = slice(at + k * n, at + (k + 1) * n)
                dst[s, 0:3] = st.positions[:n, 2 * k]
                dst[s, 3:6] = st.positions[:n, 2 * k + 1]
                dst[s, 6:10] = AXIS_COLORS[k]
                dst[s, 10] = st.sizes[:n]
                dst[s, 11] = 0.0
                dst[s, 12] = 0.0
            return at + 3 * n
        s = slice(at, at + n)
        dst[s, 0:3] = st.positions[:n, 0]
        dst[s, 3:6] = st.positions[:n, 1]
        dst[s, 6:10] = st.colors[:n]
        dst[s, 10] = st.sizes[:n]

        dst[s, 11] = (
            st.sizes[:n] * ARROW_HEAD_RATIO if st.primitive_type is PrimitiveType.ARROW else 0.0
        )
        dst[s, 12] = st.extras[:n] if st.primitive_type is PrimitiveType.ARROW else 0.0
        return at + n

    @staticmethod
    def _pack_point(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        s = slice(at, at + n)
        dst[s, 0:3] = st.positions[:n, 0]
        dst[s, 3:7] = st.colors[:n]
        dst[s, 7] = st.sizes[:n]
        return at + n

    @staticmethod
    def _pack_stroke(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        s = slice(at, at + n)
        dst[s, 0:9] = st.positions[:n].reshape(n, 9)
        dst[s, 9:13] = st.colors[:n]
        dst[s, 13] = st.sizes[:n]
        return at + n

    @staticmethod
    def _pack_drag_link(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        s = slice(at, at + n)
        dst[s, 0:3] = st.positions[:n, 0]
        dst[s, 3:6] = st.positions[:n, 1]
        dst[s, 6:10] = st.colors[:n]
        dst[s, 10:14] = st.edge_colors[:n]
        dst[s, 14] = st.sizes[:n]
        dst[s, 15] = st.extras[:n]
        dst[s, 16] = st.outlines[:n]
        return at + n

    @staticmethod
    def _pack_solid(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        s = slice(at, at + n)

        dst[s, 0:16] = st.transforms[:n].transpose(0, 2, 1).reshape(n, 16)
        dst[s, 16:20] = st.colors[:n]
        return at + n

    @staticmethod
    def _pack_sector(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        s = slice(at, at + n)
        dst[s, 0:3] = st.positions[:n, 0]
        dst[s, 3:6] = st.positions[:n, 1]
        dst[s, 6:9] = st.positions[:n, 2]
        dst[s, 9:13] = st.colors[:n]
        dst[s, 13] = st.sizes[:n]
        return at + n


def _rgba(color) -> np.ndarray:
    c = np.asarray(color, np.float32).reshape(-1)
    return c if len(c) == 4 else np.array([c[0], c[1], c[2], 1.0], np.float32)


def _write_color(dst: np.ndarray, i: int, color) -> None:
    c = np.asarray(color, np.float32).reshape(-1)
    dst[i, : len(c)] = c[:4]
    if len(c) == 3:
        dst[i, 3] = 1.0


def world_size(size_px: float, px_scale: float, clip_w: float) -> float:
    """Convert a screen-space size to world units at one clip-space depth."""
    return float(size_px) * float(px_scale) * float(clip_w)


def sector_angle(center, rotvec_end) -> float:
    """Return the rotation angle encoded by a sector endpoint."""
    return float(
        np.linalg.norm(np.asarray(rotvec_end, np.float64) - np.asarray(center, np.float64))
    )


def sector_points(center, rotvec_end, ref_end, segments: int = 32) -> np.ndarray:
    """Tessellate a rotation sector as a world-space triangle fan."""
    c = np.asarray(center, np.float64).reshape(3)
    rotvec = np.asarray(rotvec_end, np.float64).reshape(3) - c
    ref = np.asarray(ref_end, np.float64).reshape(3) - c
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec / angle if angle > 1e-12 else np.array([0.0, 0.0, 1.0])
    out = np.zeros((segments + 2, 3), np.float64)
    out[0] = c
    for i in range(segments + 1):
        t = angle * i / segments

        v = (
            ref * math.cos(t)
            + np.cross(axis, ref) * math.sin(t)
            + axis * np.dot(axis, ref) * (1.0 - math.cos(t))
        )
        out[i + 1] = c + v
    return out
