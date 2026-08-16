from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..gizmo import AXIS_HEAD_LENGTH_PT, AXIS_SHAFT_HALF_PT
from ..log import get_logger
from ..types import MeshKey, MeshShape

log = get_logger("debugdraw")


class Occlusion(enum.StrEnum):
    DEPTH = "depth"

    ALWAYS = "always"

    GHOST = "ghost"


OCCLUSION_ORDER: tuple[Occlusion, ...] = (Occlusion.DEPTH, Occlusion.ALWAYS, Occlusion.GHOST)


class Prim(enum.IntEnum):
    LINE = 0
    ARROW = 1
    POINT = 2
    FRAME = 3
    BOX = 4
    SPHERE = 5
    SECTOR = 6
    STROKE = 7
    DRAG_LINK = 8


VERTEX_COUNT: dict[Prim, int] = {
    Prim.LINE: 2,
    Prim.ARROW: 2,
    Prim.POINT: 1,
    Prim.FRAME: 6,
    Prim.BOX: 1,
    Prim.SPHERE: 1,
    Prim.SECTOR: 3,
    Prim.STROKE: 3,
    Prim.DRAG_LINK: 2,
}


class Path(enum.StrEnum):
    SEGMENT = "segment"

    POINT = "point"
    STROKE = "stroke"

    SOLID = "solid"

    SECTOR = "sector"
    DRAG_LINK = "drag_link"


PRIM_PATH: dict[Prim, Path] = {
    Prim.LINE: Path.SEGMENT,
    Prim.ARROW: Path.SEGMENT,
    Prim.FRAME: Path.SEGMENT,
    Prim.POINT: Path.POINT,
    Prim.BOX: Path.SOLID,
    Prim.SPHERE: Path.SOLID,
    Prim.SECTOR: Path.SECTOR,
    Prim.STROKE: Path.STROKE,
    Prim.DRAG_LINK: Path.DRAG_LINK,
}

PRIM_MESH: dict[Prim, MeshKey] = {
    Prim.BOX: MeshKey(MeshShape.BOX),
    Prim.SPHERE: MeshKey(MeshShape.SPHERE),
}


RECORD_FLOATS: dict[Path, int] = {
    Path.SEGMENT: 13,  # a(3) b(3) rgba(4) width_px(1) head_px(1) start_mask_px(1)
    Path.POINT: 8,  # p(3) rgba(4) radius_px(1)
    Path.SOLID: 20,
    Path.SECTOR: 14,  # center(3) rotvec_end(3) ref_end(3) rgba(4) radius_px(1)
    Path.STROKE: 14,  # prev/a/b(9) rgba(4) width_px(1)
    # a(3) b(3) core_rgba(4) edge_rgba(4) width/radius/edge_px(3)
    Path.DRAG_LINK: 17,
}

NEVER = -1.0


AXIS_COLORS = np.array(
    [[0.90, 0.25, 0.22, 1.0], [0.35, 0.78, 0.30, 1.0], [0.30, 0.50, 0.92, 1.0]], np.float32
)


FRAME_WIDTH_PX = 2.0


ARROW_HEAD_RATIO = AXIS_HEAD_LENGTH_PT / (2.0 * AXIS_SHAFT_HALF_PT)


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
        "kind",
        "outlines",
        "positions",
        "sizes",
        "transforms",
        "verts",
    )

    def __init__(self, kind: Prim) -> None:
        self.kind = kind
        self.verts = VERTEX_COUNT[kind]
        self.count = 0
        self.positions = np.zeros((0, self.verts, 3), np.float32)
        self.colors = np.zeros((0, 4), np.float32)
        self.edge_colors = np.zeros((0, 4), np.float32)
        self.sizes = np.zeros(0, np.float32)

        self.extras = np.zeros(0, np.float32)

        self.outlines = np.zeros(0, np.float32)
        self.transforms = (
            np.zeros((0, 4, 4), np.float32)
            if kind in PRIM_MESH
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
    kind: Prim
    start: int
    count: int
    expires: float


@dataclass
class TextLabel:
    text: str
    anchor: np.ndarray
    color: np.ndarray
    offset_px: np.ndarray
    align: np.ndarray
    occlusion: Occlusion
    expires: float


@dataclass
class DebugStats:
    primitives: int = 0
    layers: int = 0
    dropped: int = 0
    vertices: int = 0
    expiring: int = 0

    moves: int = 0

    expired: int = 0


class Layer:
    __slots__ = (
        "_finite",
        "_index",
        "_owner",
        "_stores",
        "_text_finite",
        "_texts",
        "name",
        "occlusion",
    )

    def __init__(self, name: str, occlusion: Occlusion, owner: DebugDraw) -> None:
        self.name = name
        self.occlusion = occlusion
        self._owner = owner
        self._stores: dict[Prim, _Store] = {}
        self._index: dict[str, _Entry] = {}
        self._texts: dict[str, TextLabel] = {}
        self._finite: set[str] = set()
        self._text_finite: set[str] = set()

    def _store(self, kind: Prim) -> _Store:
        st = self._stores.get(kind)
        if st is None:
            st = _Store(kind)
            self._stores[kind] = st
        return st

    def _alloc(self, kind: Prim, ident: str, count: int, duration: float) -> int:

        now = self._owner.now
        expires = math.inf if duration < 0 else now + float(duration)
        if ident in self._texts:
            self._remove_text(ident)
        old = self._index.get(ident)
        if old is not None and old.kind is kind and old.count == count:
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
        st = self._store(kind)
        start = st.count
        st.reserve(start + count)
        st.count = start + count
        self._owner._primitives += count
        self._index[ident] = _Entry(kind, start, count, expires)
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
        moved = self._stores[entry.kind].shift_left(entry.start, entry.count)
        self._owner._primitives -= entry.count
        for other in self._index.values():
            if other.kind is entry.kind and other.start > entry.start:
                other.start -= entry.count
        return moved

    def _remove_text(self, ident: str) -> None:
        if self._texts.pop(ident, None) is not None:
            self._text_finite.discard(ident)
            self._owner._primitives -= 1

    def line(self, ident: str, a, b, color, width_px: float = 1.5, duration: float = NEVER) -> None:

        i = self._alloc(Prim.LINE, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[Prim.LINE]
        st.positions[i, 0] = a
        st.positions[i, 1] = b
        _write_color(st.colors, i, color)
        st.sizes[i] = width_px

    def lines(
        self, ident: str, pts_a, pts_b, color, width_px: float = 1.5, duration: float = NEVER
    ) -> None:

        self._many_segments(Prim.LINE, ident, pts_a, pts_b, color, width_px, duration)

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

        p = np.asarray(points, np.float32).reshape(-1, 3)
        count = len(p) if closed else len(p) - 1
        if count <= 0:
            self._remove(ident)
            return
        i = self._alloc(Prim.STROKE, ident, count, duration)
        if i < 0:
            return
        st = self._stores[Prim.STROKE]
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

        i = self._alloc(Prim.ARROW, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[Prim.ARROW]
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

        self._many_segments(
            Prim.ARROW, ident, pts_a, pts_b, color, width_px, duration, start_mask_px
        )

    def point(self, ident: str, p, color, radius_px: float = 4.0, duration: float = NEVER) -> None:

        i = self._alloc(Prim.POINT, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[Prim.POINT]
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

        i = self._alloc(Prim.DRAG_LINK, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[Prim.DRAG_LINK]
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

        p = np.asarray(positions, np.float32).reshape(-1, 3)
        if not len(p):
            self._remove(ident)
            return
        i = self._alloc(Prim.POINT, ident, len(p), duration)
        if i < 0:
            return
        st = self._stores[Prim.POINT]
        st.positions[i : i + len(p), 0] = p
        colors = np.asarray(color, np.float32)
        st.colors[i : i + len(p)] = colors[: len(p)] if colors.ndim == 2 else _rgba(color)
        st.sizes[i : i + len(p)] = radius_px

    def _many_segments(
        self,
        kind: Prim,
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
        i = self._alloc(kind, ident, n, duration)
        if i < 0:
            return
        st = self._stores[kind]
        st.positions[i : i + n, 0] = a[:n]
        st.positions[i : i + n, 1] = b[:n]
        colors = np.asarray(color, np.float32)
        if colors.ndim == 2:
            st.colors[i : i + n] = colors[:n]
        else:
            st.colors[i : i + n] = _rgba(color)
        st.sizes[i : i + n] = width_px
        st.extras[i : i + n] = extra if kind is Prim.ARROW else 0.0

    def frame(
        self, ident: str, transform4x4, axis_len: float = 0.1, duration: float = NEVER
    ) -> None:

        i = self._alloc(Prim.FRAME, ident, 1, duration)
        if i < 0:
            return
        m = np.asarray(transform4x4, np.float32).reshape(4, 4)
        st = self._stores[Prim.FRAME]
        origin = m[:3, 3]
        for k in range(3):
            axis = m[:3, k]
            n = float(np.linalg.norm(axis))
            direction = axis / n if n > 1e-12 else np.zeros(3, np.float32)
            st.positions[i, 2 * k] = origin
            st.positions[i, 2 * k + 1] = origin + direction * float(axis_len)
        st.colors[i] = 1.0
        st.sizes[i] = FRAME_WIDTH_PX

    def box(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:

        self._solid(Prim.BOX, ident, transform4x4, color, duration)

    def sphere(self, ident: str, transform4x4, color, duration: float = NEVER) -> None:

        self._solid(Prim.SPHERE, ident, transform4x4, color, duration)

    def _solid(self, kind: Prim, ident: str, transform4x4, color, duration: float) -> None:
        i = self._alloc(kind, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[kind]
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

        i = self._alloc(Prim.SECTOR, ident, 1, duration)
        if i < 0:
            return
        st = self._stores[Prim.SECTOR]
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
        for st in self._stores.values():
            self._owner._primitives -= st.count
            st.count = 0
        self._index.clear()
        self._finite.clear()
        self._owner._primitives -= len(self._texts)
        self._texts.clear()
        self._text_finite.clear()

    def erase(self, ident: str) -> None:

        self._owner.moves += self._remove(ident)
        self._remove_text(ident)

    @property
    def primitives(self) -> int:
        return sum(st.count for st in self._stores.values()) + len(self._texts)

    def count_of(self, kind: Prim) -> int:
        st = self._stores.get(kind)
        return st.count if st is not None else 0

    def positions_of(self, kind: Prim) -> np.ndarray:

        st = self._stores.get(kind)
        return (
            st.positions[: st.count]
            if st is not None
            else np.zeros((0, VERTEX_COUNT[kind], 3), np.float32)
        )

    def expiry_of(self, ident: str) -> float:
        entry = self._index.get(ident)
        return entry.expires if entry is not None else math.nan


@dataclass
class Batch:
    occlusion: Occlusion = Occlusion.DEPTH
    path: Path = Path.SEGMENT
    mesh: MeshKey | None = None
    start: int = 0
    count: int = 0


@dataclass
class PackedFrame:
    streams: dict[Path, np.ndarray] = field(default_factory=dict)

    counts: dict[Path, int] = field(default_factory=dict)
    batches: list[Batch] = field(default_factory=list)
    batch_count: int = 0
    texts: list[TextLabel] = field(default_factory=list)
    text_count: int = 0

    def stream(self, path: Path) -> np.ndarray:
        return self.streams[path][: self.counts[path]]

    def active(self) -> list[Batch]:
        return self.batches[: self.batch_count]

    def active_texts(self) -> list[TextLabel]:
        return self.texts[: self.text_count]


class DebugDraw:
    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = int(limit)
        self.dropped = 0
        self.moves = 0
        self.expired = 0
        self.now = 0.0

        self._layers: dict[str, Layer] = {}
        self._by_occlusion: dict[Occlusion, list[Layer]] = {o: [] for o in OCCLUSION_ORDER}
        self._frame = PackedFrame(
            streams={p: np.zeros((0, RECORD_FLOATS[p]), np.float32) for p in Path},
            counts=dict.fromkeys(Path, 0),
        )
        self._need = dict.fromkeys(Path, 0)
        self._primitives = 0

        self._last_drop_note = ""

    def layer(self, name: str, occlusion: Occlusion = Occlusion.DEPTH) -> Layer:

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
        return tuple(self._layers.values())

    def drop(self, count: int, note: str) -> None:

        self.dropped += int(count)
        if note and note != self._last_drop_note:
            self._last_drop_note = note
            log.warning("Debug draw dropped primitives: {}", note)

    def render_frame(self, emit, now: float | None = None) -> PackedFrame:

        self.now = time.monotonic() if now is None else float(now)
        frame = self.build()
        emit(frame)
        self.expire(self.now)
        return frame

    def expire(self, now: float | None = None) -> int:

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
        for layer in self._layers.values():
            layer.clear()

    def stats(self) -> DebugStats:
        prims = 0
        verts = 0
        expiring = 0
        for layer in self._layers.values():
            expiring += len(layer._finite) + len(layer._text_finite)
            prims += len(layer._texts)
            verts += sum(6 * len(label.text) for label in layer._texts.values())
            for kind, st in layer._stores.items():
                prims += st.count
                verts += st.count * VERTEX_COUNT[kind]
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
        for path in Path:
            frame.counts[path] = 0
        frame.batch_count = 0
        frame.text_count = 0
        self._reserve(frame)

        for occ in OCCLUSION_ORDER:
            layers = self._by_occlusion[occ]
            if not layers:
                continue

            self._batch_solid(frame, occ, layers)
            self._batch(frame, occ, layers, Path.SECTOR, (Prim.SECTOR,), self._pack_sector)
            self._batch(frame, occ, layers, Path.STROKE, (Prim.STROKE,), self._pack_stroke)
            self._batch(
                frame,
                occ,
                layers,
                Path.DRAG_LINK,
                (Prim.DRAG_LINK,),
                self._pack_drag_link,
            )
            self._batch(
                frame,
                occ,
                layers,
                Path.SEGMENT,
                (Prim.LINE, Prim.ARROW, Prim.FRAME),
                self._pack_segment,
            )
            self._batch(frame, occ, layers, Path.POINT, (Prim.POINT,), self._pack_point)
            for layer in layers:
                for label in layer._texts.values():
                    if frame.text_count == len(frame.texts):
                        frame.texts.append(label)
                    else:
                        frame.texts[frame.text_count] = label
                    frame.text_count += 1
        return frame

    def _reserve(self, frame: PackedFrame) -> None:

        need = self._need
        for path in Path:
            need[path] = 0
        for layer in self._layers.values():
            for kind, st in layer._stores.items():
                if st.count == 0:
                    continue

                factor = 3 if kind is Prim.FRAME else 1
                need[PRIM_PATH[kind]] += st.count * factor
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

    def _batch(self, frame, occ, layers, path, kinds, pack) -> None:
        start = frame.counts[path]
        dst = frame.streams[path]
        at = start
        for layer in layers:
            for kind in kinds:
                st = layer._stores.get(kind)
                if st is not None and st.count:
                    at = pack(dst, at, st)
        if at == start:
            return
        frame.counts[path] = at
        b = self._new_batch(frame)
        b.occlusion, b.path, b.mesh, b.start, b.count = occ, path, None, start, at - start

    def _batch_solid(self, frame, occ, layers) -> None:

        for kind, mesh in PRIM_MESH.items():
            start = frame.counts[Path.SOLID]
            dst = frame.streams[Path.SOLID]
            at = start
            for layer in layers:
                st = layer._stores.get(kind)
                if st is not None and st.count:
                    at = self._pack_solid(dst, at, st)
            if at == start:
                continue
            frame.counts[Path.SOLID] = at
            b = self._new_batch(frame)
            b.occlusion, b.path, b.mesh, b.start, b.count = occ, Path.SOLID, mesh, start, at - start

    @staticmethod
    def _pack_segment(dst: np.ndarray, at: int, st: _Store) -> int:
        n = st.count
        if st.kind is Prim.FRAME:
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

        dst[s, 11] = st.sizes[:n] * ARROW_HEAD_RATIO if st.kind is Prim.ARROW else 0.0
        dst[s, 12] = st.extras[:n] if st.kind is Prim.ARROW else 0.0
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

    return float(size_px) * float(px_scale) * float(clip_w)


def sector_angle(center, rotvec_end) -> float:

    return float(
        np.linalg.norm(np.asarray(rotvec_end, np.float64) - np.asarray(center, np.float64))
    )


def sector_points(center, rotvec_end, ref_end, segments: int = 32) -> np.ndarray:

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
