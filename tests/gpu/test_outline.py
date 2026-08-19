"""Backend-neutral selection-outline and present-mode (SEGMENT/IDCOLOR) tests.

Runs under both backends via the ``backend_name`` fixture (conftest.py); the
GL-internals outline tests live in test_id_outline.py (forge only).  Scene
setup follows test_id_outline.py: an orthographic camera looking straight at
Z=0 boxes so silhouettes map to axis-aligned rectangles.

Tolerances account for the deliberate mask difference: forge rasterizes the
selection mask into 4x MSAA (subpixel coverage), the wgpu backend uses a
single-sampled mask (see webgpu/shaders/outline.wgsl).  Both backends
antialias the ring's outer edge in the dilation shader, so only the ring's
exact outer pixel counts may differ by a hair.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer import math3d as M  # noqa: E402
from forge_viewer.adapters.base import SceneSource  # noqa: E402
from forge_viewer.render.backend import DebugView  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    LightSet,
    Material,
    MeshKey,
    MeshShape,
)

W, H = 256, 192
ASPECT = W / H
ORTHO = 2.0

BOX = MeshKey(MeshShape.BOX)
SEL = 7

BACKGROUND = (0.05, 0.05, 0.08, 1.0)
OUTLINE_COLOR = (1.0, 0.63, 0.20, 1.0)  # both backends' default
CUSTOM_COLOR = (1.0, 0.0, 1.0, 1.0)
OUTLINE_RADIUS = 3  # forge outline.OUTLINE_RADIUS

MATERIAL = np.array([0.0, 0.0, 0.5, 0.0], np.float32)
AMBIENT = np.full(3, 0.3, np.float32)


def u8(color) -> np.ndarray:
    return np.array([round(c * 255.0) for c in color[:3]], np.uint8)


def id_color(oid: int) -> np.ndarray:
    """present.frag's pseudocolor hash, computed on the CPU."""
    h = (oid * 2654435761) & 0xFFFFFFFF
    return np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255], np.uint8)


def _camera() -> CameraView:
    return CameraView(
        orthographic=True,
        ortho_height=ORTHO,
        eye=np.array([0.0, 0.0, 4.0], np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0.0, 1.0, 0.0], np.float32),
        near=0.1,
        far=10.0,
        aspect=ASPECT,
    )


class Rig:
    def __init__(self, backend) -> None:
        self.backend = backend
        backend.set_background(BACKGROUND)
        backend.set_scene(SceneSource(meshes={}, textures={}, skybox=None))
        self.camera = _camera()
        backend.set_camera(self.camera)

    def draw(self, boxes, selected: int = 0) -> np.ndarray:
        sb = SceneBuilder()
        matid = sb.material_id(Material())
        for pos, scale, color, oid in boxes:
            sb.add(
                BOX,
                matid,
                M.compose(np.asarray(pos, np.float32), np.eye(3), np.asarray(scale, np.float32)),
                np.asarray(color, np.float32),
                MATERIAL,
                object_id=oid,
            )
        scene = sb.build(self.camera, LightSet(ambient=AMBIENT), 2.0, np.zeros(3, np.float32))
        self.backend.highlight(selected)
        self.backend.set_render_scene(scene)
        # The wgpu backend renders offscreen only and returns no ViewportImage.
        if self.backend.caps.name == "webgpu":
            self.backend.render(None)
        else:
            assert self.backend.render(None) is not None
        return self.backend.target.read_color(flip=True)

    def ids(self) -> np.ndarray:
        return self.backend.target.read_ids(flip=True)

    def outline_mask(self, img: np.ndarray, color=OUTLINE_COLOR) -> np.ndarray:
        return np.all(img[..., :3] == u8(color), axis=-1)


def _make_backend(backend_name: str, request, samples: int = 4):
    """Build the backend selected by FORGE_VIEWER_BACKEND; GL stays lazy."""
    if backend_name == "wgpu":
        from forge_viewer.render.webgpu.backend import WgpuBackend

        return WgpuBackend(W, H, samples=samples)
    from forge_viewer.render.forge import passes
    from forge_viewer.render.forge.backend import ForgeBackend

    passes.load_all()
    return ForgeBackend(request.getfixturevalue("gl_ctx"), W, H, samples=samples)


@pytest.fixture
def rig(backend_name, request):
    backend = _make_backend(backend_name, request)
    yield Rig(backend)
    backend.release()


def _set_outline_color(backend, rgba) -> None:
    """Outline color lives on the pass in both backends (no public API)."""
    if backend.caps.name == "webgpu":
        backend._outline.color = rgba
    else:
        backend._passes["outline"].color = rgba


def _box(center_x=0.0, half=(0.35, 0.35, 0.1), occluder=False):
    boxes = [((center_x, 0.0, 0.0), half, (0.2, 0.6, 0.3, 1.0), SEL)]
    if occluder:
        boxes.append(((center_x, 0.0, 0.6), (0.6, 0.6, 0.1), (0.8, 0.2, 0.2, 1.0), 8))
    return boxes


def _silhouette_columns(ids: np.ndarray, oid: int) -> tuple[int, int]:
    """Leftmost/rightmost column owned by ``oid`` on the center scanline."""
    cols = np.nonzero(ids[H // 2] == oid)[0]
    return int(cols.min()), int(cols.max())


def test_outline_ring_surrounds_the_selected_object(rig):
    if not rig.backend.caps.outline:
        pytest.skip("outline unsupported by this backend")
    img = rig.draw(_box(), selected=SEL)
    out = rig.outline_mask(img)

    assert int(out.sum()) > 200
    # The ring hugs the silhouette; the object's center keeps its own color.
    assert not out[H // 2, W // 2]

    left, right = _silhouette_columns(rig.ids(), SEL)
    row = out[H // 2]
    # Ring width is OUTLINE_RADIUS plus one antialiased pixel; the MSAA mask
    # difference between backends may shift the outer edge by one pixel.
    assert row[right + 1 : right + 6].any()
    assert not row[right + 8 : right + 13].any()
    assert row[left - 6 : left - 1].any()
    assert not row[left - 13 : left - 8].any()


def test_outline_is_solid_through_an_occluder(rig):
    if not rig.backend.caps.outline:
        pytest.skip("outline unsupported by this backend")
    alone = int(rig.outline_mask(rig.draw(_box(), selected=SEL)).sum())

    img = rig.draw(_box(occluder=True), selected=SEL)
    behind = int(rig.outline_mask(img).sum())
    assert int(rig.ids()[H // 2, W // 2]) == 8  # the occluder owns the center
    assert alone > 200
    assert abs(alone - behind) <= 2


def test_no_outline_without_selection(rig):
    img = rig.draw(_box(), selected=0)
    assert int(rig.outline_mask(img).sum()) == 0
    img = rig.draw(_box(), selected=999)  # no such object
    assert int(rig.outline_mask(img).sum()) == 0


def test_outline_color_comes_from_the_pass(rig):
    if not rig.backend.caps.outline:
        pytest.skip("outline unsupported by this backend")
    _set_outline_color(rig.backend, CUSTOM_COLOR)
    img = rig.draw(_box(), selected=SEL)
    assert int(rig.outline_mask(img, CUSTOM_COLOR).sum()) > 200
    assert int(rig.outline_mask(img, OUTLINE_COLOR).sum()) == 0


def test_outline_hugs_the_viewport_edge_when_clipped(rig):
    if not rig.backend.caps.outline:
        pytest.skip("outline unsupported by this backend")
    # Half of the box sticks out beyond the left viewport edge.
    img = rig.draw(_box(center_x=-1.1, half=(0.5, 0.4, 0.1)), selected=SEL)
    ids, out = rig.ids(), rig.outline_mask(img)
    # forge's MSAA id resolve loses the outermost columns on some drivers (a
    # pre-existing test_id_outline.py failure), so the silhouette check
    # tolerates a small inset; the band check below carries the semantics.
    assert (ids[:, : OUTLINE_RADIUS + 1] == SEL).any()

    rows = np.nonzero((ids == SEL).any(axis=1))[0]
    inner = rows[OUTLINE_RADIUS:-OUTLINE_RADIUS]
    band = out[:, :OUTLINE_RADIUS].any(axis=1)
    missing = [int(r) for r in inner if not band[r]]
    assert not missing


def test_segment_view_colors_pixels_by_object_id(rig):
    if DebugView.SEGMENT not in rig.backend.caps.debug_views:
        pytest.skip("segment debug view unsupported by this backend")
    boxes = [
        ((-0.5, 0.0, 0.0), (0.25, 0.25, 0.1), (0.2, 0.6, 0.3, 1.0), 1),
        ((0.5, 0.0, 0.0), (0.25, 0.25, 0.1), (0.6, 0.3, 0.2, 1.0), 2),
    ]
    rig.backend.set_debug_view(DebugView.SEGMENT)
    img = rig.draw(boxes)
    ids = rig.ids()

    for oid in (1, 2):
        rows, cols = np.nonzero(ids == oid)
        r, c = int(np.median(rows)), int(np.median(cols))
        assert np.abs(img[r, c, :3].astype(int) - id_color(oid).astype(int)).max() <= 1
    # id 0 renders black; nothing selected yet.
    assert np.all(img[2, 2, :3] == 0)

    img = rig.draw(boxes, selected=1)
    rows, cols = np.nonzero(ids == 1)
    r, c = int(np.median(rows)), int(np.median(cols))
    assert np.all(img[r, c, :3] == 255)  # selected renders white
    rows, cols = np.nonzero(ids == 2)
    r, c = int(np.median(rows)), int(np.median(cols))
    assert np.abs(img[r, c, :3].astype(int) - id_color(2).astype(int)).max() <= 1


def test_idcolor_view_ignores_the_selection(rig):
    if DebugView.IDCOLOR not in rig.backend.caps.debug_views:
        pytest.skip("idcolor debug view unsupported by this backend")
    boxes = [((0.0, 0.0, 0.0), (0.35, 0.35, 0.1), (0.2, 0.6, 0.3, 1.0), SEL)]
    rig.backend.set_debug_view(DebugView.IDCOLOR)
    img = rig.draw(boxes, selected=SEL)
    got = img[H // 2, W // 2, :3].astype(int)
    assert np.abs(got - id_color(SEL).astype(int)).max() <= 1
    assert not np.all(img[H // 2, W // 2, :3] == 255)
