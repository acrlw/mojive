"""Offscreen rendering for authored scenes and custom scene providers."""

from __future__ import annotations

import contextlib
import operator
from contextlib import contextmanager

import numpy as np

from .adapters.base import FrameNeeds, SceneFrame, SceneProvider, SceneSource
from .render.backend import (
    DebugView,
    RenderBackend,
    RenderFlag,
    RenderProduct,
    RenderRequest,
    ShadowQuality,
)
from .render.context import _select_backend
from .types import CameraView


class SceneRenderer:
    """Render shared scene contracts without a physics engine or UI window.

    Call ``set_scene`` after structure changes and ``update`` for dynamic frames,
    or use ``update_from`` to follow a provider's structure revision automatically.
    Providers remain caller-owned and are never stepped or released here. Graphics
    calls belong on the thread that constructed the renderer.
    """

    def __init__(
        self,
        source: SceneSource | None = None,
        *,
        width: int = 320,
        height: int = 240,
        samples: int = 4,
        renderer: str | None = None,
        camera: CameraView | None = None,
        shadow_quality: ShadowQuality | str = ShadowQuality.BALANCED,
    ) -> None:
        self._closed = False
        self._context = None
        self._backend: RenderBackend | None = None
        self._width, self._height = _dimensions(width, height)
        self._camera = camera or CameraView()
        self._provider: SceneProvider | None = None
        self._provider_revision: int | None = None
        quality = ShadowQuality(shadow_quality)
        self._context, self._backend = _select_backend(width, height, samples, renderer)
        try:
            with self._current():
                self._backend.set_scene(source if source is not None else SceneSource())
                self._backend.set_background((0.0, 0.0, 0.0, 1.0))
                self._backend.set_camera(self._camera.with_aspect(width / height))
                if not self._backend.set_shadow_quality(quality):
                    raise RuntimeError("The renderer does not support shadow quality presets")
        except Exception:
            self.close()
            raise

    @contextmanager
    def _current(self):
        if self._closed:
            raise RuntimeError("SceneRenderer is closed")
        if self._context is None:
            yield
        else:
            with self._context.current():
                yield

    @property
    def width(self) -> int:
        """Return the physical image width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Return the physical image height in pixels."""
        return self._height

    def set_scene(self, source: SceneSource) -> None:
        """Upload stable structure; call again when its topology or resources change."""
        with self._current():
            self._backend.set_scene(source)
        self._provider = None
        self._provider_revision = None

    def update(self, frame: SceneFrame, *, camera: CameraView | None = None) -> None:
        """Upload one dynamic frame, optionally changing the world camera."""
        with self._current():
            if camera is not None:
                self._camera = camera
            self._backend.set_camera(self._camera.with_aspect(self._width / self._height))
            self._backend.update(frame)

    def set_flag(self, flag: RenderFlag | str, enabled: bool) -> bool:
        """Set a renderer feature, returning whether the backend supports it."""
        with self._current():
            return self._backend.set_flag(RenderFlag(flag), enabled)

    def set_debug_view(self, view: DebugView | str) -> bool:
        """Select an RGB diagnostic view without changing data-product formats."""
        with self._current():
            return self._backend.set_debug_view(DebugView(view))

    def update_from(
        self,
        provider: SceneProvider,
        *,
        needs: FrameNeeds | None = None,
        camera: CameraView | None = None,
    ) -> None:
        """Read a prepared provider frame and upload structure only when it changes.

        Request deformables by default. Pass explicit frame needs when the scene
        is known to contain only rigid geometry or needs additional diagnostics.
        The caller controls simulation steps and remote publisher preparation.
        """
        frame = provider.frame(needs if needs is not None else FrameNeeds(deformables=True))
        revision = provider.structure_revision
        if provider is not self._provider or revision != self._provider_revision:
            self.set_scene(provider.scene_source())
            self._provider = provider
            self._provider_revision = revision
        self.update(frame, camera=camera)

    def render(
        self, *, product: RenderProduct = RenderProduct.COLOR, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Return one top-left-oriented image, optionally reusing ``out``.

        COLOR returns RGB uint8; METRIC_DEPTH returns float32 world distances;
        OBJECT_ID returns uint32 selection identity (zero background);
        SEGMENTATION returns int32 pairs defined by the source's semantic metadata.
        Request one product per call. ``out`` must match its shape and dtype.
        """
        product = RenderProduct(product)
        h, w = self._height, self._width
        formats = {
            RenderProduct.COLOR: ((h, w, 3), np.dtype("uint8"), "read_rgb"),
            RenderProduct.METRIC_DEPTH: ((h, w), np.dtype("float32"), "read_metric_depth"),
            RenderProduct.OBJECT_ID: ((h, w), np.dtype("uint32"), "read_ids"),
            RenderProduct.SEGMENTATION: ((h, w, 2), np.dtype("int32"), "read_segmentation"),
        }
        if product not in formats:
            raise ValueError("Select exactly one render product")
        shape, dtype, reader = formats[product]
        if out is not None and (
            out.shape != shape or out.dtype != dtype or not out.flags.writeable
        ):
            raise ValueError(f"out must be a writable {dtype} array with shape {shape}")
        direct = out if out is not None and out.flags.c_contiguous else None
        with self._current():
            self._backend.render(request=RenderRequest(product))
            read = getattr(self._backend.target, reader)
            image = (
                read(flip=True)
                if product is RenderProduct.OBJECT_ID
                else read(flip=True, out=direct)
            )
        if out is None or image is out:
            return image
        np.copyto(out, image)
        return out

    def resize(self, width: int, height: int) -> None:
        """Resize image targets and update camera aspect without changing its pose."""
        width, height = _dimensions(width, height)
        with self._current():
            self._backend.resize(width, height)
            self._backend.set_camera(self._camera.with_aspect(width / height))
        self._width, self._height = width, height

    def close(self) -> None:
        """Release GPU resources, leaving the caller's provider alive."""
        if self._closed:
            return
        try:
            if self._backend is not None:
                with self._current():
                    self._backend.release()
        finally:
            self._closed = True
            self._backend = None
            self._provider = None
            if self._context is not None:
                self._context.close()
                self._context = None

    def __enter__(self) -> SceneRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _dimensions(width: int, height: int) -> tuple[int, int]:
    width, height = operator.index(width), operator.index(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive integers")
    return width, height
