"""Capture composed session state with explicit offscreen resource ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from .adapters.base import FrameNeeds
from .render.backend import DebugView, RenderFlag, RenderProduct
from .scene_renderer import SceneRenderer
from .session import Session
from .types import CameraView


class SessionCapture:
    """Render a caller-owned session without creating a second scene adapter.

    The caller serializes session access, including this operation. Headless RPC
    uses a dedicated graphics thread so persistent clients never migrate a context
    between socket workers. Attached viewers use their existing UI thread.
    """

    def __init__(self, session: Session, *, threaded: bool = False) -> None:
        self.session = session
        self.flags: dict[RenderFlag, bool] = {}
        self.debug_view: DebugView | None = None
        self.dynamic_opacity = 1.0
        self._uploaded_opacity = 1.0
        self._renderer: SceneRenderer | None = None
        self._generation = -1
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="mojive-capture")
            if threaded
            else None
        )

    def render(
        self,
        camera: CameraView,
        *,
        width: int,
        height: int,
        product: RenderProduct,
        camera_id: int = -1,
    ) -> np.ndarray:
        """Refresh composed scene data and synchronously return a top-left image."""
        if self._closed:
            raise RuntimeError("SessionCapture is closed")
        # Diagnostics are requested only when explicitly enabled. Tendons and
        # deformable meshes can be part of the scene's ordinary visible geometry.
        diagnostics = any(
            self.flags.get(flag, False)
            for flag in (
                RenderFlag.JOINT,
                RenderFlag.ACTUATOR,
                RenderFlag.ACTIVATION,
                RenderFlag.COM,
                RenderFlag.INERTIA,
                RenderFlag.SCLINERTIA,
                RenderFlag.CAMERA,
                RenderFlag.LIGHT,
                RenderFlag.RANGEFINDER,
                RenderFlag.CONSTRAINT,
                RenderFlag.AUTOCONNECT,
                RenderFlag.BODYBVH,
                RenderFlag.MESHBVH,
            )
        )
        frame = self.session.tick(
            FrameNeeds(
                contacts=any(
                    self.flags.get(flag, False)
                    for flag in (
                        RenderFlag.CONTACTPOINT,
                        RenderFlag.CONTACTFORCE,
                        RenderFlag.CONTACTSPLIT,
                        RenderFlag.ISLAND,
                    )
                ),
                tendons=True,
                deformables=True,
                actuator=self.flags.get(RenderFlag.ACTUATOR, False)
                or self.flags.get(RenderFlag.ACTIVATION, False),
                diagnostics=diagnostics,
                islands=self.flags.get(RenderFlag.ISLAND, False),
                bvh=self.flags.get(RenderFlag.BODYBVH, False)
                or self.flags.get(RenderFlag.MESHBVH, False),
            ),
            wall_dt=0.0,
        )
        if camera_id >= 0:
            camera = self.session.camera_view(camera_id) or camera
        args = (
            self.session.source,
            self.session.structure_generation,
            frame,
            camera,
            width,
            height,
            product,
        )
        if self._executor is not None:
            return self._executor.submit(self._render, *args).result()
        return self._render(*args)

    def _render(self, source, generation, frame, camera, width, height, product):
        structure_changed = (
            self._generation != generation or self._uploaded_opacity != self.dynamic_opacity
        )
        if structure_changed and self.dynamic_opacity != 1.0:
            rgba = source.geom_rgba.copy()
            rgba[~source.geom_static, 3] *= self.dynamic_opacity
            source = replace(source, geom_rgba=rgba)
        if self._renderer is None:
            self._renderer = SceneRenderer(source, width=width, height=height)
            self._generation = generation
        else:
            if (self._renderer.width, self._renderer.height) != (width, height):
                self._renderer.resize(width, height)
            if structure_changed:
                self._renderer.set_scene(source)
                self._generation = generation
        self._uploaded_opacity = self.dynamic_opacity
        for flag, enabled in self.flags.items():
            if not self._renderer.set_flag(flag, enabled) and enabled:
                raise NotImplementedError(f"The capture renderer does not support {flag.value}")
        if self.debug_view is not None and not self._renderer.set_debug_view(self.debug_view):
            raise NotImplementedError(
                f"The capture renderer does not support {self.debug_view.value}"
            )
        self._renderer.update(frame, camera=camera)
        return self._renderer.render(product=product)

    def reset(self) -> None:
        """Release capture resources on their owner thread, retaining flag choices."""
        if self._renderer is None:
            return
        if self._executor is not None:
            self._executor.submit(self._release).result()
        else:
            self._release()

    def _release(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._generation = -1

    def close(self) -> None:
        """Release graphics resources and stop the worker, leaving the session alive."""
        self.reset()
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self._closed = True
