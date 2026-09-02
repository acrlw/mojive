"""Viewer construction and high-level runtime helpers."""

from __future__ import annotations

import contextlib
import operator
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .log import get_logger

log = get_logger("composition")

if TYPE_CHECKING:
    from .adapters.base import SceneAdapter
    from .scene import Scene


def render_backend_name() -> str:
    """The renderer selected by MOJIVE_BACKEND ("opengl" or "wgpu")."""
    requested = (
        os.environ.get("MOJIVE_BACKEND", os.environ.get("FORGE_VIEWER_BACKEND", "")).strip().lower()
    )
    if requested == "forge":
        requested = "opengl"
    if requested == "webgpu":
        requested = "wgpu"
    if requested not in {"", "opengl", "wgpu"}:
        raise ValueError(f"Unsupported MOJIVE_BACKEND: {requested}")
    return requested or "opengl"


@dataclass
class Viewer:
    """Own the objects that make up one interactive viewer.

    Create instances through :func:`build`, :func:`build_workspace`,
    :func:`build_scene`, or :func:`build_from_adapter`. Call :meth:`release`
    when an application embeds the viewer without using a context manager.
    """

    app: Any
    session: Any
    backend: Any
    window: Any
    bridge: Any
    _released: bool = field(default=False, init=False, repr=False)

    def run(self, max_frames: int | None = None) -> None:
        """Run the UI event loop until the window closes or a frame limit is reached."""
        self.app.run(max_frames=max_frames)

    def sync(self) -> None:
        """Advance the session and render one frame without entering the event loop."""
        self.app.sync()

    def release(self) -> None:
        """Release application-owned resources and the native window once."""
        if self._released:
            return
        self._released = True
        try:
            self.app.release()
        except Exception as e:
            log.warning("Failed to release viewer application: {}", e)
        try:
            self.window.close()
        except Exception as e:
            log.warning("Failed to close the window: {}", e)

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.release()

    def record(
        self,
        output: Path,
        frames: int,
        fps: float = 30.0,
        before_frame: Callable[[int, Viewer], None] | None = None,
        size: tuple[int, int] | None = None,
    ) -> Path:
        """Render a fixed number of frames to a video file.

        Args:
            output: Destination video path.
            frames: Number of frames to encode.
            fps: Playback frame rate stored in the video.
            before_frame: Optional callback invoked before each rendered frame.
            size: Optional fixed render width and height.

        Returns:
            The destination path.
        """
        from .recording import VideoRecorder

        try:
            frame_count = operator.index(frames)
        except TypeError as exc:
            raise TypeError("frame count must be an integer") from exc
        if frame_count <= 0:
            raise ValueError("frame count must be positive")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("frame rate must be finite and positive")
        previous_size = self.app.fixed_render_size if size is not None else None
        if size is not None:
            self.app.set_fixed_render_size(*size)
        recorder = None
        pending = deque()
        target = None
        async_read = None

        def append(image: np.ndarray) -> None:
            nonlocal recorder
            if recorder is None:
                recorder = VideoRecorder(
                    output, (int(image.shape[1]), int(image.shape[0])), fps=fps
                )
            recorder.append(image)

        try:
            for index in range(frame_count):
                if before_frame is not None:
                    before_frame(index, self)
                if target is None:
                    target = self.backend.target
                    async_read = getattr(target, "read_rgb_async", None)
                self.sync()
                if callable(async_read):
                    pending.append(async_read(flip=True))
                    if len(pending) >= 3:
                        append(pending.popleft().result())
                else:
                    append(target.read_color(flip=True)[..., :3])
            while pending:
                append(pending.popleft().result())
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                if size is not None:
                    if previous_size is None:
                        self.app.clear_fixed_render_size()
                    else:
                        self.app.set_fixed_render_size(*previous_size)
        return Path(output)


def build(
    asset: Path,
    backend_name: str = "mujoco",
    *,
    paused: bool = True,
    vsync: bool = True,
    width: int = 1600,
    height: int = 1000,
    samples: int = 4,
    title: str = "Mojive",
    show_window: bool = True,
) -> Viewer:
    """Build an interactive viewer for a model or scene asset.

    Args:
        asset: Asset path accepted by the selected scene adapter.
        backend_name: Scene adapter name, such as ``"mujoco"``.
        paused: Start physics in the paused state.
        vsync: Synchronize presentation to the display.
        width: Initial logical window width.
        height: Initial logical window height.
        samples: Requested MSAA sample count.
        title: Native window title.
        show_window: Show the native window when rendering starts. Disable for
            automated UI tests and off-screen capture.

    Returns:
        A composed viewer ready to run or step manually.
    """
    from .backends import make_adapter

    return _compose(
        lambda: make_adapter(backend_name, asset),
        asset_path=asset,
        paused=paused,
        vsync=vsync,
        width=width,
        height=height,
        samples=samples,
        title=title,
        show_window=show_window,
    )


def build_workspace(
    asset: Path,
    backend_name: str = "mujoco",
    *,
    paused: bool = True,
    vsync: bool = True,
    width: int = 1600,
    height: int = 1000,
    samples: int = 4,
    title: str = "Mojive",
    show_window: bool = True,
) -> Viewer:
    """Build an editable workspace around a model adapter."""
    from .adapters.workspace import WorkspaceAdapter
    from .backends import make_adapter

    return _compose(
        lambda: WorkspaceAdapter(make_adapter(backend_name, asset)),
        asset_path=asset,
        paused=paused,
        vsync=vsync,
        width=width,
        height=height,
        samples=samples,
        title=title,
        show_window=show_window,
    )


def build_from_adapter(
    adapter: SceneAdapter,
    *,
    paused: bool = False,
    vsync: bool = True,
    width: int = 1600,
    height: int = 1000,
    samples: int = 4,
    title: str = "Mojive",
    show_window: bool = True,
) -> Viewer:
    """Build an interactive viewer around an initialized scene adapter."""

    return _compose(
        lambda: adapter,
        asset_path=None,
        paused=paused,
        vsync=vsync,
        width=width,
        height=height,
        samples=samples,
        title=title,
        show_window=show_window,
    )


def build_scene(scene: Scene, **kwargs) -> Viewer:
    """Build an interactive viewer for a backend-neutral authored scene."""

    from .adapters.static import StaticSceneAdapter

    return build_from_adapter(StaticSceneAdapter(scene), **kwargs)


def build_editor(**kwargs) -> Viewer:
    """Build an empty workspace with MuJoCo models and Mojive-authored entities."""
    from .adapters.mujoco_adapter import MuJoCoAdapter
    from .adapters.workspace import WorkspaceAdapter

    primary = MuJoCoAdapter()
    primary.new_scene()
    return build_from_adapter(WorkspaceAdapter(primary), paused=True, **kwargs)


def _compose(
    adapter_factory: Callable[[], SceneAdapter],
    *,
    asset_path: Path | None,
    paused: bool,
    vsync: bool,
    width: int,
    height: int,
    samples: int,
    title: str,
    show_window: bool,
) -> Viewer:
    from . import commands as cmd
    from .bridge import DebugBridge
    from .session import Session
    from .ui.app import ViewerApp
    from .ui.window import WindowConfig, layout_settings_path

    ini = str(layout_settings_path()) if vsync else ""
    if ini:
        Path(ini).parent.mkdir(parents=True, exist_ok=True)
    config = WindowConfig(
        title=title,
        width=width,
        height=height,
        vsync=vsync,
        ini_path=ini,
        show_on_start=show_window,
    )

    window = None
    backend = None
    debug_bridge = None
    adapter = None
    session = None
    try:
        if render_backend_name() == "wgpu":
            from .render.webgpu.backend import WgpuBackend
            from .ui.window_wgpu import WgpuWindow

            window = WgpuWindow(config)
            fb_w, fb_h = window.size_pixels
            backend = WgpuBackend(fb_w, fb_h, samples, device=window.device)
        else:
            from .render.opengl.backend import OpenGLBackend
            from .ui.window import Window

            window = Window(config)
            window.make_current()
            fb_w, fb_h = window.size_pixels
            backend = OpenGLBackend(None, fb_w, fb_h, samples)

        debug_bridge = DebugBridge(backend)
        debug_bridge.serve()

        adapter = adapter_factory()
        session = Session(adapter, asset_path)
        if paused and not session.paused:
            session.submit(cmd.Pause())

        app = ViewerApp(session, backend, window, title=title, debug_bridge=debug_bridge)
        return Viewer(
            app=app,
            session=session,
            backend=backend,
            window=window,
            bridge=debug_bridge,
        )
    except Exception:
        if debug_bridge is not None:
            with contextlib.suppress(Exception):
                debug_bridge.close()
        if backend is not None:
            with contextlib.suppress(Exception):
                backend.release()
        if session is not None:
            with contextlib.suppress(Exception):
                session.release()
        elif adapter is not None:
            with contextlib.suppress(Exception):
                adapter.release()
        if window is not None:
            with contextlib.suppress(Exception):
                window.close()
        raise


def doctor(asset: Path, backend_name: str = "mujoco", frames: int = 90) -> dict:
    """Exercise viewer composition and return structured diagnostic checks.

    The diagnostic creates a real window and renderer, renders several frames,
    checks readback and simulation progress, then releases all resources.
    """
    checks: list[tuple[str, bool, str]] = []
    viewer: Viewer | None = None
    try:
        viewer = build(asset, backend_name, paused=False, vsync=False, width=960, height=720)
        caps = viewer.backend.caps
        is_wgpu = caps.name == "wgpu"

        if is_wgpu:
            checks.append(("GPU device", True, caps.renderer))
        else:
            gl = viewer.backend.gl_caps
            checks.append(
                ("GL context", gl.usable, f"{gl.version} ({gl.renderer}) core={gl.core_profile}")
            )
        target = viewer.backend.target
        target_detail = f"{target.width}×{target.height} {target.samples}× MSAA"
        id_layout = getattr(target, "id_layout", None)
        if id_layout is not None:
            target_detail += f", id layout {id_layout}"
        checks.append(("render target", target.width > 0 and target.height > 0, target_detail))
        failures = getattr(viewer.backend, "pass_load_failures", {})
        checks.append(
            ("pass loading", not failures, "complete" if not failures else f"failed: {failures}")
        )

        last_image = None
        for _ in range(frames):
            viewer.sync()
            last_image = viewer.app._viewport_image
            if viewer.window.should_close():
                break

        checks.append((f"render {frames} frames", True, f"{viewer.window.frame_index} frames"))
        checks.append(
            (
                "viewport image",
                last_image is not None,
                "ViewportImage received" if last_image is not None else "render() returned None",
            )
        )
        if last_image is not None:
            # WebGPU color targets are top-row-first; the GL resolve texture is not.
            expect_flip = not is_wgpu
            checks.append(
                ("flip_y", last_image.flip_y is expect_flip, f"flip_y={last_image.flip_y}")
            )

        frame_px = viewer.window.read_frame()
        if frame_px is None:
            checks.append(("window readback", False, "read_frame() returned None"))
        else:
            spread = float(np.asarray(frame_px)[..., :3].std())
            checks.append(
                ("window content", spread > 1.0, f"pixel standard deviation {spread:.2f}")
            )

        f = viewer.session.frame
        checks.append(("simulation step", f.step > 0, f"step={f.step} time={f.time:.3f}s"))

        if not is_wgpu:
            from .render.opengl import gl_native as G

            err = G.native().drain_errors()
            checks.append(("GL errors", err == 0, f"glGetError={err}"))

        stats = viewer.backend.stats
        checks.append(
            (
                "statistics",
                stats.instances > 0,
                f"draws {stats.draw_calls} · instances {stats.instances} · triangles {stats.triangles}",
            )
        )
        checks.append(("capabilities", bool(caps.render_flags), f"{len(caps.render_flags)} flags"))
    except Exception as e:
        checks.append(("composition", False, f"{type(e).__name__}: {e}"))
        log.exception("Doctor setup failed")
    finally:
        if viewer is not None:
            viewer.release()

    return {"ok": all(ok for _n, ok, _m in checks), "frames": frames, "checks": checks}


def capture(
    asset: Path,
    output: Path,
    backend_name: str = "mujoco",
    *,
    include_ui: bool = False,
    size: tuple[int, int] | None = None,
    settle_frames: int = 30,
    render_flags: tuple[str, ...] = (),
    camera_name: str = "",
) -> bool:
    """Render an asset to a PNG image.

    Args:
        asset: Model or scene path.
        output: Destination image path.
        backend_name: Scene adapter used to load the asset.
        include_ui: Capture the complete application window.
        size: Optional render width and height.
        settle_frames: Frames rendered before capture.
        render_flags: Renderer feature names enabled for the capture.
        camera_name: Optional named model camera.

    Returns:
        ``True`` when the image was written successfully.
    """
    viewer: Viewer | None = None
    try:
        w, h = size or (1600, 1000)
        viewer = build(asset, backend_name, paused=True, vsync=False, width=w, height=h)
        from .render.backend import RenderFlag

        for name in render_flags:
            viewer.backend.set_flag(RenderFlag(name), True)
        if camera_name:
            camera = next(
                (item for item in viewer.session.cameras if item.name == camera_name), None
            )
            if camera is None:
                raise ValueError(f"model camera {camera_name!r} is unavailable")
            viewer.app.select_model_camera(camera.camera_id)
        for _ in range(max(1, settle_frames)):
            viewer.sync()

        output.parent.mkdir(parents=True, exist_ok=True)
        if include_ui:
            from PIL import Image

            px = viewer.window.read_frame()
            if px is None:
                return False
            arr = np.asarray(px)

            Image.fromarray(arr[::-1][..., :3], "RGB").save(output)
            return True
        return bool(viewer.backend.capture(output, size=size))
    except Exception:
        log.exception("Capture failed")
        return False
    finally:
        if viewer is not None:
            viewer.release()
