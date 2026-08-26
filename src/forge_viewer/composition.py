"""Viewer construction and high-level runtime helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .log import get_logger

log = get_logger("composition")

if TYPE_CHECKING:
    from .adapters.base import SceneAdapter
    from .scene import Scene


def render_backend_name() -> str:
    """The renderer selected by FORGE_VIEWER_BACKEND ("forge" or "wgpu")."""
    requested = os.environ.get("FORGE_VIEWER_BACKEND", "").strip().lower()
    if requested == "webgpu":
        requested = "wgpu"
    if requested not in {"", "forge", "wgpu"}:
        raise ValueError(f"Unsupported FORGE_VIEWER_BACKEND: {requested}")
    return requested or "forge"


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

    def run(self, max_frames: int | None = None) -> None:
        """Run the UI event loop until the window closes or a frame limit is reached."""
        self.app.run(max_frames=max_frames)

    def sync(self) -> None:
        """Advance the session and render one frame without entering the event loop."""
        self.app.sync()

    def release(self) -> None:
        """Release the bridge, renderer, session, and native window."""
        self.bridge.close()
        for obj, what in ((self.backend, "backend"), (self.session, "session")):
            try:
                obj.release()
            except Exception as e:
                log.warning("Failed to release {}: {}", what, e)
        try:
            self.window.close()
        except Exception as e:
            log.warning("Failed to close the window: {}", e)

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

        if frames <= 0:
            raise ValueError("frame count must be positive")
        if size is not None:
            self.app.set_fixed_render_size(*size)
        recorder = None
        try:
            for index in range(int(frames)):
                if before_frame is not None:
                    before_frame(index, self)
                self.sync()
                image = self.backend.target.read_color(flip=True)[..., :3]
                if recorder is None:
                    recorder = VideoRecorder(
                        output, (int(image.shape[1]), int(image.shape[0])), fps=fps
                    )
                recorder.append(image)
        finally:
            if recorder is not None:
                recorder.close()
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
    title: str = "forge-viewer",
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
    title: str = "forge-viewer",
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
    )


def build_from_adapter(
    adapter: SceneAdapter,
    *,
    paused: bool = False,
    vsync: bool = True,
    width: int = 1600,
    height: int = 1000,
    samples: int = 4,
    title: str = "forge-viewer",
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
    )


def build_scene(scene: Scene, **kwargs) -> Viewer:
    """Build an interactive viewer for a backend-neutral authored scene."""

    from .adapters.static import StaticSceneAdapter

    return build_from_adapter(StaticSceneAdapter(scene), **kwargs)


def build_editor(**kwargs) -> Viewer:
    """Build an empty workspace with MuJoCo models and Forge-authored entities."""
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
) -> Viewer:
    from . import commands as cmd
    from .bridge import DebugBridge
    from .session import Session
    from .ui.app import ViewerApp
    from .ui.window import WindowConfig

    ini = "imgui.ini" if vsync else ""
    config = WindowConfig(
        title=title,
        width=width,
        height=height,
        vsync=vsync,
        ini_path=ini,
    )

    if render_backend_name() == "wgpu":
        from .render.webgpu.backend import WgpuBackend
        from .ui.window_wgpu import WgpuWindow

        window = WgpuWindow(config)
        fb_w, fb_h = window.size_pixels
        backend = WgpuBackend(fb_w, fb_h, samples, device=window.device)
    else:
        from .render.forge.backend import ForgeBackend
        from .ui.window import Window

        window = Window(config)
        window.make_current()
        fb_w, fb_h = window.size_pixels
        backend = ForgeBackend(None, fb_w, fb_h, samples)

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
            from .render.forge import gl_native as G

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
