"""Typed panel dependencies, without panel registration or control implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.paint_protocol import Draw2D
from mojive.ui.theme import THEME, Theme
from mojive.ui.viewport_widgets import ToolHint

if TYPE_CHECKING:
    from mojive.capture import RecordingInfo
    from mojive.commands import Command, CommandResult
    from mojive.config import (
        CameraNavigationConfig,
        CameraTrackingConfig,
        InteractionConfig,
        RecordingConfig,
        SelectionStyle,
        ViewportLayers,
        ViewportOverlayConfig,
    )
    from mojive.interaction.input import InputClaim
    from mojive.render.backend import RenderBackend, ShadowQuality
    from mojive.session import Session
    from mojive.text.sources import FontReport
    from mojive.types import CameraView, ContactStyle, GeometryStyle
    from mojive.ui.camera import OrbitCamera
    from mojive.ui.camera_preview import CameraPreview
    from mojive.ui.gizmo import ObjectGizmo
    from mojive.ui.input_bindings import InputAction, InputBindings
    from mojive.ui.messages import OutputBuffer
    from mojive.ui.panels import PanelManager
    from mojive.ui.perturb import PerturbController
    from mojive.ui.pointer_bindings import PointerAction
    from mojive.ui.scene_entities import SceneEntityHelpers
    from mojive.ui.viewcube import ViewCube


class TextureImport(Protocol):
    """Open the texture importer with optional material assignment and texture kind."""

    def __call__(
        self, model_id: int, material_index: int = -1, texture_type: str = "2d"
    ) -> None: ...


class CameraNavigationSetter(Protocol):
    def __call__(self, value: CameraNavigationConfig, *, persist: bool = True) -> None: ...


@dataclass
class PanelContext:
    """Per-frame panel dependencies; document changes route through Session commands."""

    session: Session
    backend: RenderBackend
    camera: OrbitCamera | None = None
    set_camera_navigation: CameraNavigationSetter | None = None

    model_camera_id: int = -1
    model_camera_view: CameraView | None = None
    select_model_camera: Callable[[int], None] | None = None
    tracking: CameraTrackingConfig | None = None
    tracking_node_id: int | None = None
    track_node: Callable[[int | None], None] | None = None
    set_camera_tracking: Callable[[CameraTrackingConfig], None] | None = None
    focus_node: Callable[[int], bool] | None = None
    focus_joint: Callable[[int], bool] | None = None
    request_rename: Callable[[int], None] | None = None
    request_model_rename: Callable[[int], None] | None = None
    request_texture_import: TextureImport | None = None
    request_geometry_resource_import: Callable[[int, str], None] | None = None
    request_model_asset_import: Callable[[int, str, tuple[tuple[str, str], ...]], None] | None = (
        None
    )
    request_model_asset_replace: Callable[[int, str, str], None] | None = None
    queue_model_edit: Callable[[Command, Callable[[CommandResult], None] | None], None] | None = (
        None
    )
    model_keyframe_names: Callable[[int], set[str]] | None = None
    live_model_updates: bool = False
    set_live_model_updates: Callable[[bool], None] | None = None

    theme: Theme = THEME
    gizmo: ObjectGizmo | None = None
    view_cube: ViewCube | None = None
    perturb: PerturbController | None = None
    scene_entities: SceneEntityHelpers | None = None
    camera_preview: CameraPreview | None = None

    style_scale: float = 1.0

    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    dt: float = 0.0

    info: dict[str, Any] = field(default_factory=dict)

    status: str = ""
    popup_owned_frame: bool = False
    # Each panel publishes its available grammar independently of hover.
    # PanelManager collects it by name; the application selects the clicked panel.
    status_hints: tuple[ToolHint, ...] = ()
    status_hints_by_panel: dict[str, tuple[ToolHint, ...]] = field(default_factory=dict)

    panels: PanelManager | None = None

    language: str = "en"
    translate: Callable[[str], str] | None = None
    set_language: Callable[[str], None] | None = None
    set_shadow_quality: Callable[[ShadowQuality | str], bool] | None = None
    set_geometry_style: Callable[[GeometryStyle], bool] | None = None
    set_contact_style: Callable[[ContactStyle], bool] | None = None
    interactions: InteractionConfig | None = None
    set_interactions: Callable[[InteractionConfig], None] | None = None
    selection_style: SelectionStyle | None = None
    set_selection_style: Callable[[SelectionStyle], None] | None = None
    set_precise_input_memory: Callable[[bool], None] | None = None
    set_view_selection_padding: Callable[[float], None] | None = None
    set_perturb_strength: Callable[[float, float], None] | None = None
    viewport_overlay_scale: float = 1.0
    set_viewport_overlay_scale: Callable[[float], None] | None = None
    viewport_overlays: ViewportOverlayConfig | None = None
    set_viewport_overlays: Callable[[ViewportOverlayConfig], None] | None = None
    viewport_layers: ViewportLayers | None = None
    set_viewport_layers: Callable[[ViewportLayers], None] | None = None
    recording_config: RecordingConfig | None = None
    set_recording_config: Callable[[RecordingConfig], None] | None = None
    set_take_pause_at_end: Callable[[bool], None] | None = None
    recording: RecordingInfo | None = None
    take_video_active: bool = False
    start_take_video: Callable[[], Path] | None = None
    stop_recording: Callable[[], Path | None] | None = None
    set_viewport_capsule_scale: Callable[[str, float], None] | None = None
    input_bindings: InputBindings | None = None
    set_input_binding: Callable[[InputAction, str | None], None] | None = None
    set_pointer_binding: Callable[[PointerAction, tuple[str, ...]], None] | None = None
    set_navigation_preset: Callable[[str], None] | None = None
    input_claim: InputClaim | None = None
    reset_input_bindings: Callable[[], None] | None = None
    font_report: FontReport | None = None
    output: OutputBuffer | None = None

    # Resolve the painter inside the current child/table scope; never retain a draw list.
    painter: Callable[[], Draw2D] = ImguiDraw2D

    def submit(self, command: Command) -> CommandResult:
        result = self.session.submit(command)
        if result.message:
            self.status = result.message
        return result

    def submit_model_edit(
        self, command: Command, completed: Callable[[CommandResult], None] | None = None
    ) -> None:
        """Defer a rebuilding UI edit while retaining the synchronous Session API."""
        if self.queue_model_edit is not None:
            self.queue_model_edit(command, completed)
        else:
            result = self.submit(command)
            if completed is not None:
                completed(result)

    def report(
        self,
        message: str,
        *,
        level: str = "warning",
        duration: float | None = 5.0,
    ) -> None:
        """Keep a panel diagnostic visible in the shared status channel."""
        self.status = str(message)
        self.session.report_message(self.status, level=level, duration=duration)

    def tr(self, value: str) -> str:
        return self.translate(value) if self.translate is not None else value
