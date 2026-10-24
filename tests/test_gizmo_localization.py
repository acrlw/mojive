"""Localize interaction feedback without translating editable scene identities."""

from types import SimpleNamespace

import numpy as np

from mojive.interaction.gizmo import GizmoHandle
from mojive.scene.geometry import geometry_dimensions
from mojive.types import MeshShape
from mojive.ui.gizmo import ObjectGizmo
from mojive.ui.localization import Language, Localizer


def test_drag_feedback_follows_language_and_preserves_joint_names_and_values():
    localizer = Localizer()
    gizmo = ObjectGizmo("rotate", translate=localizer.text)
    gizmo._active = GizmoHandle.ROTATE_SCREEN
    gizmo._rotation_angle = np.radians(12)
    gizmo._snapping = True
    gizmo.rotation_snap_deg = 5
    assert gizmo._format_value((0, 0, 0)) == "Screen +12.0° · SNAP 5°"
    localizer.set_language(Language.SIMPLIFIED_CHINESE, persist=False)
    assert gizmo._format_value((0, 0, 0)) == "屏幕 +12.0° · 吸附 5°"
    gizmo._active = GizmoHandle.ROTATE_TRACKBALL
    gizmo._trackball_angles[:] = np.radians((12, -4))
    assert gizmo._format_value((0, 0, 0)) == "轨迹球 +12.0° -4.0° · 吸附 5°"
    gizmo._active = GizmoHandle.ROTATE_SCREEN
    gizmo._active_joint = SimpleNamespace(type="hinge", name="Screen")
    assert gizmo._format_value((0, 0, 0)) == "Screen +12.0° · 吸附 5°"
    localizer.set_language(Language.ENGLISH, persist=False)
    assert gizmo._format_value((0, 0, 0)) == "Screen +12.0° · SNAP 5°"


def test_dimension_feedback_translates_captions_without_changing_dimension_values():
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    gizmo = ObjectGizmo("dimensions", translate=localizer.text)
    dimensions = geometry_dimensions(MeshShape.BOX, (1, 2, 3))
    gizmo._dimension_start = SimpleNamespace(dimensions=dimensions)
    gizmo._active = GizmoHandle.X
    assert gizmo._format_dimension_value() == "宽度 2.000 m"
    gizmo._active = GizmoHandle.SCREEN
    assert gizmo._format_dimension_value((2, 4, 6)) == "缩放 1.000×"
    localizer.set_language(Language.ENGLISH, persist=False)
    gizmo._active = GizmoHandle.X
    assert gizmo._format_dimension_value() == "Width 2.000 m"
    assert dimensions.values == (2, 4, 6)


def test_every_primitive_dimension_caption_has_a_chinese_translation():
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    for shape in MeshShape:
        for size in ((1, 1, 1), (1, 2, 3)):
            dimensions = geometry_dimensions(shape, size)
            if dimensions is not None:
                assert localizer.text(dimensions.label) != dimensions.label, dimensions.label


def test_workspace_runtime_messages_and_about_copy_are_localized():
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    for message in (
        "Dimensions: select a geometry",
        "Simulation paused",
        "Simulation resumed",
        "Hide left panel",
        "Show left panel",
        "Hide right panel",
        "Show right panel",
        "Version",
        "Commit",
        "Backend-neutral 3D scene editing and simulation tools.",
        "Scene authoring, materials, cameras and lights.",
        "Joint posing, transform gizmos and simulation playback.",
        "OpenGL and bgfx rendering; Python APIs and remote scene sources.",
    ):
        assert localizer.text(message) != message
