"""Inspector: core."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    BodyProperties,
    FrameNeeds,
    GeometryAdvancedProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    ModelComponentInfo,
    NodeType,
    SceneNode,
    SiteProperties,
)
from mojive.ui.controls import pill_label
from mojive.ui.draw2d import ImguiDraw2D
from mojive.ui.panels import (
    Panel,
    PanelContext,
    pointer_hint,
    pointer_pressed,
)
from mojive.ui.panels.filters import _tint, node_filter_color
from mojive.ui.pointer_bindings import PointerAction

from .environment import _Environment
from .geometry import _Geometry
from .model import _Model
from .physics import _Physics
from .transform import _Transform


class InspectorPanel(_Model, _Transform, _Physics, _Geometry, _Environment, Panel):
    id = "inspector"

    name = "Inspector"

    default_open = True

    shortcut = "F4"

    closable = False

    def __init__(self) -> None:
        super().__init__()
        self.show_transform = True
        self._transform_velocity = False
        self.show_velocity = False
        self._rotation_node = -1
        self._rotation_euler = np.zeros(3, np.float64)
        self._rotation_matrix = np.eye(3, dtype=np.float64)
        self._edit_transaction = False
        self._model_name_source: tuple[int, int, int, str] | None = None
        self._model_name = ""
        self._renaming = False
        self._rename_focus = False
        self._rename_active = False
        self._rename_generation = -1
        self._name_draw_frame = -1
        self._rename_model_element = False
        self._rename_error = ""
        self._source_model_id = -1
        self._source_text = ""
        self._source_error = ""
        self._open_source_popup = False
        self._component_cache_generation = -1
        self._component_cache_model = -1
        self._component_cache: dict[str, tuple[ModelComponentInfo, ...]] = {}
        self._component_presets: dict[str, tuple[str, ...]] = {}
        self._component_counts: dict[str, int] = {}
        self._component_edit: ModelComponentInfo | None = None
        self._component_name = ""
        self._component_fields: list[list[str]] = []
        self._component_path: list[tuple[str, list[list[str]]]] = []
        self._component_path_choices: list[dict[str, tuple[str, ...]]] = []
        self._component_path_presets = ()
        self._component_error = ""
        self._open_component_popup = False
        self._model_transform_model = -1
        self._model_transform_generation = -1
        self._model_transform_position = np.zeros(3, np.float32)
        self._model_transform_euler = np.zeros(3, np.float64)
        self._camera_sync_source = -1
        self._camera_angular_degrees = True
        self._camera_initial_key = None
        self._camera_initial_view = None
        self._body_property_node = -1
        self._body_property_generation = -1
        self._body_property_edit: BodyProperties | None = None
        self._body_inertial_euler = np.zeros(3, np.float64)
        self._body_property_error = ""
        self._geometry_advanced_node = -1
        self._geometry_advanced_generation = -1
        self._geometry_advanced_edit: GeometryAdvancedProperties | None = None
        self._geometry_advanced_error = ""
        self._geometry_shape_node = -1
        self._geometry_shape_generation = -1
        self._geometry_shape_edit: GeometryShapeProperties | None = None
        self._geometry_shape_error = ""
        self._joint_advanced_id = -1
        self._joint_advanced_generation = -1
        self._joint_advanced_edit: JointAdvancedProperties | None = None
        self._joint_advanced_error = ""
        self._site_property_node = -1
        self._site_property_generation = -1
        self._site_property_edit: SiteProperties | None = None
        self._site_property_error = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(
            poses=True,
            qvel=(self.show_transform and self._transform_velocity) or self.show_velocity,
        )

    def finish_frame(self, ctx: PanelContext) -> None:
        if self._renaming and (
            not self.open or not self.enabled or self._name_draw_frame != imgui.get_frame_count()
        ):
            self._finish_name_edit(ctx)
        gizmo = ctx.gizmo
        if gizmo is not None and gizmo.model_placement_model_id >= 0:
            node = ctx.session.selected_node
            model_id = gizmo.model_placement_model_id
            invalid_placement = (
                not ctx.session.paused
                or node is None
                or node.type is not NodeType.MODEL
                or node.model_id != model_id
                or not gizmo.model_placement_active(ctx.session, model_id)
            )
            if invalid_placement and gizmo.cancel_model_placement(ctx.session).ok:
                self._model_transform_model = -1
        if self._edit_transaction and not imgui.is_any_item_active():
            ctx.submit(cmd.EndEditTransaction())
            self._edit_transaction = False

    def _submit_edit(self, ctx: PanelContext, command) -> None:
        if imgui.is_any_item_active() and not self._edit_transaction and not ctx.session.editing:
            result = ctx.submit(cmd.BeginEditTransaction("Inspector edit"))
            self._edit_transaction = result.ok
        ctx.submit(command)

    def draw(self, ctx: PanelContext) -> None:
        self._draw_model_source(ctx)
        self._draw_component_editor(ctx)
        s = ctx.session
        node = s.selected_node
        if self._renaming and (
            node is None or self._model_name_source != self._name_identity(node)
        ):
            self._finish_name_edit(ctx)
        if node is None:
            self._transform_velocity = False
            imgui.text_disabled(ctx.tr("nothing selected"))
            imgui.set_item_tooltip(ctx.tr("click an object in the viewport or the Hierarchy panel"))
            return

        self._name_editor(ctx, node)
        self._identity(ctx, node)
        if node.type is NodeType.MODEL:
            self._model(ctx, node)
            return
        if node.type is NodeType.LIGHT:
            self._light(ctx, node)
            return
        if node.type is NodeType.CAMERA:
            self._camera(ctx, node)
            return
        if node.type is NodeType.ENVIRONMENT:
            self._environment(ctx)
            return
        if node.type is NodeType.JOINT:
            self._joint(ctx, node)
            return
        self._transform(ctx, node)
        self._gizmo_reason(ctx, node)
        self._velocity(ctx, node)
        if node.type in (NodeType.LINK, NodeType.ROBOT):
            self._body_properties(ctx, node)
        if node.type is NodeType.SITE:
            self._site_properties(ctx, node)
        self._material(ctx, node)

    @staticmethod
    def _name_identity(node: SceneNode) -> tuple[int, int, int, str]:
        return node.node_id, node.model_id, node.object_id, node.name

    def _finish_name_edit(self, ctx: PanelContext, *, cancel: bool = False) -> None:
        source = self._model_name_source
        self._renaming = self._rename_focus = self._rename_active = False
        if cancel or source is None:
            return
        node = ctx.session.node(source[0])
        # A rebuild may recycle node IDs. Only commit against the structure
        # and entity from which this edit started, including on selection blur.
        if (
            ctx.session.structure_generation != self._rename_generation
            or node is None
            or self._name_identity(node) != source
        ):
            return
        value = self._model_name.strip()
        if not value or value == node.name.removeprefix(f"opengl_{node.model_id}_"):
            return
        command = (
            cmd.RenameModelElement(node.node_id, value)
            if self._rename_model_element
            else cmd.RenameSceneEntity(node.object_id, value)
        )

        def completed(result):
            if self._model_name_source == source:
                self._rename_error = "" if result.ok else result.message

        if self._rename_model_element:
            ctx.submit_model_edit(command, completed)
        else:
            completed(ctx.submit(command))

    def _name_editor(self, ctx: PanelContext, node: SceneNode) -> None:
        self._name_draw_frame = imgui.get_frame_count()
        model_element = node.source_editable and node.type not in (NodeType.WORLD, NodeType.MODEL)
        scene_entity = (
            node.model_id < 0
            and node.object_id > 0
            and node.type in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
            and ctx.session.adapter.caps.scene_authoring
        )
        editable = model_element or scene_entity
        identity = self._name_identity(node)
        if not self._renaming and self._model_name_source != identity:
            self._model_name_source = identity
            self._model_name = node.name.removeprefix(f"opengl_{node.model_id}_")
            self._rename_error = ""
        height = imgui.get_frame_height()
        available = max(1.0, imgui.get_content_region_avail().x)
        type_label = node.type.value
        badge_width = imgui.calc_text_size(type_label).x + 12 * ctx.style_scale
        inline = available > badge_width + 90 * ctx.style_scale
        name_width = (
            max(1.0, available - badge_width - imgui.get_style().item_spacing.x)
            if inline
            else available
        )
        if self._renaming:
            if self._rename_focus:
                imgui.set_keyboard_focus_here()
                self._rename_focus = False
            imgui.set_next_item_width(name_width)
            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
            imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
            imgui.push_style_color(imgui.Col_.nav_cursor, imgui.ImVec4(0, 0, 0, 0))
            entered, self._model_name = imgui.input_text(
                "##entity_name",
                self._model_name,
                imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
            )
            active = imgui.is_item_active()
            blur = imgui.is_item_deactivated() or (self._rename_active and not active)
            self._rename_active = active
            cancel = active and imgui.is_key_pressed(imgui.Key.escape, False)
            imgui.pop_style_color(2)
            imgui.pop_style_var()
            if entered or blur or cancel:
                self._finish_name_edit(ctx, cancel=cancel)
        else:
            origin = imgui.get_cursor_screen_pos()
            width = name_width
            imgui.invisible_button("##entity_name_label", imgui.ImVec2(width, height))
            hovered = imgui.is_item_hovered()
            if editable and hovered and pointer_pressed(ctx, PointerAction.NAME_EDIT):
                self._renaming = self._rename_focus = True
                self._rename_active = False
                self._rename_generation = ctx.session.structure_generation
                self._rename_model_element = model_element
                self._rename_error = ""
            draw = ImguiDraw2D()
            imgui.push_clip_rect(origin, (origin.x + width, origin.y + height), True)
            draw.text(
                (
                    origin.x + imgui.get_style().frame_padding.x,
                    origin.y + imgui.get_style().frame_padding.y,
                ),
                ctx.theme.text,
                node.name.removeprefix(f"opengl_{node.model_id}_") or "?",
            )
            imgui.pop_clip_rect()
            if hovered:
                hint = (
                    pointer_hint(ctx, PointerAction.NAME_EDIT, ctx.tr("Rename")) if editable else ""
                )
                imgui.set_tooltip(node.name + ("\n" + hint if hint else ""))
        if self._rename_error:
            imgui.text_wrapped(self._rename_error)
        if inline:
            imgui.same_line()
        accent = node_filter_color(ctx.theme, node.type)
        pill_label(
            type_label,
            width=badge_width,
            height=height,
            background=_tint(ctx.theme.bg_frame, accent, 0.22),
            color=accent,
        )
