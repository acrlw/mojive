"""Open real scene states needed by production-only parameter acceptance."""

from xml.etree import ElementTree as ET

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.assets import resolve
from mojive.commands import SelectNode
from mojive.types import CameraView


def write_audit_model(path):
    """Extend the joint example with sensor data and an editable spatial path."""
    tree = ET.parse(resolve("joint_types"))
    root, world = tree.getroot(), tree.getroot().find("worldbody")
    ET.SubElement(world, "site", name="audit_path_start", pos="0 0 1", size="0.025")
    body = world.find(".//body[@name='hinge_body']")
    ET.SubElement(body, "site", name="audit_path_end", pos="0 0 0.35", size="0.025")
    tendon = ET.SubElement(root, "tendon")
    spatial = ET.SubElement(tendon, "spatial", name="audit_spatial_path", width="0.01")
    ET.SubElement(spatial, "site", site="audit_path_start")
    ET.SubElement(spatial, "site", site="audit_path_end")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "jointpos", name="audit_hinge_position", joint="hinge_limited")
    ET.SubElement(sensors, "jointvel", name="audit_slide_velocity", joint="slide")
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="audit_pose_start", time="0.12")
    ET.SubElement(keys, "key", name="audit_pose_end", time="0.32")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="unicode")
    return path


def write_component_model(path):
    """Use an added model so the production hierarchy exposes a MODEL node."""
    path.write_text(
        '<mujoco model="audit_components"><worldbody>'
        '<site name="path_start" pos="0 0 1" size="0.025"/>'
        '<body name="path_body" pos="0 0 0.5">'
        '<geom type="box" size="0.1 0.1 0.1"/>'
        '<site name="path_end" pos="0 0 0.1" size="0.025"/>'
        '</body></worldbody><tendon><spatial name="spatial_path" width="0.01">'
        '<site site="path_start"/><site site="path_end"/>'
        "</spatial></tendon></mujoco>\n",
        encoding="utf-8",
    )
    return path


class ProductionScenes:
    """Keep fixture selection separate from production drafts and commands."""

    def __init__(self, audit, asset):
        self.audit, self.asset = audit, asset
        self.component_asset = write_component_model(asset.with_name("audit_components.xml"))
        self.empty = False
        self.component = None
        self.component_text_overrides = {}
        self.source_model_id = None
        self.default_source_model_id = None
        self.pending = False
        self.settings_size = None
        self.narrow_timeline_dock = None
        self.wide_inspector_dock = None
        self.popup_path = ()
        self.dialog_title = None
        self.choice_key = None
        audit.input.before_editor = self.close_production_popup
        audit.input.after_editor = self.open_production_popup
        audit.ensure_visible = self.ensure_visible

    def navigate(self, part):
        audit = self.audit
        app, session = audit.viewer.app, audit.viewer.session
        self.restore_timeline()
        self.restore_inspector()
        audit.scene_state = {"source": str(self.asset), "state": "paused joint scene"}
        if self.pending:
            app.model_edits.clear()
            self.pending = False
        self.close_production_popup()
        self.component = None
        self.component_text_overrides = {}
        self.source_model_id = None
        self.popup_path = ()
        self.dialog_title = None
        self.choice_key = None
        if part == "empty" and not self.empty:
            result = session.submit(cmd.NewScene())
            if not result.ok:
                raise RuntimeError("Could not open the empty scene: " + result.message)
            self.empty = True
            audit.input.frames(5)
        elif part != "empty" and self.empty:
            result = session.submit(cmd.LoadAsset(self.asset))
            if not result.ok:
                raise RuntimeError("Could not restore the audit scene: " + result.message)
            self.empty = False
            audit.input.frames(5)
            configure_joint_example(audit.viewer)
        if part in ("component_list", "component_path") and not any(
            node.type is NodeType.MODEL for node in session.nodes
        ):
            result = session.submit(cmd.AddSceneModel(self.component_asset, position=(10, 0, 0)))
            if not result.ok:
                raise RuntimeError("Could not add the component model: " + result.message)
            audit.input.frames(5)
        audit.bench.navigate(app.workspace)
        self.popup_path = app.workspace.menus.requested_path
        self.dialog_title = app.workspace.menus.dialog_pending
        if part == "empty":
            app.workspace.open["Inspector"] = False
            audit.scene_state = {"command": "NewScene", "state": "empty production viewport"}
        if part == "pending_edits":
            node = next(
                (
                    node
                    for node in session.nodes
                    if node.type is NodeType.GEOM and node.source_editable
                ),
                None,
            )
            if node is None:
                raise RuntimeError("The loaded scene has no editable model geometry")
            result = app.model_edits.stage(
                cmd.RenameModelElement(node.node_id, "audit_pending_geom")
            )
            if result is None or not result.ok:
                raise RuntimeError("Could not create a pending model-rename draft")
            self.pending = True
            audit.scene_state = {
                "state": "pending model geometry rename",
                "nodeId": node.node_id,
                "command": "RenameModelElement(audit_pending_geom)",
            }
        audit.input.frames(5)

        if part.startswith("output_"):
            app.output.clear()
            app.panels.get("output").auto_scroll = False
            for index, level in enumerate(("info", "warning", "error")):
                app.output.write(
                    f"Audit message {index + 1}: production output row",
                    level=level,
                    timestamp="12:00:00",
                )
            audit.input.frames(3)
            for window in imgui.get_current_context().windows:
                if window.active and "output_messages" in window.name:
                    imgui.internal.set_scroll_y(window, 0)
            audit.input.frames(2)
        if part == "button_row":
            geometry = next(
                (
                    node
                    for node in session.nodes
                    if node.type is NodeType.GEOM and node.source_editable
                ),
                None,
            )
            if geometry is None:
                raise RuntimeError(
                    "The audit scene has no actual editable geometry for action rows"
                )
            result = session.submit(cmd.SelectNode(geometry.node_id))
            if not result.ok:
                raise RuntimeError("Could not select the geometry action-row source")
            app.workspace.activate("Inspector")
            app.workspace.inspector_tab = "Material"
            audit.input.frames(4)
            ui = app.workspace.ui
            key = next(
                (key for key in ("section-材质", "section-MATERIAL") if key in ui.hits), None
            )
            if key is None:
                raise RuntimeError("The selected geometry has no actual Material detail section")
            x, y, width, height = ui.hits[key]
            audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
            audit.input.event("add_mouse_button_event", 0, True)
            audit.input.event("add_mouse_button_event", 0, False)
            audit.input.frames(3)
            audit.scene_state.update(
                state="geometry Material asset action rows", nodeId=geometry.node_id
            )

        if part == "timeline_details":
            panel = app.panels.get("keyframes")
            editor = panel.editor
            available = session.model_keyframes(editor.model_id)
            if not available:
                raise RuntimeError("The loaded model has no real keyframe to inspect")
            editor.selected_id = available[0].keyframe_id
            editor.selected_keyframes = {editor.selected_id}
            editor.edit_lane = "model"
            editor.selection_generation = -1
            audit.scene_state["selectedKeyframeId"] = editor.selected_id
        if part in ("visual_group", "visual_group_item"):
            for window in imgui.get_current_context().windows:
                if window.active and "Reference settings content" in window.name:
                    imgui.internal.set_scroll_y(window, window.scroll_max.y)
            audit.input.frames(3)
        if part == "component_list":
            self.expand_model_components()
        if part == "component_path":
            model = session.selected_node
            if model is None or model.type is not NodeType.MODEL:
                raise RuntimeError("The path fixture requires an actual selected MODEL node")
            candidates = session.model_components(model.model_id, "tendon")
            self.component = next((item for item in candidates if item.path), None)
            if self.component is None:
                raise RuntimeError("The loaded model has no editable tendon path")
            self.open_component_popup()
            audit.scene_state.update(
                componentModel=str(self.component_asset),
                modelId=model.model_id,
                componentId=self.component.component_id,
                pathItems=len(self.component.path),
                state="real model-component modal",
            )
        if part == "textarea":
            app.selected = "hinge_body"
            self.source_model_id = session.scene_models[0].model_id
            self.default_source_model_id = self.source_model_id
            self.popup_path = ()
            self.dialog_title = None
            self.open_model_source()
            audit.scene_state.update(state="real MJCF source modal", modelId=self.source_model_id)
        audit.input.frames(5)
        if part in ("select_option", "select_popup"):
            self.choice_key = "##ui_language"
            self.open_production_popup()
            audit.scene_state.update(state="expanded production language selector")

    def prepare_field(self, part, field):
        """Open the native layout state required by the field's documented condition."""
        if part == "workspace" and field == "bottomHeight":
            self.audit.viewer.app.workspace.activate("Timeline")
            self.audit.input.frames(4)
            self.audit.scene_state["bottomPanel"] = "Timeline"
        if part == "menu_item_content" and field in ("gap", "columnGap"):
            self.popup_path = ("Edit",)
            self.audit.viewer.app.workspace.menus.preview(self.popup_path)
            self.audit.input.frames(3)
        if part == "visual_group_item" and field in ("gap", "columnGap"):
            window = imgui.internal.find_window_by_name("Settings")
            if window is not None:
                scale = self.audit.viewer.window.style_scale
                imgui.set_window_pos(window.name, (300 * scale, 100 * scale))
                imgui.set_window_size(window.name, (740 * scale, 720 * scale))
                self.audit.input.frames(3)
            self.audit.scene_state["visualGroupTarget"] = "six-column numbered groups"
        else:
            self.audit.scene_state.pop("visualGroupTarget", None)
        wide_form = part in ("form_row", "vector_row", "switch") and field in (
            "gap",
            "labelWidth",
            "labelGap",
            "columnGap",
            "alignItems",
            "layout",
        )
        wide_buttons = part == "button_row" and field in ("gap", "columnGap", "align")
        if wide_form or wide_buttons:
            inspector = next(
                (
                    window
                    for window in imgui.get_current_context().windows
                    if window.active and window.name.rsplit("###", 1)[-1] == "Inspector"
                ),
                None,
            )
            if inspector is not None:
                if inspector.dock_id:
                    self.wide_inspector_dock = inspector.dock_id
                    imgui.internal.dock_context_queue_undock_window(
                        imgui.get_current_context(), inspector
                    )
                    self.audit.input.frames(2)
                scale = self.audit.viewer.window.style_scale
                imgui.set_window_pos(inspector.name, (360 * scale, 110 * scale))
                imgui.set_window_size(inspector.name, (520 * scale, 720 * scale))
                self.audit.input.frames(3)
                self.audit.scene_state["floatingInspectorWidth"] = 520 * scale
        if part == "panel" and field == "radius":
            self.audit.viewer.app.workspace.activate("Settings")
            self.audit.input.frames(4)
            self.audit.scene_state["floatingPanel"] = "Settings"
        if part in ("timeline_details", "range_bounds") and field == "rowGap":
            window = imgui.internal.find_window_by_name("Timeline")
            if window is None:
                raise RuntimeError("The real Timeline window is not open")
            if window.dock_id:
                self.narrow_timeline_dock = window.dock_id
                imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), window)
                self.audit.input.frames(2)
            scale = self.audit.viewer.window.style_scale
            imgui.set_window_pos("Timeline", (310 * scale, 160 * scale))
            width = 180 if part == "range_bounds" else 240
            imgui.set_window_size("Timeline", (width * scale, 720 * scale))
            self.audit.input.frames(3)
            self.audit.scene_state["timelineWidth"] = width * scale
        elif part in ("timeline_details", "range_bounds"):
            self.restore_timeline()
        if part in ("settings_nav", "settings_layout"):
            settings = imgui.internal.find_window_by_name("Settings")
            if settings is not None and settings.active:
                if self.settings_size is None:
                    self.settings_size = (settings.size.x, settings.size.y)
                narrow = (part, field) in (
                    ("settings_nav", "columnGap"),
                    ("settings_layout", "rowGap"),
                )
                width = 540 * self.audit.viewer.app.workspace.s if narrow else self.settings_size[0]
                imgui.set_window_size("Settings", (width, self.settings_size[1]))
                self.audit.input.frames(3)
                self.audit.scene_state.update(
                    settingsWidth=width,
                    settingsNavigation="multiple columns" if narrow else "side navigation",
                )
        if part != "textarea":
            return
        if field != "minHeight":
            if self.component is not None:
                self.close_production_popup()
                self.component = None
                self.component_text_overrides = {}
                self.source_model_id = self.default_source_model_id
                self.open_model_source()
            return
        self.close_production_popup()
        session = self.audit.viewer.session
        components = session.model_components(self.source_model_id, "actuator")
        self.component = next(
            (item for item in components if any(value.name == "gear" for value in item.fields)),
            None,
        )
        if self.component is None:
            raise RuntimeError("The real audit model has no editable actuator gear field")
        self.source_model_id = None
        self.component_text_overrides = {
            "gear": " ".join(("1.0000000000000000",) + ("0.0000000000000000",) * 5)
        }
        self.open_component_popup()
        self.audit.scene_state.update(
            state="real actuator gear multiline draft",
            componentId=self.component.component_id,
            draftFields=[(item.name, len(item.value)) for item in self.component.fields],
        )

    def restore_timeline(self):
        if self.narrow_timeline_dock is None:
            return
        imgui.internal.dock_builder_dock_window("Timeline", self.narrow_timeline_dock)
        imgui.internal.dock_builder_finish(self.audit.viewer.window.dockspace_id)
        self.narrow_timeline_dock = None
        self.audit.input.frames(3)

    def restore_inspector(self):
        if self.wide_inspector_dock is None:
            return
        # ImGui deletes an empty docking leaf after its last window undocks.
        # Restore through the application's real layout command instead of
        # referring to a deleted leaf identifier.
        self.audit.viewer.app.reset_layout(persist=False)
        self.wide_inspector_dock = None
        self.audit.input.frames(5)

    def close_production_popup(self):
        """Return keyboard focus to the workbench before each native editor gesture."""
        self.close_component_popup()
        self.close_model_source()
        audit = self.audit
        menu = audit.viewer.app.workspace.menus
        if self.choice_key is not None or self.dialog_title is not None:
            audit.input.key(imgui.Key.escape)
            audit.input.frames(2)
        elif self.popup_path:
            menu.preview(())
            audit.input.frames(2)

    def ensure_visible(self):
        if self.audit.scene_state.get("visualGroupTarget"):
            records = [
                record
                for record in self.audit.production_raw_records()
                if record["key"].startswith("visual-group-item-")
            ]
            if not records:
                raise RuntimeError("The actual settings page has no numbered visual groups")
            record = records[0]
            for window in imgui.get_current_context().windows:
                if window.active and window.id_ == record.get("windowId"):
                    target = window.scroll.y + record["hit"][1] - record["clip"][1] - 8
                    imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, target)))
            self.audit.input.frames(2)
            return
        if self.audit.bench.part == "button_row":
            records = [
                record
                for record in self.audit.production_raw_records()
                if "button_row" in record["parts"]
            ]
            if records:
                record = records[0]
                for window in imgui.get_current_context().windows:
                    if window.active and window.id_ == record.get("windowId"):
                        target = window.scroll.y + record["hit"][1] - record["clip"][1] - 8
                        imgui.internal.set_scroll_y(
                            window, max(0, min(window.scroll_max.y, target))
                        )
                self.audit.input.frames(2)
            return
        if self.audit.bench.part == "table":
            records = [
                record
                for record in self.audit.production_raw_records()
                if "table" in record["parts"]
            ]
            for record in records:
                for window in imgui.get_current_context().windows:
                    if window.active and window.id_ == record.get("windowId"):
                        target = window.scroll.y + record["hit"][1] - record["clip"][1] - 8
                        imgui.internal.set_scroll_y(
                            window, max(0, min(window.scroll_max.y, target))
                        )
            self.audit.input.frames(2)
            return
        if self.audit.bench.part == "textarea" and self.component is not None:
            records = [
                record
                for record in self.audit.production_raw_records()
                if "textarea" in record["parts"]
            ]
            for record in records:
                for window in imgui.get_current_context().windows:
                    if window.active and window.id_ == record.get("windowId"):
                        target = window.scroll.y + record["hit"][1] - record["clip"][1] - 8
                        imgui.internal.set_scroll_y(
                            window, max(0, min(window.scroll_max.y, target))
                        )
            self.audit.input.frames(2)
            return
        if self.audit.bench.part not in ("timeline_details", "visual_group", "visual_group_item"):
            return
        name = (
            "Timeline body"
            if self.audit.bench.part == "timeline_details"
            else "Reference settings content"
        )
        moved = False
        for window in imgui.get_current_context().windows:
            if (
                window.active
                and name in window.name
                and abs(window.scroll.y - window.scroll_max.y) > 0.1
            ):
                imgui.internal.set_scroll_y(window, window.scroll_max.y)
                moved = True
        if moved:
            self.audit.input.frames(2)

    def open_production_popup(self):
        self.open_component_popup()
        self.open_model_source()
        audit = self.audit
        menu = audit.viewer.app.workspace.menus
        if self.choice_key is not None:
            record = next(
                item for item in audit.production_raw_records() if item["key"] == self.choice_key
            )
            x, y, width, height = record["hit"]
            audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
            audit.input.event("add_mouse_button_event", 0, True)
            audit.input.event("add_mouse_button_event", 0, False)
            audit.input.frames(3)
        elif self.dialog_title is not None:
            menu.dialog_pending = self.dialog_title
            audit.input.frames(3)
        elif self.popup_path:
            menu.preview(self.popup_path)
            audit.input.frames(3)

    def expand_model_components(self):
        audit = self.audit
        ui = audit.viewer.app.workspace.ui
        candidates = [
            key
            for key in ui.hits
            if key.startswith("section-")
            and any(
                category in key.upper()
                for category in ("ACTUATOR", "TENDON", "SENSOR", "执行器", "肌腱", "传感器")
            )
        ]
        if not candidates:
            raise RuntimeError(
                "The selected model has no submitted component categories: "
                + repr(
                    {
                        "selection": audit.viewer.session.selected_node,
                        "sections": [key for key in ui.hits if "section" in key],
                    }
                )
            )
        key = next(
            (key for key in candidates if "TENDON" in key.upper() or "肌腱" in key),
            candidates[0],
        )
        x, y, width, height = ui.hits[key]
        for window in imgui.get_current_context().windows:
            if window.active and "Inspector content" in window.name:
                target = window.scroll.y + y - window.clip_rect.min.y - 8
                imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, target)))
        audit.input.frames(2)
        x, y, width, height = ui.hits[key]
        audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        audit.input.event("add_mouse_button_event", 0, True)
        audit.input.event("add_mouse_button_event", 0, False)
        audit.input.frames(2)

    def close_component_popup(self):
        """Dismiss the real modal through its native Cancel control before editing."""
        if self.component is None:
            return
        audit = self.audit
        ui = audit.viewer.app.workspace.ui
        keys = [
            key
            for key in ui.hits
            if key in ("Cancel", "取消", "Cancel##component", "取消##component")
        ]
        if not keys:
            # A popup can be below the viewport before its first layout pass.
            for window in imgui.get_current_context().windows:
                if window.active and "Model Component" in window.name:
                    imgui.internal.set_scroll_y(window, window.scroll_max.y)
            audit.input.frames(2)
            keys = [
                key
                for key in ui.hits
                if key in ("Cancel", "取消", "Cancel##component", "取消##component")
            ]
        if not keys:
            raise RuntimeError("The model-component modal has no reachable Cancel control")
        x, y, width, height = ui.hits[keys[0]]
        audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        audit.input.event("add_mouse_button_event", 0, True)
        audit.input.event("add_mouse_button_event", 0, False)
        audit.input.frames()

    def open_component_popup(self):
        if self.component is None:
            return
        audit = self.audit
        inspector = audit.viewer.app.panels.get("inspector")
        inspector._begin_component_edit(self.component)
        for item in inspector._component_fields:
            if item[0] in self.component_text_overrides:
                item[1] = self.component_text_overrides[item[0]]
        audit.input.frames(2)
        for window in imgui.get_current_context().windows:
            if window.active and "Model Component" in window.name:
                imgui.internal.set_scroll_y(
                    window, 0 if self.component_text_overrides else window.scroll_max.y
                )
        audit.input.frames(2)

    def close_model_source(self):
        if self.source_model_id is None:
            return
        audit = self.audit
        ui = audit.viewer.app.workspace.ui
        key = next((key for key in ui.hits if key in ("Cancel", "取消")), None)
        if key is None:
            # Already dismissed while switching to another production state.
            if not any(
                window.active and "MJCF Source" in window.name
                for window in imgui.get_current_context().windows
            ):
                return
            raise RuntimeError("The actual MJCF Source modal has no Cancel control")
        x, y, width, height = ui.hits[key]
        audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        audit.input.event("add_mouse_button_event", 0, True)
        audit.input.event("add_mouse_button_event", 0, False)
        audit.input.frames()

    def open_model_source(self):
        if self.source_model_id is None:
            return
        audit = self.audit
        inspector = audit.viewer.app.panels.get("inspector")
        inspector._source_model_id = self.source_model_id
        inspector._source_text = audit.viewer.session.adapter.scene_model_source(
            self.source_model_id
        )
        inspector._open_source_popup = True
        audit.input.frames(4)


def configure_joint_example(workspace):
    workspace.app.camera.adopt(
        CameraView(
            eye=np.array([4.3, -5.9, 4.2]),
            target=np.array([0, 0, 0.55]),
            fov_y=np.radians(42),
            near=0.05,
            far=100,
        )
    )
    workspace.app._initial_camera_set = True
    workspace.app.tool = "rotate"
    node = next((node for node in workspace.session.nodes if node.name == "hinge_body"), None)
    if node is not None:
        workspace.session.submit(SelectNode(node.node_id))
