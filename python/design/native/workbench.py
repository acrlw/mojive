"""Coordinate production component navigation and native parameter editing."""

import json
import os
from pathlib import Path

from mojive.design.help import get_part_help
from mojive.design.native.editor_chrome import EditorChrome
from mojive.design.native.editor_controls import EditorControls
from mojive.design.native.editor_pages import EditorPages
from mojive.design.native.selection import selection_parts


class NativeWorkbench(EditorChrome, EditorControls, EditorPages):
    def __init__(self, editor, runtime, watcher, *, width=460):
        self.editor, self.runtime, self.watcher = editor, runtime, watcher
        editor.enable_background_sync()
        from mojive.design.native.specimens import ComponentSpecimens

        self.specimens = ComponentSpecimens(runtime)
        self.width = width
        self.part = "timeline_toolbar"
        self.page = "交互原型"
        self.mode = "部件参数"
        self.category = "全部部件"
        self.search = ""
        self.field_search = ""
        self.field_group = "全部参数"
        self.family = "inspector"
        self.show_bounds = True
        self.snap = True
        self.snap_step = 2
        self.file_path = str(self.editor.saved_path.with_name("exported-spec.json"))
        self.hits = {}
        self.native_instances = []
        self.frame_index = 0
        self._started = False
        self.detached = False
        self.requested_detached = None
        self.editor_rect = None
        self._requested_navigation = None
        self.pick = True
        self.pick_pressed = False
        self.selected_field = None
        self.selection_parts = []
        self.field_highlights = []
        self._viewport_rect = None
        self.last_navigation = ""
        self.actions = []
        self.ready_path = None
        self._ready_written = False
        self._catalogue_scroll_pending = True
        self._navigation_scroll = None

    def select_sample_body(self, workspace, name="hinge_body"):
        """Resolve the reference link against the current authoring structure."""
        from mojive import commands as cmd
        from mojive.adapters.base import NodeType

        session = workspace.app.session
        matches = [
            node
            for node in session.nodes
            if node.type is NodeType.LINK and (node.source_name or node.name) == name
        ]
        if len(matches) == 1:
            session.submit(cmd.SelectNode(matches[0].node_id))

    def request_navigation(self, workspace, part=None):
        if self.detached:
            self._requested_navigation = part or self.part
        else:
            self.navigate(workspace, part)

    def navigate(self, workspace, part=None):
        part = part or self.part
        if part not in self.selection_parts:
            self.selection_parts = []
        definition = self.runtime.catalog[part]
        if "controlGroup" in definition:
            part = definition["controlGroup"]
            definition = self.runtime.catalog[part]
        area = get_part_help(part)["preview"]["area"]
        part = definition.get("basePart", part)
        area = {
            "input_text": "设置",
            "checkbox": "设置",
            "input_shell": "相机",
            "badge": "场景",
            "sensor": "传感器",
            "material": "材质库",
            "material_list": "材质库",
            "data_list": "统计",
            "empty": "检查器",
            "vector_heading": "姿态",
            "range_bounds": "时间轴",
            "visual_group": "设置",
            "visual_group_item": "设置",
            "file_receipt": "视口",
            "link_button": "视口",
            "table": "统计",
        }.get(part, area)
        workspace.menus.preview(())
        for panel_id, panel_area in (("help", "帮助"), ("info", "信息")):
            panel = workspace.app.panels.get(panel_id)
            if panel is not None:
                panel.open = area == panel_area
        self.show_sample = False
        if area != "设置" and hasattr(workspace, "open"):
            workspace.open["Settings"] = False
        workspace.app._model_drop_notice = ""
        workspace.app._model_drop_notice_until = 0.0
        if area != "相机预览":
            workspace.app.camera_preview.set_enabled(False)
        if part == "notice":
            workspace.app.output.publish("操作状态示例", duration=600)
            workspace.open["Inspector"] = False
        if area in ("检查器", "材质", "颜色编辑", "相机预览"):
            if area != "相机预览":
                # Material controls require a geometry; inertia uses a body.
                geometries = [
                    entity for entity in workspace.app.entities if entity["type"] == "geom"
                ]
                if part in ("component_list", "component_path"):
                    pass
                elif geometries and area in ("材质", "颜色编辑"):
                    workspace.app.selected = geometries[0]["name"]
                else:
                    self.select_sample_body(workspace)
                workspace.activate("Inspector")
                workspace.pinned = None
                workspace.inspector_tab = "Material" if area in ("材质", "颜色编辑") else "Physics"
                if part in ("component_list", "component_path"):
                    from mojive import commands as cmd
                    from mojive.adapters.base import NodeType

                    session = workspace.app.session
                    models = [node for node in session.nodes if node.type is NodeType.MODEL]
                    if len(models) == 1:
                        session.submit(cmd.SelectNode(models[0].node_id))
                    workspace.inspector_tab = "Properties"
                if part == "empty":
                    from mojive import commands as cmd
                    from mojive.adapters.base import NodeType

                    world = next(
                        (
                            node
                            for node in workspace.app.session.nodes
                            if node.type is NodeType.WORLD
                        ),
                        None,
                    )
                    if world is not None:
                        workspace.app.session.submit(cmd.SelectNode(world.node_id))
                    workspace.inspector_tab = "Properties"
            else:
                from mojive.adapters.base import NodeType

                cameras = [
                    node for node in workspace.app.session.nodes if node.type is NodeType.CAMERA
                ]
                if cameras:
                    workspace.app.selected = cameras[0].name
                    workspace.app.camera_preview.set_enabled(True)
                workspace.activate("Cameras")
        elif area == "相机":
            workspace.activate("Cameras")
        elif area == "姿态":
            self.select_sample_body(
                workspace, "free_body" if part == "vector_heading" else "hinge_body"
            )
            workspace.app.panels.get("joints")._search = ""
            if part == "pose_tools" and not hasattr(workspace, "navigation"):
                from mojive.ui.workspace.navigation import Navigation

                workspace.navigation = Navigation(workspace)
            if hasattr(workspace, "navigation"):
                workspace.navigation.pose_search_visible = part == "pose_tools"
            workspace.activate("Pose")
        elif area == "传感器":
            workspace.activate("Sensors")
        elif area == "材质库":
            workspace.activate("Assets")
        elif area == "控制":
            workspace.activate("Control")
        elif area == "时间轴":
            workspace.activate("Timeline")
        elif area == "输出":
            workspace.activate("Output")
        elif area == "统计":
            workspace.activate("Statistics")
        elif area == "图层":
            workspace.activate("Layers")
        elif area == "曲线":
            workspace.activate("Plot")
        elif area in ("帮助", "信息"):
            pass
        elif area == "设置":
            category = (
                "Interaction"
                if part == "input_text"
                else "MuJoCo Visuals"
                if part in ("visual_group", "visual_group_item")
                else "General"
            )
            owner = definition.get("groupOwner", "")
            if owner.startswith("Settings/"):
                category = owner.removeprefix("Settings/")
            workspace.settings.show_category(category)
            workspace.activate("Settings")
        elif area == "智能体":
            self.show_sample = True
        elif area == "顶栏菜单":
            workspace.menus.preview(("Edit" if part == "menu_item_content" else "File",))
        elif area == "对话框":
            workspace.menus.dialog_pending = "Interaction guide"
        elif area in ("场景", "活动栏"):
            workspace.activate("Scene")
            if part == "search":
                workspace.search_visible = True
                workspace.query = ""
        elif area in ("工作区", "顶栏", "状态栏", "视口"):
            workspace.activate("Scene")
            if part != "notice":
                self.select_sample_body(workspace)
        if part == "bottom_header":
            workspace.collapse_bottom()
        self._receipt_pending = 3 if part in ("file_receipt", "link_button") else 0
        if area in ("检查器", "材质"):
            self._navigation_scroll = ("Inspector content", part in ("switch", "component_list"), 3)
        elif area == "设置":
            self._navigation_scroll = ("Reference settings content", part == "input_text", 3)
        else:
            self._navigation_scroll = None
        if area == "颜色编辑":
            if part in ("rgba", "color_channels"):
                workspace.menus.color_editor.mode = 2
            workspace.menus.preview(("Color",))
        self.last_navigation = area

    def poll_editor(self):
        from imgui_bundle import imgui as g

        try:
            if self.editor.pending_sync and not g.is_mouse_down(0):
                self.editor.end_gesture()
            self.editor.poll_sync()
        except OSError as error:
            self.editor.error = str(error)
        self.editor.poll()
        self.hits.clear()

    def begin_frame(self, workspace):
        if not self.detached:
            self.poll_editor()
        if not self._started or self._requested_navigation is not None:
            self.navigate(workspace, self._requested_navigation)
            self._requested_navigation = None
            self._started = True
        self.frame_index += 1
        if getattr(self, "_receipt_pending", 0):
            self._receipt_pending -= 1
            if self._receipt_pending == 0:
                path = self.editor.saved_path
                workspace.app.output.publish(
                    "规范文件：" + path.name, duration=600.0, copy_text=str(path)
                )
        pending = getattr(self, "_navigation_scroll", None)
        if pending is not None:
            from imgui_bundle import imgui as g

            name, bottom, attempts = pending
            for window in g.get_current_context().windows:
                if (window.active or window.was_active) and name in str(window.name):
                    g.internal.set_scroll_y(window, window.scroll_max.y if bottom else 0.0)
            self._navigation_scroll = (name, bottom, attempts - 1) if attempts > 1 else None

    def draw(self, workspace):
        from imgui_bundle import imgui as g

        if self.detached:
            return
        visibility = workspace.app.scrollbar_visibility
        with (
            self.editor_scope(workspace),
            visibility.panel_scope("PythonDesignWorkbench", window_name="PythonDesignWorkbench"),
        ):
            flags = (
                g.WindowFlags_.no_decoration
                | g.WindowFlags_.no_docking
                | g.WindowFlags_.no_move
                | g.WindowFlags_.no_saved_settings
            )
            sidebar_width = min(
                self.width * workspace.s,
                g.get_main_viewport().size.x * 0.48,
            )
            g.push_style_var(
                g.StyleVar_.window_min_size,
                (min(300 * workspace.s, sidebar_width), 100 * workspace.s),
            )
            visible = g.internal.begin_viewport_side_bar(
                "PythonDesignWorkbench", g.get_main_viewport(), g.Dir.right, sidebar_width, flags
            )
            visibility.bind_current_window()
            g.pop_style_var()
            try:
                if visible:
                    self._viewport_rect = (*g.get_window_pos(), *g.get_window_size())
                    self.draw_contents(workspace)
            finally:
                g.end()

    def claim_production_pointer(self):
        """Reserve the complete inspection click before production controls run."""
        from imgui_bundle import imgui as g

        io = g.get_io()
        hovered = g.get_current_context().hovered_window
        in_editor = hovered is not None and self._editor_window(hovered)
        owner = g.get_id("##design-workbench-pointer")
        self.pick_pressed = bool(self.pick and io.key_alt and io.mouse_clicked[0] and not in_editor)
        if self.pick_pressed:
            g.internal.set_key_owner(
                g.Key.mouse_left, owner, g.internal.InputFlagsPrivate_.lock_until_release
            )

    def draw_overlay(self, workspace, *, extra_records=()):
        from imgui_bundle import imgui as g

        viewport = g.get_main_viewport()
        preview_clip = (
            viewport.pos.x,
            viewport.pos.y,
            viewport.pos.x + viewport.size.x,
            viewport.pos.y + viewport.size.y,
        )
        primitives = getattr(workspace.ui, "design_primitives", None)
        records = list(primitives.observed) if primitives else []
        records.extend(self.runtime.observed)
        records.extend(extra_records)
        all_records = records
        picking = self.pick_pressed
        shared_selection = self.selected_field and self.selected_field[0] != "parts"
        parameter_part = (
            self.selected_field[1]
            if self.selected_field and self.selected_field[0] == "parts"
            else self.part
        )
        if shared_selection and not picking:
            from mojive.design.native.parameter_overlay import global_parameter_records

            records = global_parameter_records(records, self.selected_field, self.runtime)
        visible_records = []
        for record in records:
            if (
                not picking
                and not shared_selection
                and not any(part in record["parts"] for part in (self.part, parameter_part))
            ):
                continue
            clip = record.get("clip") or preview_clip
            clip = (
                max(clip[0], preview_clip[0]),
                max(clip[1], preview_clip[1]),
                min(clip[2], preview_clip[2]),
                min(clip[3], preview_clip[3]),
            )
            window = g.internal.find_window_by_id(record.get("windowId", 0))
            excluded = None if self.window_above_workbench(window) else self._viewport_rect
            segments = self.visible_native_rects(record["hit"], clip, excluded)
            if segments:
                visible = max(segments, key=lambda rect: rect[2] * rect[3])
                visible_records.append(
                    {
                        **record,
                        "clip": clip,
                        "visibleHit": visible,
                        "visibleSegments": segments,
                        "overlayExclusion": excluded,
                    }
                )
        records = visible_records
        if self.ready_path is not None and not self._ready_written and self.frame_index >= 4:
            import glfw

            ready = {
                "pid": os.getpid(),
                "page": self.page,
                "windowTitle": workspace.app._window_title,
                "windowShown": bool(glfw.get_window_attrib(workspace.window._window, glfw.VISIBLE)),
                "workbenchRect": list(self.editor_rect if self.detached else self._viewport_rect),
                "detached": self.detached,
                "previewSize": list(workspace.window.size_points),
                "parts": len(self.editor.catalog),
                "fields": sum(len(p["fields"]) for p in self.editor.catalog.values()),
                "savedPath": str(self.editor.saved_path),
                "livePath": str(self.editor.live_path),
                "specificationPages": ["交互原型", "设计规范"],
            }
            ready_path = Path(self.ready_path)
            temporary = ready_path.with_name(
                "." + ready_path.name + "." + str(os.getpid()) + ".tmp"
            )
            temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
            os.replace(temporary, ready_path)
            self._ready_written = True
            print(
                "Native design workbench ready: interactive prototype and design specifications.",
                flush=True,
            )
        if picking:
            mouse = g.get_io().mouse_pos
            hovered = g.get_current_context().hovered_window
            window_ids = set()
            while hovered is not None:
                window_ids.add(hovered.id_)
                if hovered.flags & g.WindowFlags_.popup:
                    break
                hovered = hovered.parent_window
            candidates = [
                r
                for r in records
                if (
                    not window_ids
                    or r.get("windowId") in window_ids
                    or r.get("drawListSource") == "foreground"
                )
                and any(
                    rect[0] <= mouse.x <= rect[0] + rect[2]
                    and rect[1] <= mouse.y <= rect[1] + rect[3]
                    for rect in r["visibleSegments"]
                )
            ]
            if candidates:
                self.selection_parts = selection_parts(candidates, self.editor.catalog)
                self.part = self.selection_parts[0]
                self.selected_field = None
                shared_selection = False
                self.category, self.search, self.mode = "全部部件", "", "部件参数"
                self.field_search = ""
                self.field_group = "全部参数"
                self._catalogue_scroll_pending = True
        seen = set()
        self.native_instances = []
        for record in records:
            if not shared_selection and self.part not in record["parts"]:
                continue
            signature = tuple(round(v, 2) for v in record["hit"])
            if signature in seen:
                continue
            seen.add(signature)
            self.native_instances.append(record)
        self.field_highlights = []
        if self.mode != "部件参数" and not self.selected_field:
            return
        draw = g.get_foreground_draw_list()
        color = g.get_color_u32((0.59, 0.77, 0.53, 0.6))
        center = g.get_color_u32((0.54, 0.67, 0.77, 0.35))
        for record in self.native_instances if self.show_bounds else ():
            x, y, w, h = record["hit"]
            if w <= 0 or h <= 0:
                continue
            for sx, sy, sw, sh in record["visibleSegments"]:
                draw.push_clip_rect((sx, sy), (sx + sw, sy + sh), True)
                draw.add_rect((x, y), (x + w, y + h), color, thickness=1)
                draw.add_line((x + w / 2, y), (x + w / 2, y + h), center)
                draw.add_line((x, y + h / 2), (x + w, y + h / 2), center)
                draw.pop_clip_rect()
        selected = self.selected_field
        parameter_instances = (
            self.native_instances
            if shared_selection or parameter_part == self.part
            else [record for record in records if parameter_part in record["parts"]]
        )
        if selected and selected[-1] in ("gap", "rowGap", "columnGap", "labelGap", "headerGap"):
            from mojive.design.native.parameter_overlay import complete_child_geometry

            parameter_instances = complete_child_geometry(parameter_instances, all_records)
            if parameter_part == self.part:
                self.native_instances = parameter_instances
        if selected:
            from mojive.design.native.parameter_overlay import draw_parameter_overlay

            self.field_highlights = draw_parameter_overlay(
                draw, parameter_instances, selected[-1], workspace.s, self._viewport_rect
            )

    @staticmethod
    def window_above_workbench(window):
        """Native popups render above the sidebar, including their child windows."""
        from imgui_bundle import imgui as g

        popup = False
        while window is not None:
            if window.name.split("###")[-1] == "PythonDesignWorkbench":
                return False
            popup |= bool(window.flags & g.WindowFlags_.popup)
            window = window.parent_window
        return popup

    @staticmethod
    def visible_rect(rect, clip):
        left, top = max(rect[0], clip[0]), max(rect[1], clip[1])
        right, bottom = min(rect[0] + rect[2], clip[2]), min(rect[1] + rect[3], clip[3])
        return (left, top, right - left, bottom - top) if right > left and bottom > top else None

    @staticmethod
    def visible_native_rects(rect, clip, workbench=None):
        """Subtract the actual editor window while retaining visible native chrome."""
        visible = NativeWorkbench.visible_rect(rect, clip)
        if visible is None:
            return []
        if workbench is None:
            return [visible]
        x, y, w, h = visible
        cut = NativeWorkbench.visible_rect(workbench, (x, y, x + w, y + h))
        if cut is None:
            return [visible]
        cx, cy, cw, ch = cut
        candidates = (
            (x, y, w, cy - y),
            (x, cy, cx - x, ch),
            (cx + cw, cy, x + w - cx - cw, ch),
            (x, cy + ch, w, y + h - cy - ch),
        )
        return [piece for piece in candidates if piece[2] > 0 and piece[3] > 0]

    def report(self):
        return {
            "nativePreview": True,
            "savedPath": str(self.editor.saved_path),
            "livePath": str(self.editor.live_path),
            "part": self.part,
            "page": self.page,
            "partsAvailable": len(self.editor.catalog),
            "fieldsAvailable": sum(len(p["fields"]) for p in self.editor.catalog.values()),
            "writes": self.editor.writes,
            "actions": self.actions,
            "overrides": self.editor.overrides(),
            "error": self.editor.error,
        }
