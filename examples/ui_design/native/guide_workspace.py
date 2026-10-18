"""Production panel presenters hosted by an isolated, local example session."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from xml.etree import ElementTree

from imgui_bundle import imgui as g

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.groups import group_scope
from mojive.design.native.presentation import PANEL_FAMILIES
from mojive.session import Session
from mojive.ui.preferences import Preferences
from mojive.ui.window_native import NativeWindow
from mojive.ui.workspace.app import WorkspaceApp
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.panel_geometry import floating_panel_geometry
from mojive.ui.workspace.panel_host import draw_content
from mojive.ui.workspace.viewport import transport
from mojive.ui.workspace.viewport_controls import BACKGROUND, camera_controls, toolbox
from mojive.ui.workspace.window_chrome import panel_content_corners, surface_ink


class GuideWorkspace:
    """Reuse native headers, controls and command routing without persistent edits."""

    def __init__(self, guide):
        self.guide = guide
        window = guide.window
        backend = session = None
        try:
            if isinstance(window, NativeWindow):
                from mojive.render.native.backend import NativeBackend

                backend = NativeBackend(640, 480)
            else:
                from mojive.render.opengl.backend import OpenGLBackend

                window.make_current()
                backend = OpenGLBackend(None, 640, 480)
            # The bundled scene is copied before adding diagnostic sample channels.
            document = ElementTree.parse(resolve("joint_types"))
            sensors = ElementTree.SubElement(document.getroot(), "sensor")
            for kind, name, joint in (
                ("jointpos", "hinge_position", "hinge_limited"),
                ("jointvel", "hinge_velocity", "hinge_limited"),
                ("jointpos", "slide_position", "slide"),
            ):
                ElementTree.SubElement(sensors, kind, name=name, joint=joint)
            ElementTree.SubElement(
                sensors, "framepos", name="hinge_position_xyz", objtype="body", objname="hinge_body"
            )
            ElementTree.SubElement(
                sensors, "user", name="component_readings", dim="14", needstage="pos"
            )
            asset = guide.output / "sample-scene.xml"
            document.write(asset, encoding="unicode")
            session = Session(MuJoCoAdapter(asset), asset)
            session.submit(cmd.Pause())
            session.tick(
                FrameNeeds(qpos=True, qvel=True, actuator=True, sensors=True, joint_frames=True), 0
            )
            typography = guide.ui.typography
            self.app = WorkspaceApp(
                session,
                backend,
                window,
                ui=guide.ui,
                design=guide.runtime,
                config=ViewerConfig(
                    layout=LayoutConfig(persistence=False),
                    debug_server=False,
                    threaded_physics=False,
                ),
            )
        except Exception:
            if backend is not None:
                backend.release()
            if session is not None:
                session.release()
            raise
        guide.ui.typography = typography
        # Settings retain the real setters and validation with an in-memory destination.
        self.app.preferences = Preferences(values=deepcopy(self.app.preferences.values))
        self.app.localizer = guide.localizer
        self.app.workspace.ui.translate = guide.tr
        self.app.workspace.activate = self.activate
        self.app.workspace.menus.dispatch = self.dispatch
        self.app.start_take_video = self.preview_video
        self.app.selected = "hinge_body"
        self.app._sync_structure()
        self.app._frame_scene(animate=False)
        self.app.projection.refresh_document()
        self.app.panels.get("camera").bookmark_directory = guide.output / "camera-bookmarks"
        self.restore_output()
        self._inspector_selection = None
        self._settings_selection = None
        self.context = None
        self._drawn = set()

    def activate(self, name):
        page = "Curves" if name == "Plot" else name
        self.app.workspace.open[name] = True
        self.guide.page = page

    def dispatch(self, action):
        from mojive.ui.workspace.menu_actions import dispatch

        prefix = action.partition(":")[0]
        if prefix in ("file", "capture", "record-surface", "layout", "dialog"):
            self.guide.message = self.guide.tr("Preview action: {action}", action=action)
            return
        dispatch(self.app.workspace.menus, action)

    def preview_action(self, label, *args, **kwargs):
        self.guide.message = self.guide.tr("Preview action: {action}", action=self.guide.tr(label))

    def apply_queued_edits(self):
        app = self.app
        app._start_pending_model_edits()
        while app._model_load_queue:
            job = app._model_load_queue.pop(0)
            if job.action != "edit":
                result = cmd.CommandResult.bad("File operations are preview actions in this guide.")
            else:
                try:
                    result = app._load_model(job.command)
                except Exception as error:
                    result = cmd.CommandResult.bad(str(error))
            app.session._record_result(result)
            if job.completed is not None:
                job.completed(result)
            if not result.ok:
                self.guide.message = result.message
        app._model_prepared_resources = None

    def preview_video(self):
        self.guide.message = self.guide.tr(
            "Video export settings are available in the native application."
        )

    def restore_output(self):
        self.app.output.clear()
        for level, text in (
            ("info", "Scene loaded"),
            ("warning", "Material texture missing; base color is active."),
            ("error", "Position update failed; previous value retained."),
        ):
            self.app.output.write(text, level=level, timestamp="15:42:17")

    def prepare_frame(self):
        guide, app, workspace = self.guide, self.app, self.app.workspace
        self._drawn.clear()
        workspace.s = guide.s
        workspace.ui = guide.ui
        workspace.ui.numeric_edits.clear()
        app._popup_owned_frame = guide.window.popup_owned_frame
        workspace.menus.s = guide.s
        app.localizer = guide.localizer
        workspace.ui.translate = guide.tr
        app.theme = guide.theme
        app._dt = 1 / 60
        self.apply_queued_edits()
        app.session.tick(
            FrameNeeds(qpos=True, qvel=True, actuator=True, sensors=True, joint_frames=True),
            1 / 60 if guide.animate else 0,
        )
        app._sync_structure()
        app.projection.refresh_document()
        workspace.menus.apply_actions()
        workspace.menus.begin_frame()
        self.context = app._panel_context()
        for callback, label in (
            ("request_texture_import", "Import texture"),
            ("request_geometry_resource_import", "Import geometry resource"),
            ("request_model_asset_import", "Import model asset"),
            ("request_model_asset_replace", "Replace model asset"),
        ):
            setattr(self.context, callback, partial(self.preview_action, label))
        workspace._panel_context = self.context
        workspace.ui.value_context = self.context
        guide.sample_output = app.panels.get("Output")
        guide.sample_history = app.output
        selected = guide.selected.get("Inspector tabs", 0)
        if selected != self._inspector_selection:
            workspace.inspector_tab = ("Properties", "Physics", "Material", "Joint")[selected]
            self._inspector_selection = selected
        selected = guide.selected.get("Settings navigation", 0)
        if selected != self._settings_selection:
            from mojive.ui.workspace.settings_panel import CATEGORIES

            workspace.settings.show_category(CATEGORIES[selected])
            self._settings_selection = selected

    @contextmanager
    def panel_region(self, name, x, y, width, height):
        guide, workspace = self.guide, self.app.workspace
        rect = guide.rect(x, y, width, height)
        g.set_cursor_screen_pos((rect.x, rect.y))
        with group_scope(name), guide.ui.design.family_scope(PANEL_FAMILIES.get(name)):
            geometry = floating_panel_geometry(guide.ui, name) if name == "Settings" else None
            background = surface_ink(
                guide.ui, "panel", "floating" if name == "Settings" else "background"
            )
        g.push_style_var(
            g.StyleVar_.window_padding,
            (geometry.padding,) * 2
            if geometry is not None
            else (DESIGN.panel_padding_x * guide.s, DESIGN.panel_padding_y * guide.s),
        )
        if geometry is not None:
            g.push_style_var(g.StyleVar_.child_rounding, geometry.radius)
        g.push_style_color(g.Col_.child_bg, tuple(value / 255 for value in background))
        g.begin_child(
            "Native specimen/" + name,
            (rect.width, rect.height),
            g.ChildFlags_.always_use_window_padding,
        )
        g.pop_style_color()
        g.pop_style_var(2 if geometry is not None else 1)
        corners = (
            panel_content_corners(guide.ui, name, g.internal.get_current_window())
            if geometry is not None
            else nullcontext()
        )
        with group_scope(name), guide.ui.design.family_scope(PANEL_FAMILIES.get(name)), corners:
            position, available = g.get_cursor_screen_pos(), g.get_content_region_avail()
            box = guide.ui.component_box(
                "panel", position.x, position.y, available.x, available.y, key="panel-" + name
            )
            workspace.open[name] = True
            try:
                yield box.content
            finally:
                try:
                    self.app._draw_rename_popup()
                    if self.app.model_edits.active:
                        self.app._viewport_rect = rect.tuple()
                        self.app._draw_pending_model_edits()
                finally:
                    g.end_child()

    def panel(self, name, x, y, width, height):
        app, workspace = self.app, self.app.workspace
        self._drawn.add(name)
        with self.panel_region(name, x, y, width, height) as content:
            draw_content(workspace, name, self.context, content)
            panel = app.panels.get(name)
            if name == "Inspector":
                panel.finish_frame(
                    self.context,
                    keep_edit_transaction=bool(
                        getattr(
                            getattr(workspace, "inspector_presentation", None),
                            "color_edit_open",
                            False,
                        )
                    ),
                )
                self.guide.selected["Inspector tabs"] = (
                    "Properties",
                    "Physics",
                    "Material",
                    "Joint",
                ).index(workspace.inspector_tab)
                self._inspector_selection = self.guide.selected["Inspector tabs"]
            else:
                panel.finish_frame(self.context)
            if name == "Settings":
                from mojive.ui.workspace.settings_panel import CATEGORIES

                self.guide.selected["Settings navigation"] = CATEGORIES.index(
                    workspace.settings._category
                )
                self._settings_selection = self.guide.selected["Settings navigation"]
        return y + height

    def viewport(self, x, y, width, height):
        with self.guide.ui.design.family_scope("viewport"):
            return self._viewport(x, y, width, height)

    def _viewport(self, x, y, width, height):
        guide, workspace = self.guide, self.app.workspace
        rect = guide.rect(x, y, width, height)
        guide.ui.rect(*rect.tuple(), guide.ink("canvas"), 8)
        overlays = []
        camera_controls(workspace, rect.tuple(), overlays)

        def shell(px, py, w, h, *, radius=8):
            guide.ui.rect(px, py, w * guide.s, h * guide.s, BACKGROUND, radius)
            overlays.append((px, py, w * guide.s, h * guide.s))

        transport(workspace, rect.x, rect.y, rect.width, shell)
        horizontal = height < 440
        toolbox(
            workspace,
            rect.tuple(),
            rect.y + (height - 64 if horizontal else 148) * guide.s,
            horizontal,
            False,
            overlays,
        )
        guide.values["playback-running"] = workspace.playing
        guide.values["playback-time"] = workspace.time
        for original, alias in (
            ("play", "playback-toggle"),
            ("next", "playback-next"),
            ("previous", "playback-previous"),
            ("go-start", "playback-reset"),
            ("record", "playback-record"),
        ):
            if original in guide.ui.hits:
                guide.hits[alias] = guide.ui.hits[original]
        workspace.menus.finish_color_edit()
        return y + height

    def finish_frame(self):
        workspace = self.app.workspace
        for panel in self.app.panels.panels:
            if panel.name in self._drawn:
                continue
            if panel.name == "Inspector":
                presentation = getattr(workspace, "inspector_presentation", None)
                if presentation is not None and presentation.color_edit_open:
                    if panel._edit_transaction:
                        self.context.submit(cmd.CancelEditTransaction())
                        panel._edit_transaction = False
                    presentation.color_edit_open = False
                    presentation.color_edit_owner = None
            panel.finish_frame(self.context)
        workspace.menus.finish_color_edit()
        workspace.finish_numeric_frame(self.context)

    def release(self):
        self.app.release()
