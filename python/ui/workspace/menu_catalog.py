"""Native menu inventory and availability from current product capabilities."""

from imgui_bundle import imgui

from mojive.capture import CaptureSurface, RecordingPhase
from mojive.ui.app.support import _model_filters
from mojive.ui.input_bindings import InputAction

from .constants import LEFT_PANELS, PANEL_ICONS, PANELS, VIEWPORT_TOOLS
from .menu_item import Item
from .panel_manager import PANEL_NAMES

TOP_MENUS = ("File", "Edit", "Add", "View", "Simulate", "Window", "Help")


def _shortcut(app, action):
    binding = app.input_bindings.binding(action)
    return binding.label if binding.key is not None else ""


def items(menu, name):
    st = menu.workspace
    e = st.app.entity
    modifier = "Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl"
    caps = st.app.session.adapter.caps
    if name == "File":
        models = bool(_model_filters(caps))
        return [
            Item("New Scene", "file:new", enabled=caps.supports("scene_new")),
            Item("Open Scene...", "file:load", enabled=caps.supports("scene_open")),
            Item("Save", "file:save", enabled=caps.supports("scene_save")),
            Item("Save As...", "file:save_as", enabled=caps.supports("scene_save")),
            None,
            Item("Open Model...", "file:model", enabled=models),
            Item("Add Models...", "file:add", enabled=models and caps.model_composition),
            Item(
                "Remove Model",
                submenu="Remove Model",
                enabled=any(m.removable for m in st.app.session.scene_models),
            ),
            Item(
                "Reload Model",
                "file:reload",
                enabled=caps.reload and st.app.session.asset_path is not None,
            ),
            Item(
                "Resource Directories",
                submenu="Resource Directories",
                enabled=caps.scene_authoring,
            ),
            None,
            Item("Quit", "file:quit"),
        ]
    if name == "Remove Model":
        return [
            Item(m.name, "remove-model:" + str(m.model_id))
            for m in st.app.session.scene_models
            if m.removable
        ]
    if name == "Resource Directories":
        return [Item("Add Directory...", "file:resource", "plus"), None] + [
            Item(str(root), "remove-resource:" + str(root), "close")
            for root in st.app.session.adapter.resource_roots
        ]
    if name == "Capture":
        return [
            Item(label, "capture:" + surface.value)
            for label, surface in (
                ("Scene Image", CaptureSurface.SCENE),
                ("Viewport with UI", CaptureSurface.VIEWPORT),
                ("Entire Window", CaptureSurface.WINDOW),
            )
        ]
    if name == "Record":
        return [
            Item(label, "record-surface:" + surface.value, "video")
            for label, surface in (
                ("Scene Only", CaptureSurface.SCENE),
                ("Viewport with UI", CaptureSurface.VIEWPORT),
                ("Entire Window", CaptureSurface.WINDOW),
            )
        ]
    if name == "Edit":
        history_allowed = caps.edit_history and not st.app.model_edits.active
        return [
            Item(
                "Undo",
                "undo",
                "undo",
                modifier + "+Z",
                history_allowed and st.app.session.can_undo,
                reason="No changes to undo.",
            ),
            Item(
                "Redo",
                "redo",
                "redo",
                modifier + "+Shift+Z",
                history_allowed and st.app.session.can_redo,
                reason="No changes to redo.",
            ),
            None,
            Item(
                "Visible",
                "visibility",
                "eye",
                enabled=bool(e),
                checkable=True,
                checked=bool(e) and not e["hidden"],
            ),
        ]
    if name == "Add":
        return [
            Item(
                n,
                icon="shape-" + n.lower(),
                action="create:" + n,
                enabled=n != "Capsule" or caps.topology_editing,
            )
            for n in ("Box", "Sphere", "Cylinder", "Capsule", "Cone", "Ellipsoid", "Plane")
        ] + [
            None,
            Item("Point Light", "create:Point Light", "light", enabled=caps.scene_authoring),
            Item("Camera", "create:Camera", "camera", enabled=caps.scene_authoring),
            Item("Site", "create:Site", "target", enabled=caps.topology_editing),
        ]
    if name == "View":
        phase = st.app.recording.phase
        if phase is RecordingPhase.COUNTDOWN:
            recording = [Item("Cancel Recording", "stop-video", "close")]
        elif phase is RecordingPhase.FINALIZING:
            recording = [Item("Finalizing recording", enabled=False)]
        elif st.app.recording.active:
            paused = phase is RecordingPhase.PAUSED
            recording = [
                Item(
                    "Resume Recording" if paused else "Pause Recording",
                    "resume-video" if paused else "pause-video",
                    "play" if paused else "pause",
                ),
                Item("Stop Recording", "stop-video", "video"),
            ]
        else:
            recording = [Item("Record", submenu="Record", icon="video")]
        return [
            Item("Frame all", "frame", "frame", _shortcut(st.app, InputAction.FRAME_SCENE)),
            Item("Capture", submenu="Capture", icon="capture-image"),
            *recording,
            None,
            Item("Recording Settings...", "recording-settings"),
            Item("Projection", submenu="Projection"),
            Item("Transform tool", submenu="Transform tool"),
            Item("Gizmo orientation", submenu="Gizmo orientation"),
            None,
            Item("Transform handles", "gizmos", "move", checkable=True, checked=st.app.gizmos),
        ]
    if name in ("Camera", "Projection"):
        rows = [
            Item(
                "Perspective",
                "projection:perspective",
                "perspective",
                checkable=True,
                checked=not st.app.camera.orthographic,
            ),
            Item(
                "Orthographic",
                "projection:orthographic",
                "orthographic",
                checkable=True,
                checked=st.app.camera.orthographic,
            ),
        ]
        if name == "Camera":
            rows += [
                None,
                Item("Frame all", "frame", "frame", _shortcut(st.app, InputAction.FRAME_SCENE)),
                Item("Cameras and saved views", "panel:Cameras", "camera"),
            ]
        return rows
    if name == "Transform tool":
        return [
            Item(
                n.title(),
                "tool:" + n,
                n,
                "V" if action is None else _shortcut(st.app, action),
                checkable=True,
                checked=st.app.tool == n,
            )
            for n, action in VIEWPORT_TOOLS
        ]
    if name == "Gizmo orientation":
        return [
            Item(
                n + " frame",
                "space:" + n,
                "world" if n == "World" else "body-frame",
                checkable=True,
                checked=st.space == n,
            )
            for n in ("World", "Body")
        ]
    if name == "Simulate":
        return [
            Item(
                "Pause simulation" if st.playing else "Play simulation",
                "play",
                "pause" if st.playing else "play",
                _shortcut(st.app, InputAction.TOGGLE_PAUSE),
            ),
            Item("Go to start", "start", "reset"),
            Item(
                "Stop recording" if st.app.session.state_take_recording else "Record simulation",
                "record",
                icon="record",
                enabled=st.app.session.adapter.caps.simulation,
                checkable=True,
                checked=st.app.session.state_take_recording,
            ),
            None,
            Item("Playback speed", submenu="Playback speed"),
            Item("Physics adapter", "panel:Control"),
            Item(
                "Physics Options...",
                "physics-options",
                enabled=caps.simulation or caps.supports("physics.options"),
            ),
        ]
    if name == "Playback speed":
        return [
            Item(f"{v:g}× speed", "speed:" + str(v), checkable=True, checked=st.speed == v)
            for v in (0.25, 0.5, 1, 2)
        ]
    if name == "Playback":
        return [
            Item("Previous frame", "step:-1", "previous", enabled=not st.playing),
            Item("Next frame", "step:1", "next", enabled=not st.playing),
            None,
            Item("Playback speed", submenu="Playback speed"),
            Item(
                "Loop playback",
                "loop",
                checkable=True,
                checked=st.loop,
                enabled=bool(st.app.session.state_take_times),
                reason="Record a take first.",
            ),
        ]
    if name == "Window":
        return [
            Item(
                "Hide left panel"
                if any(st.open.get(n, False) for n in LEFT_PANELS)
                else "Show left panel",
                "toggle-side:left",
                "sidebar-left",
            ),
            Item(
                "Hide right panel" if st.open["Inspector"] else "Show right panel",
                "toggle:Inspector",
                "inspect",
            ),
            None,
            Item("Panels", submenu="Panels"),
            Item("Inspector position", submenu="Inspector position"),
            None,
            Item("Save layout", "layout:save"),
            Item(
                "Restore saved layout",
                "layout:restore",
                enabled=bool(st.saved),
                reason="Save a layout first.",
            ),
            Item("Reset Layout", "layout:reset"),
            None,
            Item(
                "Show Viewport tab",
                "viewport-tab",
                checkable=True,
                checked=st.show_viewport_tab,
            ),
            Item("Settings", "panel:Settings", shortcut="F9"),
            Item(
                "Physics Options...",
                "physics-options",
                enabled=caps.simulation or caps.supports("physics.options"),
            ),
        ]
    if name == "Panels":
        unloaded = {
            PANEL_NAMES.get(panel_id, title): panel_id
            for panel_id, title in st.app.panels.unloaded_builtins()
        }
        available = {panel.name for panel in st.app.panels if panel.enabled} | unloaded.keys()
        return [
            Item(
                n,
                "toggle:" + n,
                icon,
                checkable=True,
                checked=st.open[n],
                enabled=n in available,
            )
            for n, icon in menu.panels
            if n != "Settings"
        ] + [
            *(
                Item(
                    p.name,
                    "extra-panel:" + p.id,
                    PANEL_ICONS.get(p.id, "inspect"),
                    checkable=True,
                    checked=p.open,
                )
                for p in st.app.panels
                if p.id
                not in (
                    "hierarchy",
                    "joints",
                    "control",
                    "assets",
                    "camera",
                    "sensors",
                    "layers",
                    "output",
                    "stats",
                    "inspector",
                    "keyframes",
                    "settings",
                    "info",
                )
                and p.enabled
            ),
            *(
                Item(title, "extra-panel:" + panel_id, PANEL_ICONS.get(panel_id, "inspect"))
                for title, panel_id in unloaded.items()
                if title not in (*PANELS, "Inspector", "Timeline") and panel_id != "info"
            ),
            None,
            Item(
                "Inspector",
                "toggle:Inspector",
                "inspect",
                checkable=True,
                checked=st.open["Inspector"],
                enabled="Inspector" in available,
            ),
            Item(
                "Timeline",
                "timeline",
                "key",
                "T",
                checkable=True,
                checked=st.open["Timeline"],
                enabled="Timeline" in available,
            ),
        ]
    if name == "Inspector position":
        return [
            Item("Float", "layout:float"),
            Item("Dock on the right", "layout:dock"),
            Item("Group with Scene", "layout:group"),
        ]
    if name == "Shading":
        return [
            Item(n, "shading:" + n, "shading", checkable=True, checked=menu.shading == n)
            for n in ("Solid", "Wireframe")
        ]
    if name == "Help":
        return [
            Item("Documentation", "documentation"),
            Item("Interaction guide", "dialog:Interaction guide"),
            Item("About Mojive", "dialog:About Mojive"),
        ]
    raise ValueError(name)
