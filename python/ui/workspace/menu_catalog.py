"""Native menu inventory and availability from current product capabilities."""

from imgui_bundle import imgui

from mojive.capture import CaptureSurface, RecordingPhase
from mojive.ui.app.support import _model_filters

from .menu_item import Item

TOP_MENUS = ("File", "Edit", "Add", "View", "Simulate", "Window", "Help")


def items(menu, name):
    st = menu.workspace
    e = st.app.entity
    modifier = "Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl"
    caps = st.app.session.adapter.caps
    if name == "File":
        models = bool(_model_filters(caps))
        return [
            Item("New Scene", "file:new", "file", enabled=caps.supports("scene_new")),
            Item("Open Scene...", "file:load", "assets", enabled=caps.supports("scene_open")),
            Item("Save", "file:save", "save", enabled=caps.supports("scene_save")),
            Item("Save As...", "file:save_as", "save", enabled=caps.supports("scene_save")),
            None,
            Item("Open Model...", "file:model", "body", enabled=models),
            Item("Add Models...", "file:add", "plus", enabled=models and caps.model_composition),
            Item(
                "Remove Model",
                submenu="Remove Model",
                enabled=any(m.removable for m in st.app.session.scene_models),
            ),
            Item(
                "Reload Model",
                "file:reload",
                "reset",
                enabled=caps.reload and st.app.session.asset_path is not None,
            ),
            Item(
                "Resource Directories",
                submenu="Resource Directories",
                enabled=caps.scene_authoring,
            ),
            None,
            Item("Quit", "file:quit", "close"),
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
            Item(label, "capture:" + surface.value, "capture-image")
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
            Item("Frame all", "frame", "frame", "F"),
            Item("Capture", submenu="Capture", icon="capture-image"),
            *recording,
            Item("Recording Settings...", "recording-settings", "video"),
            None,
            Item(
                "Projection",
                icon="orthographic" if st.app.camera.orthographic else "perspective",
                submenu="Projection",
            ),
            Item("Transform tool", icon=st.app.tool, submenu="Transform tool"),
            Item("Gizmo orientation", icon="world", submenu="Gizmo orientation"),
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
                Item("Frame all", "frame", "frame", "F"),
                Item("Cameras and saved views", "panel:Cameras", "camera"),
            ]
        return rows
    if name == "Transform tool":
        return [
            Item(n.title(), "tool:" + n, n, key, checkable=True, checked=st.app.tool == n)
            for n, key in [("select", "V"), ("move", "W"), ("rotate", "E"), ("scale", "R")]
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
                "Space",
            ),
            Item("Go to start", "start", "reset"),
            Item("Playback speed", icon="play", submenu="Playback speed"),
            None,
            Item(
                "Stop recording" if st.app.session.state_take_recording else "Record simulation",
                "record",
                icon="record",
                enabled=st.app.session.adapter.caps.simulation,
                checkable=True,
                checked=st.app.session.state_take_recording,
            ),
            Item("Physics adapter", "panel:Control", "control"),
            Item(
                "Physics Options...",
                "physics-options",
                "control",
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
            Item("Playback speed", icon="play", submenu="Playback speed"),
            Item(
                "Loop playback",
                "loop",
                "loop",
                checkable=True,
                checked=st.loop,
                enabled=bool(st.app.session.state_take_times),
                reason="Record a take first.",
            ),
        ]
    if name == "Window":
        return [
            Item("Panels", icon="scene", submenu="Panels"),
            Item("Inspector position", icon="inspect", submenu="Inspector position"),
            None,
            Item("Save layout", "layout:save", "save"),
            Item(
                "Restore saved layout",
                "layout:restore",
                "undo",
                enabled=bool(st.saved),
                reason="Save a layout first.",
            ),
            Item("Reset Layout", "layout:reset", "reset"),
            None,
            Item(
                "Show Viewport tab",
                "viewport-tab",
                "inspect",
                checkable=True,
                checked=st.show_viewport_tab,
            ),
            Item("Settings", "panel:Settings", "settings", "F9"),
            Item(
                "Physics Options...",
                "physics-options",
                "control",
                enabled=caps.simulation or caps.supports("physics.options"),
            ),
        ]
    if name == "Panels":
        return [
            Item(
                n,
                "toggle:" + n,
                icon,
                checkable=True,
                checked=st.open[n],
                enabled=st.app.panels.get(n) is not None and st.app.panels.get(n).enabled,
            )
            for n, icon in menu.panels
            if n != "Settings"
        ] + [
            *(
                Item(p.name, "extra-panel:" + p.id, checkable=True, checked=p.open)
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
                )
                and p.enabled
            ),
            None,
            Item(
                "Inspector",
                "toggle:Inspector",
                "inspect",
                checkable=True,
                checked=st.open["Inspector"],
            ),
            Item("Timeline", "timeline", "key", "T", checkable=True, checked=st.open["Timeline"]),
        ]
    if name == "Inspector position":
        return [
            Item("Float", "layout:float", "inspect"),
            Item("Dock on the right", "layout:dock", "inspect"),
            Item("Group with Scene", "layout:group", "scene"),
        ]
    if name == "Shading":
        return [
            Item(n, "shading:" + n, "shading", checkable=True, checked=menu.shading == n)
            for n in ("Solid", "Wireframe")
        ]
    if name == "Help":
        return [
            Item("Interaction guide", "dialog:Interaction guide", "help"),
            Item("About Mojive", "extra-panel:info", "info"),
        ]
    raise ValueError(name)
