"""Route native menu actions to the existing application and Session commands."""

from mojive import commands as cmd
from mojive.capture import CaptureSurface


def dispatch(menu, action):
    st = menu.workspace
    app, session = st.app, st.app.session
    prefix, _, value = action.partition(":")
    if prefix == "layout":
        st.pending = value
    elif prefix == "panel":
        st.activate(value)
    elif prefix == "toggle":
        if value == "Timeline":
            st.toggle_timeline()
        elif st.open[value]:
            st.open[value] = False
        else:
            st.activate(value)
    elif prefix == "extra-panel":
        if app.panels.get(value) is None:
            app.panels.open(value)
        else:
            app.panels.toggle(value)
    elif prefix == "remove-model":
        app.remove_model(int(value))
    elif prefix == "remove-resource":
        session.submit(cmd.RemoveResourceRoot(value))
    elif prefix == "capture":
        app.request_capture(surface=CaptureSurface(value))
    elif prefix == "record-surface":
        app._start_recording_from_menu(CaptureSurface(value))
    elif action == "stop-video":
        app._request_recording_stop()
    elif action == "pause-video":
        app.pause_recording()
    elif action == "resume-video":
        app.resume_recording()
    elif action == "recording-settings":
        app._open_recording_settings()
    elif action == "physics-options":
        app.show_physics_options()
    elif prefix == "tool":
        app.tool = value
    elif prefix == "space":
        st.space = value
    elif prefix == "projection":
        app.camera.set_orthographic(value == "orthographic")
    elif prefix == "shading":
        from mojive.render.backend import DebugView

        menu.shading = value
        app.backend.set_debug_view(
            DebugView.WIREFRAME if value == "Wireframe" else DebugView.SHADED
        )
    elif prefix == "speed":
        st.speed = float(value)
    elif prefix == "step":
        if not st.playing:
            session.submit(cmd.Step() if int(value) > 0 else cmd.StepBack())
    elif prefix == "create":
        from mojive.types import MeshShape

        if value == "Capsule":
            if session.adapter.caps.topology_editing:
                app._add_model_primitive("capsule", "capsule")
        elif value == "Site":
            if session.adapter.caps.topology_editing:
                app._add_model_site()
        elif value == "Point Light":
            if session.adapter.caps.scene_authoring:
                app._add_scene_light()
        elif value == "Camera":
            if session.adapter.caps.scene_authoring:
                app._add_scene_camera()
        elif value == "Ellipsoid":
            app._add_scene_object(MeshShape.SPHERE, "ellipsoid", size=(0.65, 0.45, 0.35))
        else:
            app._add_scene_object(MeshShape(value.lower()), value.lower())
    elif prefix == "dialog":
        menu.dialog_pending = value
    elif action == "documentation":
        import webbrowser

        webbrowser.open("https://github.com/acrlw/mojive#readme")
    elif action == "play":
        app._toggle_playback()
    elif action == "record":
        app._toggle_state_take_recording()
    elif action == "start":
        app._reset_playback()
    elif action == "loop":
        st.loop = not st.loop
    elif action == "frame":
        app._frame_scene(animate=True)
    elif action == "gizmos":
        app.gizmos = not app.gizmos
    elif action == "shadows":
        app.shadows = not app.shadows
    elif action == "viewport-tab":
        st.show_viewport_tab = not st.show_viewport_tab
    elif action == "timeline":
        st.toggle_timeline()
    elif action == "undo":
        st.undo()
    elif action == "redo":
        session.submit(cmd.Redo())
    elif action == "visibility" and session.selected_node is not None:
        node = session.selected_node
        session.submit(cmd.SetVisible(node.node_id, not node.visible))
    elif prefix == "file":
        actions = {
            "new": lambda: app._request_document_action("new_scene"),
            "load": lambda: app._open_scene_dialog("open"),
            "model": app._open_model_dialog,
            "add": lambda: app._open_model_dialog("add"),
            "reload": lambda: app._queue_model_load("reload", session.asset_path),
            "resource": app._open_resource_dialog,
            "save": app._save_current_scene,
            "save_as": lambda: app._open_scene_dialog("save"),
            "quit": lambda: app._request_document_action("quit"),
        }
        actions[value]()
    else:
        raise ValueError(action)
