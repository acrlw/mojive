"""Scene-only capture using a reusable peer without editor overlay state."""

from ..render.backend import DebugView, FrameMode, LabelMode, RenderFlag, RenderRequest

_HELPER_FLAGS = frozenset(
    RenderFlag(name)
    for name in (
        "joint",
        "actuator",
        "activation",
        "camera",
        "light",
        "rangefinder",
        "constraint",
        "flex_vertex",
        "flex_edge",
        "contactpoint",
        "contactforce",
        "contactsplit",
        "island",
        "autoconnect",
        "com",
        "inertia",
        "scaled_inertia",
        "body_bvh",
        "mesh_bvh",
        "outline",
    )
)


class SceneCapture:
    def __init__(self):
        self._backend = None
        self._generation = -1

    def render(self, main_backend, session, camera):
        size = (main_backend.target.width, main_backend.target.height)
        if self._backend is None:
            self._backend = main_backend.create_peer(*size)
        backend = self._backend
        backend.resize(*size)
        if self._generation != session.structure_generation:
            backend.set_scene(session.source)
            self._generation = session.structure_generation
        for flag in main_backend.render_options():
            backend.set_flag(flag, flag not in _HELPER_FLAGS and main_backend.get_flag(flag))
        backend.set_shadow_quality(main_backend.get_shadow_quality())
        backend.set_debug_view(DebugView.SHADED)
        backend.set_label_mode(LabelMode.NONE)
        backend.set_frame_mode(FrameMode.NONE)
        backend.set_camera(camera)
        backend.update(session.frame)
        backend.render(request=RenderRequest.color())
        return backend.target

    def read(self, main_backend, session, camera, *, out=None):
        return self.render(main_backend, session, camera).read_rgb(flip=True, out=out)

    def release(self):
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        self._generation = -1
