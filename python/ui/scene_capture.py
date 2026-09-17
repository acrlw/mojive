"""Scene-only capture using a reusable peer without editor overlay state."""

from ..render.backend import RenderRequest


class SceneCapture:
    def __init__(self):
        self._backend = None
        self._generation = -1

    def render(self, main_backend, session, camera, *, size=None):
        if size is None:
            size = (main_backend.target.width, main_backend.target.height)
        if self._backend is None:
            self._backend = main_backend.create_peer(*size)
        backend = self._backend
        backend.resize(*size)
        if self._generation != session.structure_generation:
            backend.set_scene(session.source)
            self._generation = session.structure_generation
        for flag in main_backend.render_options():
            backend.set_flag(flag, main_backend.get_flag(flag))
        backend.set_background(main_backend.get_background())
        backend.set_geometry_style(main_backend.get_geometry_style())
        backend.set_contact_style(main_backend.get_contact_style())
        backend.set_shadow_quality(main_backend.get_shadow_quality())
        backend.set_debug_view(main_backend.get_debug_view())
        backend.set_label_mode(main_backend.get_label_mode())
        backend.set_frame_mode(main_backend.get_frame_mode())
        backend.set_bvh_depth(main_backend.get_bvh_depth())
        backend.set_camera(camera)
        backend.update(session.frame)
        backend.render(request=RenderRequest.color())
        return backend.target

    def read(self, main_backend, session, camera, *, size=None, out=None):
        return self.render(main_backend, session, camera, size=size).read_rgb(flip=True, out=out)

    def release(self):
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        self._generation = -1
