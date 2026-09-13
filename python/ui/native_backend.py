"""ImGui texture ownership and bounded packet staging for an injected native device."""

from __future__ import annotations

import ctypes

import numpy as np
from imgui_bundle import imgui


class NativeImguiBackend:
    """Keep UI memory on its context thread; submit copied packets through the device."""

    def __init__(self, device, *, max_bytes=64 * 1024 * 1024):
        self.device, self.api, self.runtime = device, device.api, device.runtime
        self.max_bytes = max_bytes
        self._context = imgui.get_current_context()
        self._closed = False
        self._font_textures = {}
        self._viewport_textures = {}
        self._vertices = np.empty(0, np.uint8)
        self._indices = np.empty(0, np.uint32)
        io = imgui.get_io()
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        io.backend_flags |= imgui.BackendFlags_.renderer_has_vtx_offset

    def register_texture(self, texture):
        key = texture.id
        if key not in self._viewport_textures:
            self._viewport_textures[key] = self.device.register_texture(texture)
        return imgui.ImTextureRef(self._viewport_textures[key])

    def _update_textures(self, draw_data):
        for tex in draw_data.textures or ():
            if tex.status in (
                imgui.ImTextureStatus.want_create,
                imgui.ImTextureStatus.want_updates,
            ):
                pixels = np.asarray(tex.get_pixels_array(), np.uint8).reshape(
                    tex.height, tex.width, tex.bytes_per_pixel
                )
                if tex.bytes_per_pixel == 1:
                    rgba = np.full((tex.height, tex.width, 4), 255, np.uint8)
                    rgba[..., 3] = pixels[..., 0]
                    pixels = rgba
                texture = self.runtime.upload_texture(pixels)
                key = tex.tex_id
                if key in self._font_textures:
                    self.runtime.destroy_texture(self._font_textures[key])
                    self.device.textures[key] = texture
                else:
                    key = self.device.register_texture(texture)
                    tex.set_tex_id(key)
                self._font_textures[key] = texture
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_destroy:
                key = tex.tex_id
                if key in self._font_textures:
                    self.runtime.destroy_texture(self._font_textures.pop(key))
                    self.device.unregister_texture(key)
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)

    def prepare(self, draw, size):
        if self._closed or imgui.get_current_context() != self._context:
            raise RuntimeError("Native UI backend is closed or belongs to another ImGui context")
        if draw is None or min(*draw.display_size, *draw.framebuffer_scale) <= 0:
            return np.empty(0, np.uint8), np.empty(0, np.uint32), []
        batches = [
            (batch, len(batch.vtx_buffer), len(batch.idx_buffer)) for batch in draw.cmd_lists
        ]
        if (sum(nv for _, nv, _ in batches), sum(ni for _, _, ni in batches)) != (
            draw.total_vtx_count,
            draw.total_idx_count,
        ):
            raise ValueError("ImGui draw counts do not match its buffers")
        if sum(len(batch.cmd_buffer) for batch, _, _ in batches) > 65536:
            raise ValueError("UI command budget exceeded")
        for batch, _, _ in batches:
            for command in batch.cmd_buffer:
                kind = command.callback_kind
                if kind not in (
                    imgui.DrawCallbackKind.none,
                    imgui.DrawCallbackKind.reset_render_state,
                ):
                    raise ValueError("Unsupported ImGui draw callback")
        self._update_textures(draw)
        vertex_bytes = draw.total_vtx_count * imgui.VERTEX_SIZE
        index_count = draw.total_idx_count
        required = vertex_bytes, index_count * 4
        current = self._vertices.nbytes, self._indices.nbytes
        capacity = tuple(max(old, count) for old, count in zip(current, required, strict=True))
        if sum(capacity) > self.max_bytes:
            raise ValueError("UI vertex/index upload budget exceeded")
        grown = tuple(
            old if old >= count else max(4096, old * 2, count)
            for old, count in zip(current, required, strict=True)
        )
        if sum(grown) <= self.max_bytes:
            capacity = grown
        if capacity[0] > current[0]:
            self._vertices = np.empty(capacity[0], np.uint8)
        if capacity[1] > current[1]:
            self._indices = np.empty(capacity[1] // 4, np.uint32)
        vertices = self._vertices[:vertex_bytes]
        indices = self._indices[:index_count]
        sx, sy = size[0] / draw.display_size.x, size[1] / draw.display_size.y
        px, py = draw.display_pos.x, draw.display_pos.y
        vo = io = 0
        commands = []
        used_textures = set()
        for batch, nv, ni in batches:
            ctypes.memmove(
                vertices.ctypes.data + vo * imgui.VERTEX_SIZE,
                batch.vtx_buffer.data_address(),
                nv * imgui.VERTEX_SIZE,
            )
            index_type = ctypes.c_uint16 if imgui.INDEX_SIZE == 2 else ctypes.c_uint32
            src = np.ctypeslib.as_array(
                (index_type * ni).from_address(batch.idx_buffer.data_address())
            )
            indices[io : io + ni] = src
            for command in batch.cmd_buffer:
                if command.callback_kind != imgui.DrawCallbackKind.none or not command.elem_count:
                    continue
                native = self.api.UiCommand()
                native.first_index = io + command.idx_offset
                native.index_count = command.elem_count
                native.vertex_offset = vo + command.vtx_offset
                clip = command.clip_rect
                native.clip = [
                    (clip.x - px) * sx,
                    (clip.y - py) * sy,
                    (clip.z - px) * sx,
                    (clip.w - py) * sy,
                ]
                texture_key = command.tex_ref.get_tex_id()
                native.texture = self.device.textures[texture_key]
                used_textures.add(texture_key)
                commands.append(native)
            vo += nv
            io += ni
        # Closed previews must not accumulate texture references in a long-lived window.
        for texture_id, key in list(self._viewport_textures.items()):
            if key not in used_textures:
                self.device.unregister_texture(key)
                del self._viewport_textures[texture_id]
        xyuv = vertices.view(np.float32).reshape(-1, 5)
        xyuv[:, 0] = (xyuv[:, 0] - px) * sx
        xyuv[:, 1] = (xyuv[:, 1] - py) * sy
        return vertices, indices, commands

    def close(self):
        if self._closed:
            return
        for key, texture in self._font_textures.items():
            self.runtime.destroy_texture(texture)
            self.device.unregister_texture(key)
        for key in self._viewport_textures.values():
            self.device.unregister_texture(key)
        self._font_textures.clear()
        self._viewport_textures.clear()
        self._vertices = np.empty(0, np.uint8)
        self._indices = np.empty(0, np.uint32)
        self._closed = True
