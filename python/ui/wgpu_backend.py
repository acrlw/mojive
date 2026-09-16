"""WebGPU ImGui compatibility, texture updates and ordered command-span execution."""

from __future__ import annotations

import itertools
import weakref
from typing import ClassVar

import wgpu
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiWgpuBackend

from .clipping import scissor_rect_for_target
from .wgpu_buffers import ImguiBuffers

_texture_ids = itertools.count(1)


def _texture_id():
    value = next(_texture_ids)
    if value > 0x7FFFFFFF:
        raise OverflowError("ImGui texture identifier space exhausted")
    return value


class WgpuImguiBackend(ImguiWgpuBackend):
    """ImGui texture ownership and reusable GPU submission for plain and mixed UI."""

    _registries: ClassVar[weakref.WeakKeyDictionary] = weakref.WeakKeyDictionary()

    def __init__(self, device, target_format) -> None:
        super().__init__(device, target_format)
        # ImGui contexts may share an atlas, but GPU handles never cross devices.
        self._textures, self._texture_views = self._registries.setdefault(device, ({}, {}))
        self._registered_textures: set[int] = set()
        self._texture_bind_groups: dict[int, tuple[object, object]] = {}
        self._context = imgui.get_current_context()
        self.buffers = ImguiBuffers(device)
        self._closed = False

    def register_texture(self, texture_view):
        if self._closed:
            raise RuntimeError("WebGPU UI backend is closed")
        if texture_view._device is not self._device:
            raise ValueError("ImGui texture belongs to another WebGPU device")
        # Each registration owns an independent reference, even to a shared view.
        key = _texture_id()
        self._texture_views[key] = texture_view
        self._registered_textures.add(key)
        return imgui.ImTextureRef(key)

    def unregister_texture(self, texture_ref):
        key = texture_ref.get_tex_id()
        if key not in self._registered_textures:
            raise ValueError("ImGui texture registration does not belong to this backend")
        self._registered_textures.remove(key)
        self._texture_bind_groups.pop(key, None)
        super().unregister_texture(texture_ref)

    def _destroy_texture(self, tex):
        key = tex.tex_id
        texture = self._textures.pop(key, None)
        self._texture_views.pop(key, None)
        self._texture_bind_groups.pop(key, None)
        if texture is not None:
            texture.destroy()
        tex.set_tex_id(0)
        tex.set_status(imgui.ImTextureStatus.destroyed)

    def _update_texture(self, tex: imgui.ImTextureData) -> None:
        if tex.status == imgui.ImTextureStatus.want_updates and tex.tex_id not in self._textures:
            raise ValueError("ImGui atlas update has no texture on this WebGPU device")
        if tex.status == imgui.ImTextureStatus.want_create:
            assert tex.tex_id == 0
            assert tex.format == imgui.ImTextureFormat.rgba32

            wgpu_tex = self._device.create_texture(
                label="Dear ImGui Texture",
                size=(tex.width, tex.height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            wgpu_tex_view = wgpu_tex.create_view()
            tex_id = _texture_id()
            tex.set_tex_id(tex_id)
            self._textures[tex_id] = wgpu_tex
            self._texture_views[tex_id] = wgpu_tex_view
            full_upload = True
        else:
            full_upload = False

        if tex.status in (
            imgui.ImTextureStatus.want_create,
            imgui.ImTextureStatus.want_updates,
        ):
            wgpu_tex = self._textures[tex.tex_id]
            if full_upload:
                upload_x = upload_y = 0
                upload_w, upload_h = tex.width, tex.height
            else:
                upload_x = tex.update_rect.x
                upload_y = tex.update_rect.y
                upload_w, upload_h = tex.update_rect.w, tex.update_rect.h

            full_data = tex.get_pixels_array()
            offset = (upload_y * tex.width + upload_x) * tex.bytes_per_pixel
            self._device.queue.write_texture(
                {
                    "texture": wgpu_tex,
                    "mip_level": 0,
                    "origin": (upload_x, upload_y, 0),
                },
                full_data[offset:],
                {"offset": 0, "bytes_per_row": tex.width * tex.bytes_per_pixel},
                (upload_w, upload_h, 1),
            )
            tex.set_status(imgui.ImTextureStatus.ok)

        if tex.status == imgui.ImTextureStatus.want_destroy and tex.unused_frames > 0:
            self._destroy_texture(tex)

    def prepare(self, draw_data):
        """Upload vertices, indices, uniforms and texture changes once for this UI frame."""
        if self._closed:
            raise RuntimeError("WebGPU UI backend is closed")
        if imgui.get_current_context() != self._context:
            raise RuntimeError("WebGPU UI backend belongs to another ImGui context")
        self._clear_pending_textures()
        if draw_data is None or min(*draw_data.display_size, *draw_data.framebuffer_scale) <= 0:
            return None
        for tex in draw_data.textures or ():
            if tex.status != imgui.ImTextureStatus.ok:
                self._update_texture(tex)
        # Another context can retire an atlas in the shared device registry.
        for key in tuple(self._texture_bind_groups):
            if key not in self._texture_views:
                del self._texture_bind_groups[key]
        offsets = self.buffers.upload(draw_data)
        self._set_render_state(draw_data)
        return offsets

    def bind_state(self, render_pass, target_size):
        render_pass.set_viewport(0, 0, *target_size, 0, 1)
        render_pass.set_pipeline(self._render_pipeline)
        if self.buffers.vertices is not None:
            render_pass.set_vertex_buffer(0, self.buffers.vertices)
        if self.buffers.indices is not None:
            render_pass.set_index_buffer(
                self.buffers.indices, "uint16" if imgui.INDEX_SIZE == 2 else "uint32"
            )
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.set_blend_constant((0, 0, 0, 0))

    def _render_list(self, draw_data, draw_list, offsets, render_pass, target_size):
        """Emit an already prepared range; preserve global vertex and index offsets."""
        vo, io = offsets[draw_list]
        scale, origin = tuple(draw_data.framebuffer_scale), tuple(draw_data.display_pos)
        draw_size = tuple(round(v * s) for v, s in zip(draw_data.display_size, scale, strict=True))
        for command in draw_list.cmd_buffer:
            kind = command.callback_kind
            if kind == imgui.DrawCallbackKind.reset_render_state:
                self.bind_state(render_pass, target_size)
                continue
            if kind != imgui.DrawCallbackKind.none:
                raise ValueError("Unsupported ImGui draw callback")
            if not command.elem_count:
                continue
            clip = scissor_rect_for_target(
                tuple(command.clip_rect), origin, scale, draw_size, target_size
            )
            if clip is None:
                continue
            render_pass.set_bind_group(1, self._texture_group(command.tex_ref.get_tex_id()))
            render_pass.set_scissor_rect(*clip)
            render_pass.draw_indexed(
                command.elem_count, 1, command.idx_offset + io, command.vtx_offset + vo, 0
            )

    def _texture_group(self, key):
        view = self._texture_views[key]
        if view._device is not self._device:
            raise ValueError("ImGui texture belongs to another WebGPU device")
        cached = self._texture_bind_groups.get(key)
        if cached is None or cached[0] is not view:
            group = self._device.create_bind_group(
                layout=self._texture_bind_group_layout,
                entries=[{"binding": 0, "resource": view}],
            )
            cached = self._texture_bind_groups[key] = (view, group)
        return cached[1]

    def render(self, draw_data, render_pass, target_size=None):
        """Render an ordinary ImGui frame using reusable upload buffers."""
        offsets = self.prepare(draw_data)
        if offsets is None:
            return
        size = target_size or tuple(
            round(v * s)
            for v, s in zip(draw_data.display_size, draw_data.framebuffer_scale, strict=True)
        )
        if min(size) <= 0:
            return
        self.bind_state(render_pass, size)
        for draw_list in draw_data.cmd_lists:
            self._render_list(
                draw_data,
                draw_list,
                offsets,
                render_pass,
                size,
            )

    def close(self):
        if self._closed:
            return
        if imgui.get_current_context() != self._context:
            raise RuntimeError("Close the WebGPU UI backend in its owning ImGui context")
        self._clear_pending_textures()
        for key in self._registered_textures:
            self._texture_views.pop(key, None)
        self._registered_textures.clear()
        self._texture_bind_groups.clear()
        # RefCount is ImGui's number of contexts sharing this atlas texture.
        for texture in imgui.get_platform_io().textures:
            if texture.ref_count == 1:
                self._destroy_texture(texture)
        self.buffers.close()
        self._uniform_buffer.destroy()
        self._context = None
        self._closed = True
