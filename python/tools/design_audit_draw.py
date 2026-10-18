"""Sample submitted production triangles from their owning native draw list."""

import ctypes
import hashlib

import numpy as np
from imgui_bundle import imgui

VERTEX_DTYPE = np.dtype([("position", "f4", 2), ("uv", "f4", 2), ("color", "u4")])


class _FontTexture:
    """Compare rendered glyph pixels independently of atlas growth and packing."""

    def __init__(self, texture):
        self.size = np.array((texture.width, texture.height))
        self.pixels = np.asarray(texture.get_pixels_array(), np.uint8).reshape(
            texture.height, texture.width, texture.bytes_per_pixel
        )
        self.signatures = {}

    def signature(self, uv):
        points = np.rint(uv * self.size).astype(int)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        if left == right or top == bottom:
            return (0, 0)
        key = (left, top, right, bottom)
        if key not in self.signatures:
            tile = self.pixels[top:bottom, left:right]
            digest = hashlib.blake2b(digest_size=8)
            digest.update(np.asarray(tile.shape, np.uint32).tobytes())
            digest.update(tile.tobytes())
            self.signatures[key] = tuple(np.frombuffer(digest.digest(), dtype=np.uint32))
        return self.signatures[key]


def triangle_overlap(points, rect):
    """Use separating axes to exclude triangles outside the actual clip rectangle."""
    left, top, right, bottom = rect
    minimum, maximum = points.min(axis=1), points.max(axis=1)
    inside = (
        (maximum[:, 0] > left)
        & (minimum[:, 0] < right)
        & (maximum[:, 1] > top)
        & (minimum[:, 1] < bottom)
    )
    center = np.array(((left + right) / 2, (top + bottom) / 2))
    half = np.array(((right - left) / 2, (bottom - top) / 2))
    for edge in range(3):
        delta = points[:, (edge + 1) % 3] - points[:, edge]
        normal = np.column_stack((-delta[:, 1], delta[:, 0]))
        projection = np.einsum("nvi,ni->nv", points, normal)
        middle = normal @ center
        radius = np.abs(normal) @ half
        inside &= (projection.max(axis=1) > middle - radius) & (
            projection.min(axis=1) < middle + radius
        )
    return inside


class NativeDrawSamples:
    """Resolve logical owners after Render, when each buffer has its final address.

    The caller supplies any scissor scale applied in place by its renderer.
    Vertices and observation rectangles remain in logical ImGui coordinates.
    """

    def __init__(self, workbench, *, clip_scale=(1.0, 1.0)):
        self.workbench = workbench
        self.windows = {}
        self.descendants = {}
        font_textures = {
            texture.get_tex_id(): _FontTexture(texture) for texture in imgui.get_io().fonts.tex_list
        }
        self.excluded = set()
        context = imgui.get_current_context()
        for window in context.windows:
            address = window.draw_list.vtx_buffer.data_address()
            if workbench._editor_window(window):
                self.excluded.add(address)
            else:
                self.windows[window.id_] = address
                parent = (
                    window.parent_window if window.flags & imgui.WindowFlags_.child_window else None
                )
                while parent is not None:
                    self.descendants.setdefault(parent.id_, set()).add(window.id_)
                    parent = parent.parent_window
        # The context exposes internal ViewportP bindings without the public ID;
        # PlatformIO retains the public viewport objects accepted by draw APIs.
        self.viewports = {viewport.id_: viewport for viewport in imgui.get_platform_io().viewports}
        self.batches = {}
        data = imgui.get_draw_data()
        for batch in data.cmd_lists if data is not None else ():
            address = batch.vtx_buffer.data_address()
            if address in self.excluded or not len(batch.vtx_buffer):
                continue
            raw = (ctypes.c_ubyte * (len(batch.vtx_buffer) * imgui.VERTEX_SIZE)).from_address(
                address
            )
            vertices = np.frombuffer(raw, VERTEX_DTYPE).copy()
            index_dtype = np.dtype("u2" if imgui.INDEX_SIZE == 2 else "u4")
            raw_indices = (
                ctypes.c_ubyte * (len(batch.idx_buffer) * imgui.INDEX_SIZE)
            ).from_address(batch.idx_buffer.data_address())
            indices = np.frombuffer(raw_indices, index_dtype).copy()
            commands = []
            for command in batch.cmd_buffer:
                count = command.elem_count // 3 * 3
                if not count:
                    continue
                triangles = indices[command.idx_offset : command.idx_offset + count].reshape(-1, 3)
                triangles = triangles.astype(np.int64) + command.vtx_offset
                triangles = triangles[np.any(vertices["color"][triangles] >> 24, axis=1)]
                if not len(triangles):
                    continue
                clip = command.clip_rect
                positions = vertices["position"][triangles]
                commands.append(
                    (
                        triangles,
                        positions,
                        positions.min(axis=1),
                        positions.max(axis=1),
                        (
                            clip.x / clip_scale[0],
                            clip.y / clip_scale[1],
                            clip.z / clip_scale[0],
                            clip.w / clip_scale[1],
                        ),
                        font_textures.get(command.get_tex_id()),
                    )
                )
            self.batches[address] = vertices, commands

    def sample(self, record):
        owner = record.get("drawListSource", "window")
        if owner == "foreground":
            viewport = self.viewports.get(record.get("viewportId"))
            if viewport is None:
                return np.empty((0, 6)), {"sourceError": "Foreground record has no viewport"}
            address = imgui.get_foreground_draw_list(viewport).vtx_buffer.data_address()
        elif owner == "window":
            address = self.windows.get(record.get("windowId"))
        else:
            return np.empty((0, 6)), {"sourceError": "Unknown logical draw-list owner"}
        children = set()
        if record.get("kind") not in (
            "button",
            "input_number",
            "axis_input",
            "input_text",
            "search",
            "range",
            "checkbox",
            "switch",
            "separator",
        ):
            children.update(self.descendants.get(record.get("windowId"), ()))
        if record.get("nativeChildWindowId") is not None:
            children.add(record["nativeChildWindowId"])
        addresses = [address] + [
            self.windows[child] for child in sorted(children) if child in self.windows
        ]
        sampled = [
            self.sample_batch(self.batches[source], record)
            for source in addresses
            if source in self.batches
        ]
        if not sampled:
            return np.empty((0, 6)), {
                "sourceError": "Production owner and native children have no submitted draw list"
            }
        vertices = [rows for rows, _ in sampled if len(rows)]
        return (
            np.concatenate(vertices) if vertices else np.empty((0, 6)),
            {
                "source": owner,
                "nativeChildWindowIds": sorted(children),
                "triangleCount": sum(count for _, count in sampled),
            },
        )

    @staticmethod
    def sample_batch(batch, record):
        vertices, commands = batch
        window_frame = (
            "panel" in record.get("parts", ())
            and record.get("docked") is False
            and record.get("windowRect") is not None
            and record.get("nativeBackgroundClip") is not None
        )
        x, y, width, height = record["windowRect"] if window_frame else record["visibleHit"]
        rectangle = x, y, x + width, y + height
        if window_frame:
            clip = record["nativeBackgroundClip"]
            rectangle = (
                max(rectangle[0], clip[0]),
                max(rectangle[1], clip[1]),
                min(rectangle[2], clip[2]),
                min(rectangle[3], clip[3]),
            )
        selected = []
        text_vertices = []
        texture_evidence = vertices["uv"].astype(np.float64)
        triangles_count = 0
        for triangles, positions, minimum, maximum, clip, font_texture in commands:
            clipped = (
                max(rectangle[0], clip[0]),
                max(rectangle[1], clip[1]),
                min(rectangle[2], clip[2]),
                min(rectangle[3], clip[3]),
            )
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                continue
            candidates = np.flatnonzero(
                (maximum[:, 0] > clipped[0])
                & (minimum[:, 0] < clipped[2])
                & (maximum[:, 1] > clipped[1])
                & (minimum[:, 1] < clipped[3])
            )
            if not len(candidates):
                continue
            visible = candidates[triangle_overlap(positions[candidates], clipped)]
            if len(visible):
                selected.append(triangles[visible].reshape(-1))
                if font_texture:
                    text_vertices.append(triangles[visible].reshape(-1))
                    for triangle in triangles[visible]:
                        texture_evidence[triangle] = font_texture.signature(
                            vertices["uv"][triangle]
                        )
                triangles_count += len(visible)
        if not selected:
            return np.empty((0, 6)), 0
        indices = np.unique(np.concatenate(selected))
        white = imgui.get_io().fonts.tex_uv_white_pixel
        font_vertex = np.zeros(len(vertices), dtype=bool)
        if text_vertices:
            font_vertex[np.unique(np.concatenate(text_vertices))] = True
        glyph = (
            font_vertex[indices]
            & (vertices["color"][indices] >> 24 != 0)
            & np.any(np.abs(vertices["uv"][indices] - (white.x, white.y)) > 1e-7, axis=1)
        )
        observed = np.column_stack(
            # Atlas growth can repack glyphs. Their bitmap content, geometry
            # and ink remain invariant; distinct rendered faces still differ.
            (
                np.round(vertices["position"][indices], 3),
                vertices["color"][indices],
                glyph,
                texture_evidence[indices],
            )
        )
        return observed, triangles_count
