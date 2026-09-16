"""Capture and click production geometry controls and menus without a desktop window."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def capture(output: Path, language: str, scale: float, panel_width: int) -> dict:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import moderngl
    from imgui_bundle import imgui
    from imgui_bundle.python_backends.opengl_backend_programmable import (
        ProgrammablePipelineRenderer,
    )
    from PIL import Image

    from mojive import commands as cmd
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.render.backend import NullBackend, RenderFlag
    from mojive.render.geometry import GeometryView, geometry_view, set_geometry_flags
    from mojive.scene.assets import resolve
    from mojive.session import Session
    from mojive.ui import fonts, theme
    from mojive.ui.app.menus import _Menus
    from mojive.ui.localization import Localizer
    from mojive.ui.panels import PanelContext
    from mojive.ui.panels.hierarchy import HierarchyPanel
    from mojive.ui.panels.inspector import InspectorPanel

    class Backend(NullBackend):
        def set_geometry_view(self, view):
            set_geometry_flags(self._flags, view)
            return True

    output.mkdir(parents=True, exist_ok=True)
    context = imgui.create_context()
    gl = moderngl.create_standalone_context(backend="egl")
    width, height = round(panel_width * 2 * scale), round(700 * scale)
    target = gl.simple_framebuffer((width, height))
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (width, height)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    fonts.load(imgui, io, size_pt=15 * scale, allow_download=False)
    theme.apply(imgui, ui_scale=scale)
    renderer = ProgrammablePipelineRenderer()
    session = Session(MuJoCoAdapter(resolve("geometry_views")))
    link = next(node for node in session.nodes if node.name == "separate")
    session.submit(cmd.SelectNode(link.node_id))
    backend = Backend()
    backend.caps = replace(backend.caps, render_flags=frozenset(RenderFlag))
    localizer = Localizer(path=output / "preferences.json")
    localizer.set_language(language, persist=False)
    ctx = PanelContext(session, backend, style_scale=scale, translate=localizer.text)
    panels = (HierarchyPanel(), InspectorPanel())
    menu = _Menus()
    menu.session, menu.localizer = session, localizer
    menu.window = SimpleNamespace(style_scale=scale)
    menu.panels = ()
    positions = {}
    current_panel = "menu"
    originals = (imgui.button, imgui.begin_combo, imgui.selectable, imgui.begin_menu)

    def remember(key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        positions[(current_panel, key)] = (lo.x, lo.y, hi.x, hi.y)

    def button(label, *args, **kwargs):
        result = originals[0](label, *args, **kwargs)
        if "##geometry_view-" in label:
            remember(label.rsplit("-", 1)[1])
        return result

    def combo(label, *args, **kwargs):
        result = originals[1](label, *args, **kwargs)
        if label == "##geometry_view" and not result:
            remember("combo")
        return result

    def selectable(label, *args, **kwargs):
        result = originals[2](label, *args, **kwargs)
        remember(label)
        return result

    def begin_menu(label, *args, **kwargs):
        result = originals[3](label, *args, **kwargs)
        if label == localizer.text("Help") and not result:
            remember("Help")
        return result

    imgui.button, imgui.begin_combo, imgui.selectable, imgui.begin_menu = (
        button,
        combo,
        selectable,
        begin_menu,
    )

    def frame():
        nonlocal current_panel
        imgui.new_frame()
        current_panel = "menu"
        menu._draw_main_menu()
        for i, panel in enumerate(panels):
            current_panel = panel.id
            imgui.set_next_window_pos((i * width / 2, 34 * scale))
            imgui.set_next_window_size((width / 2, height - 34 * scale))
            imgui.begin(localizer.text(panel.name))
            panel.draw(ctx)
            imgui.end()
        imgui.render()
        target.use()
        target.clear()
        renderer.render(imgui.get_draw_data())
        gl.finish()

    def click(scope, key):
        x0, y0, x1, y1 = positions[(scope, key)]
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        io.add_mouse_pos_event((x0 + x1) / 2, (y0 + y1) / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        frame()

    def select(scope, index):
        if (scope, "combo") in positions:
            click(scope, "combo")
            first = "Default" if scope == "hierarchy" else "Follow scene"
            click(scope, localizer.text((first, "Visual", "Collision", "Both")[index]))
        else:
            click(scope, str(index))

    try:
        for _ in range(4):
            frame()
        x0, y0, x1, y1 = positions[("menu", "Help")]
        assert 0 <= x0 < x1 <= width and y1 > y0
        assert not any(scope == "menu" and key != "Help" for scope, key in positions)
        for index in (1, 2, 3, 0):
            select("hierarchy", index)
            assert geometry_view(backend) == tuple(GeometryView)[index], (
                language,
                scale,
                panel_width,
                index,
                geometry_view(backend),
                positions,
            )
        for index in (1, 2, 3, 0):
            select("inspector", index)
            assert link.geometry_view == (tuple(GeometryView)[index] if index else None)
            assert geometry_view(backend) == GeometryView.DEFAULT
        select("inspector", 2)
        io.add_mouse_pos_event(width - 2, height - 2)
        for _ in range(3):
            frame()
        path = output / f"panels-{language}-{scale}-{panel_width}.png"
        Image.frombytes("RGB", (width, height), target.read(components=3)).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        ).save(path)
        return {
            "language": language,
            "scale": scale,
            "panel_width": panel_width,
            "image": str(path),
            "help_visible": True,
            "global_and_local_clicks": True,
            "controls": {f"{scope}:{key}": rect for (scope, key), rect in positions.items()},
        }
    except Exception:
        Image.frombytes("RGB", (width, height), target.read(components=3)).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        ).save(output / "failure.png")
        raise
    finally:
        imgui.button, imgui.begin_combo, imgui.selectable, imgui.begin_menu = originals
        renderer.shutdown()
        target.release()
        gl.release()
        session.release()
        imgui.destroy_context(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/geometry-ui"))
    args = parser.parse_args()
    results = [
        capture(args.output, language, scale, width)
        for language, scale, width in (
            ("en", 1, 420),
            ("zh_CN", 1, 420),
            ("en", 2.5, 420),
            ("zh_CN", 2.5, 420),
            ("en", 1, 280),
            ("zh_CN", 1, 280),
        )
    ]
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
