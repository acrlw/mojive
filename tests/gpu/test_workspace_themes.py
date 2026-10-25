"""Switch named themes and undo real edits in hidden production windows."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.themes import select_theme, theme_name
from mojive.tools.native_ui_review import _floating_panel, _NativeInput, _records, _reveal
from mojive.tools.native_workspace import capture

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
OUTPUT = Path(__file__).resolve().parents[2] / "output/theme-and-notice-2026-10-25/acceptance"


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_light_button_hierarchy(backend_name, monkeypatch, tmp_path, scale):
    from mojive.session.model_edits import model_edit_scope
    from mojive.ui.interaction_paint import secondary_action_ink
    from mojive.ui.theme import relative_luminance

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    target = OUTPUT.parents[1] / "button-hierarchy-2026-10-25" / f"{backend_name}-{scale:g}"
    target.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session = viewer.app, viewer.session
        app.set_language("zh_CN")
        app.design.observe_components = True
        w, ui = app.workspace, app.workspace.ui
        inputs.sync(10)
        select_theme(app.design, "light", persist=False)
        inputs.sync()
        states = {}

        def record(key):
            return next(row for row in _records(viewer) if row["key"] == key)

        def save(name, key):
            capture(viewer, target / f"{name}.png")
            row = record(key)
            states[name] = row
            pixels = Image.open(target / f"{name}.png")
            density = pixels.width / imgui.get_io().display_size.x
            x, y, width, height = row["hit"]
            pixels.crop(
                tuple(
                    round(value * density)
                    for value in (
                        x - 14 * scale,
                        y - 14 * scale,
                        x + width + 14 * scale,
                        y + height + 14 * scale,
                    )
                )
            ).save(target / f"{name}-detail.png")
            if (
                row["enabled"]
                and row["backgroundColor"] is not None
                and row["backgroundColor"][3] == 255
            ):
                # The dominant interior ink excludes glyphs and fractional AA edges.
                interior = pixels.crop(
                    tuple(
                        round(value * density)
                        for value in (
                            x + 2 * scale,
                            y + 2 * scale,
                            x + width - 2 * scale,
                            y + height - 2 * scale,
                        )
                    )
                )
                colors, counts = np.unique(
                    np.asarray(interior)[..., :3].reshape(-1, 3), axis=0, return_counts=True
                )
                sampled = colors[counts.argmax()]
                assert sampled == pytest.approx(row["backgroundColor"][:3], abs=2)
                low, high = sorted(
                    relative_luminance(tuple(v / 255 for v in row[field][:3]))
                    for field in ("backgroundColor", "textColor")
                )
                assert (high + 0.05) / (low + 0.05) >= 4.5
            return row

        def inspect(name, key, role):
            inputs.move((0, 0))
            idle = save(f"{name}-idle", key)
            assert not idle["hovered"] and not idle["held"]
            assert idle["active"] == (role == "primary")
            assert idle["solid"] == (role == "primary")
            if role == "secondary":
                assert idle["backgroundColor"] == secondary_action_ink(
                    ui.palette("field"), ui.palette("text")
                )
            elif role == "ghost":
                assert idle["backgroundColor"] is None
            x, y, width, height = idle["hit"]
            inputs.move((x + width / 2, y + height / 2))
            hover = save(f"{name}-hover", key)
            assert hover["hovered"] and not hover["held"]
            inputs.event("add_mouse_button_event", 0, True)
            held = save(f"{name}-held", key)
            assert held["held"]
            assert idle["hit"] == hover["hit"] == held["hit"]
            # Release outside the control: inspecting states must not commit it.
            inputs.move((0, 0))
            inputs.event("add_mouse_button_event", 0, False)
            inputs.sync()

        for slot, role in (
            ("button", "field"),
            ("button_hovered", "field_hovered"),
            ("button_active", "field_active"),
        ):
            actual = imgui.get_style().color_(getattr(imgui.Col_, slot))
            assert tuple(round(c * 255) for c in actual) == secondary_action_ink(
                ui.palette(role), ui.palette("text")
            )

        w.activate("Scene")
        inputs.sync()
        inspect("toolbar-search", "header-Scene-search", "ghost")
        node = next(n for n in session.nodes if n.name == "01_revolute_anchor")
        session.submit(cmd.SelectNode(node.node_id))
        w.activate("Assets")
        inputs.sync(5)
        inspect("apply-material", "apply-material", "primary")
        assert session.submit(cmd.Select(0)).ok
        inputs.sync()
        disabled = save("apply-material-disabled", "apply-material")
        assert not disabled["enabled"]
        inputs.click("apply-material")
        assert session.selected_node is None
        session.submit(cmd.SelectNode(node.node_id))
        w.activate("Camera")
        inputs.sync(5)
        inspect("camera-frame", "camera-frame", "secondary")

        w.activate("Scene")
        w.inspector_tab = "Material"
        inputs.sync(5)
        inputs.click("material-color")
        inspect("color-done", "done-color", "primary")
        inputs.click("done-color")
        w.menus.dialog_pending = "About Mojive"
        inputs.sync(5)
        inspect("about-done", "done-guide", "secondary")
        inputs.click("done-guide")

        app.set_live_model_updates(False)
        with model_edit_scope(session, app._intercept_model_edit):
            app._add_model_primitive("capsule", "capsule")
        inputs.sync(3)
        inspect("pending-apply", "apply-model-edits", "primary")
        inspect("pending-discard", "discard-model-edits", "secondary")
        inspect("notice-undo", "notice-undo", "secondary")
        inputs.click("notice-undo")
        assert not app.model_edits.active
        assert "apply-model-edits" not in ui.hits
        app.set_live_model_updates(True)
        assert session.submit(cmd.BeginEditTransaction("Translate transform")).ok
        assert session.submit(cmd.SetPose(node.node_id, np.array([-0.8, 0.0, 0.55]), np.eye(3))).ok
        assert session.submit(cmd.EndEditTransaction()).ok
        app._request_document_action("new_scene")
        inputs.sync(5)
        inspect("document-save", "保存", "primary")
        inspect("document-cancel", "取消", "secondary")
        inspect("document-discard", "放弃", "secondary")
        inputs.click("取消")
        assert app._pending_document_action is None
        assert session.dirty
        (target / "states.json").write_text(json.dumps(states, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_theme_picker_and_operation_undo(backend_name, monkeypatch, tmp_path, scale):
    from mojive.ui.theme import relative_luminance

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=1600,
        height=1100,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    ) as viewer:
        app, session = viewer.app, viewer.session
        app.set_language("zh_CN")
        app.design.observe_components = True
        w = app.workspace
        target = OUTPUT / f"{backend_name}-{scale:g}"
        target.mkdir(parents=True, exist_ok=True)
        original = deepcopy(app.design.spec)
        node = next(n for n in session.nodes if n.name == "01_revolute_anchor")
        session.submit(cmd.SelectNode(node.node_id))
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            w.activate("Settings")
            w.settings.show_category("General")
            inputs.sync()
            with _floating_panel(inputs, "Settings") as width:
                width[0] = 800
                inputs.sync()
                for theme, index in [("light", 1), ("dark", 0)]:
                    inputs.click(f"##ui_theme-{index}")
                    inputs.sync(4)
                    assert theme_name(app.design.spec) == theme
                    assert theme_name(app.design.store.latest()["spec"]) == theme
                    assert app.design.spec["parts"] == original["parts"]
                    assert app.design.spec["layout"] == original["layout"]
                    for panel_width in (640, 800):
                        width[0] = panel_width
                        inputs.sync()
                        rows = {
                            r["key"].rsplit("/", 1)[-1]: r
                            for r in _records(viewer)
                            if r["key"].startswith("settings_general/row/")
                        }
                        fields = [
                            rows[label]["fields"][1]
                            for label in ("主题", "语言", "模型更新", "滚动条隐藏延迟")
                        ]
                        for label in ("主题", "语言", "模型更新", "滚动条隐藏延迟"):
                            assert rows[label]["layout"] == "left"
                        for x, _y, field_width, _height in fields[1:]:
                            assert (x, field_width) == pytest.approx(
                                (fields[0][0], fields[0][2]), abs=scale
                            )
                        label = rows["主题"]["labelRect"]
                        assert label[1] + label[3] / 2 == pytest.approx(
                            fields[0][1] + fields[0][3] / 2, abs=scale
                        )
                        capture(viewer, target / f"settings-{theme}-{panel_width}.png")
                        (target / f"settings-{theme}-{panel_width}.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2)
                        )
                    inputs.hover(f"##ui_theme-{index}")
                    assert not any(r.get("kind") == "tooltip" for r in _records(viewer))
                    capture(viewer, target / f"settings-{theme}.png")
            w.open["Settings"] = False
            inputs.sync()
            for theme in ("light", "dark"):
                inputs.click("toggle-theme")
                assert theme_name(app.design.spec) == theme
                assert theme_name(app.design.store.latest()["spec"]) == theme
                assert app.design.spec["parts"] == original["parts"]
                inputs.hover("toggle-theme")
                assert not any(r.get("kind") == "tooltip" for r in _records(viewer))
                capture(viewer, target / f"titlebar-{theme}.png")
            app.set_live_model_updates(True)
            for theme in ("light", "dark"):
                select_theme(app.design, theme, persist=False)
                before = session.document_revision
                assert session.submit(cmd.BeginEditTransaction("Translate transform")).ok
                assert session.submit(
                    cmd.SetPose(node.node_id, np.array([-0.8, 0.0, 0.55]), np.eye(3))
                ).ok
                assert session.submit(cmd.EndEditTransaction()).ok
                inputs.sync(3)
                assert app.output.active_status().undo == (
                    session.document_id,
                    session.document_revision,
                )
                assert "平移对象" in {item["text"] for item in inputs.emitted}
                notice_target = (
                    OUTPUT.parents[1]
                    / "notice-spacing-2026-10-25"
                    / f"{backend_name}-{scale:g}"
                    / theme
                )
                notice_target.mkdir(parents=True, exist_ok=True)
                states = {}

                def save_notice(name, notice_target=notice_target, states=states):
                    capture(viewer, notice_target / f"{name}.png")
                    record = next(r for r in _records(viewer) if r["key"] == "operation-notice")
                    states[name] = deepcopy(record)
                    image = Image.open(notice_target / f"{name}.png")
                    x, y, width, height = record["hit"]
                    density = image.width / imgui.get_io().display_size.x
                    image.crop(
                        (
                            round((x - 8 * scale) * density),
                            round((y - 8 * scale) * density),
                            round((x + width + 8 * scale) * density),
                            round((y + height + 8 * scale) * density),
                        )
                    ).save(notice_target / f"{name}-detail.png")
                    return record

                active = save_notice("with-undo")
                x, y, width, height = w.ui.hits["notice-undo"]
                nx, ny, nw, nh = active["hit"]
                text_x, _ty, text_width, _th = active["fields"][1]
                assert nh == pytest.approx(36 * scale)
                assert active["radius"] == pytest.approx(8 * scale)
                assert x - text_x - text_width == pytest.approx(16 * scale)
                assert (y - ny, nx + nw - x - width, ny + nh - y - height) == pytest.approx(
                    (4 * scale,) * 3
                )
                pixels = Image.open(notice_target / "with-undo.png")
                density = pixels.width / imgui.get_io().display_size.x

                def luminance_at(px, py, pixels=pixels, density=density):
                    rgb = pixels.getpixel((round(px * density), round(py * density)))[:3]
                    return relative_luminance(tuple(v / 255 for v in rgb))

                if theme == "light":
                    low, high = sorted(
                        (
                            luminance_at(x + 6 * scale, y + 8 * scale),
                            luminance_at(nx + 6 * scale, ny + nh / 2),
                        )
                    )
                    # The caption/icon identifies the secondary action; its fill
                    # remains a subtle separation rather than primary emphasis.
                    assert high > low
                    text_luma = relative_luminance(tuple(v / 255 for v in w.ui.palette("text")[:3]))
                    assert (low + 0.05) / (text_luma + 0.05) >= 4.5
                capture(viewer, target / "notice-undo.png")
                inputs.move((x + width / 2, y + height / 2))
                save_notice("hover")
                inputs.event("add_mouse_button_event", 0, True)
                save_notice("held")
                inputs.event("add_mouse_button_event", 0, False)
                assert session.document_revision == before
                inputs.move((0, 0))
                assert app.output.active_status().undo is None
                assert "撤销：平移对象" in {item["text"] for item in inputs.emitted}
                inactive = save_notice("without-undo")
                assert (
                    *inactive["hit"][:2],
                    inactive["hit"][3],
                    inactive["radius"],
                ) == pytest.approx((nx, ny, nh, active["radius"]))
                assert "notice-undo" not in w.ui.hits
                (notice_target / "report.json").write_text(
                    json.dumps(states, ensure_ascii=False, indent=2)
                )
            inputs.hover("play")
            assert any(
                item.get("kind") == "tooltip" and item.get("text") == "播放／暂停"
                for item in _records(viewer)
            )
            capture(viewer, target / "play-tooltip.png")


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_theme_notice_badges_and_settings_consistency(backend_name, monkeypatch, tmp_path, scale):
    from mojive.session.model_edits import model_edit_scope
    from mojive.types import MeshShape
    from mojive.ui.theme import relative_luminance

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session = viewer.app, viewer.session
        app.set_language("zh_CN")
        app.design.observe_components = True
        w, ui = app.workspace, app.workspace.ui
        inputs.sync(10)
        for theme in ("light", "dark"):
            select_theme(app.design, theme, persist=False)
            target = (
                OUTPUT.parents[1]
                / "ui-consistency-2026-10-25"
                / f"{backend_name}-{scale:g}"
                / theme
            )
            target.mkdir(parents=True, exist_ok=True)
            report = {}

            def save(name, target=target, report=report):
                capture(viewer, target / (name + ".png"))
                report[name] = {"records": _records(viewer), "text": list(inputs.emitted)}

            w.activate("Settings")
            with _floating_panel(inputs, "Settings") as width:
                width[0] = 800
                w.settings.show_category("General")
                inputs.sync(8)
                for category, caption in (
                    ("General", "常规"),
                    ("Camera", "相机"),
                    ("Interaction", "交互"),
                    ("Rendering", "渲染"),
                ):
                    x, y, ww, hh = ui.hits["settings-" + category]
                    ink = next(
                        item["color"]
                        for item in inputs.emitted
                        if item["text"] == caption
                        and x <= item["position"][0] <= x + ww
                        and y <= item["position"][1] <= y + hh
                    )
                    a, b = sorted(
                        (
                            relative_luminance(ink),
                            relative_luminance(tuple(v / 255 for v in ui.palette("background"))),
                        ),
                        reverse=True,
                    )
                    assert (a + 0.05) / (b + 0.05) >= 4.5
                save("settings-general")
                w.settings.show_category("MuJoCo Visuals")
                for panel_width in (640, 800):
                    width[0] = panel_width
                    inputs.sync(8)
                    row = next(
                        r for r in _records(viewer) if r["key"] == "settings_geometry/row/几何显示"
                    )
                    assert row["layout"] == "left"
                    label, field = row["fields"]
                    assert label[1] + label[3] / 2 == pytest.approx(
                        field[1] + field[3] / 2, abs=scale
                    )
                    segment = ui.hits["settings-geometry"]
                    assert segment == pytest.approx(field, abs=scale)
                    aligned_rows = [
                        r
                        for r in _records(viewer)
                        if r["key"].startswith(
                            ("geometry_style_properties/row/", "contact_style_properties/row/")
                        )
                    ]
                    assert len(aligned_rows) >= 8
                    for other in aligned_rows:
                        other_field = other["fields"][1]
                        assert (other_field[0], other_field[2]) == pytest.approx(
                            (field[0], field[2]), abs=scale
                        ), other["key"]
                    heights = {
                        round(ui.hits[key][3], 2)
                        for key in ui.hits
                        if key.startswith("settings-geometry-")
                    }
                    assert len(heights) == 1
                    save(f"settings-visuals-{panel_width}")
            w.open["Settings"] = False
            app.set_live_model_updates(False)
            with model_edit_scope(session, app._intercept_model_edit):
                app._add_scene_object(MeshShape.SPHERE, "sphere")
            inputs.sync(3)
            assert session.last_message_undo is not None
            assert "创建球体" in {item["text"] for item in inputs.emitted}
            assert "notice-undo" in ui.hits
            save("notice-create")
            inputs.click("notice-undo")
            assert not any(n.name == "sphere" for n in session.nodes)
            with model_edit_scope(session, app._intercept_model_edit):
                app._add_model_primitive("capsule", "capsule")
            inputs.sync(3)
            assert app.model_edits.undo_token is not None
            assert "创建胶囊体" in {item["text"] for item in inputs.emitted}
            save("notice-pending")
            inputs.click("notice-undo")
            assert not app.model_edits.active
            assert not any(n.name == "capsule" for n in session.nodes)
            with model_edit_scope(session, app._intercept_model_edit):
                app._add_model_primitive("capsule", "capsule")
            inputs.sync(3)
            inputs.click("apply-model-edits")
            for _ in range(120):
                inputs.sync(2)
                if not app.model_edits.active and app._model_load_future is None:
                    break
            assert not app.model_edits.active
            assert "应用模型编辑" in {item["text"] for item in inputs.emitted}
            save("notice-apply")
            inputs.click("notice-undo")
            assert not any(n.name == "capsule" for n in session.nodes)
            node = next(n for n in session.nodes if n.name == "01_revolute_anchor")
            session.submit(cmd.SelectNode(node.node_id))
            w.inspector_tab = "Material"
            inputs.sync(6)
            channels = [
                r
                for r in _records(viewer)
                if r.get("kind") == "input_number" and r.get("axisWidth", 0)
            ]
            assert channels
            assert all(r["axisWidth"] == pytest.approx(20 * scale) for r in channels)
            channel_text = [
                item for item in inputs.emitted if item["text"] in "RGBA" and len(item["text"]) == 1
            ]
            assert {item["text"] for item in channel_text} == set("RGBA")
            assert all(item["fontPixels"] == pytest.approx(12 * scale) for item in channel_text), (
                channel_text
            )
            save("material-badges")
            w.inspector_tab = "Properties"
            inputs.sync(6)
            save("axis-badges")
            with _floating_panel(inputs, "Inspector") as width:
                width[0] = 640
                w.inspector_tab = "Material"
                inputs.sync(6)
                channel_fields = [
                    r
                    for r in _records(viewer)
                    if r.get("kind") == "input_number" and r.get("axisWidth", 0)
                ]
                assert len(channel_fields) == 4
                assert len({round(r["hit"][1], 2) for r in channel_fields}) == 1
                save("material-badges-wide")
                w.inspector_tab = "Properties"
                inputs.sync(6)
                save("axis-badges-wide")
            (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_theme_overlays_and_projection_transition(backend_name, monkeypatch, tmp_path, scale):
    from mojive.ui.app import status
    from mojive.ui.gizmo import joint_labels, joint_ranges

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    observed = {}
    draw_status, draw_rail = status.draw_status, joint_ranges.draw_value_rail
    draw_readout = joint_labels.draw_gizmo_readout

    def record_status(*args, **kwargs):
        observed["status"] = draw_status(*args, **kwargs)
        return observed["status"]

    def record_rail(*args, **kwargs):
        observed["rail_theme"] = args[5]
        return draw_rail(*args, **kwargs)

    def record_readout(*args, **kwargs):
        observed["readout"] = draw_readout(*args, **kwargs)
        return observed["readout"]

    monkeypatch.setattr(status, "draw_status", record_status)
    monkeypatch.setattr(joint_ranges, "draw_value_rail", record_rail)
    monkeypatch.setattr(joint_labels, "draw_gizmo_readout", record_readout)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session = viewer.app, viewer.session
        ui, workspace = app.workspace.ui, app.workspace
        app.set_language("zh_CN")
        app.design.observe_components = True
        # Readback must not accelerate animations on slower CI renderers.
        advance = app._advance_camera
        monkeypatch.setattr(app, "_advance_camera", lambda dt: advance(1 / 60))
        inputs.sync(10)
        initial_camera = app._camera_view()
        for theme in ("dark", "light"):
            report = {}
            target = (
                OUTPUT.parents[1]
                / "light-theme-audit-2026-10-25"
                / f"{backend_name}-{scale:g}"
                / theme
            )
            target.mkdir(parents=True, exist_ok=True)
            select_theme(app.design, theme, persist=False)
            viewer.set_camera(initial_camera)
            workspace.activate("Scene")
            node = next(n for n in session.nodes if n.name == "01_revolute_anchor")
            assert session.submit(cmd.SelectNode(node.node_id)).ok
            inputs.move((800, 600))
            inputs.sync(6)

            def pixels():
                return Image.fromarray(np.asarray(viewer.window.read_frame())[::-1].copy())

            def save(name, target=target, report=report):
                pixels().save(target / f"{name}.png")
                report[name] = {
                    "records": _records(viewer),
                    "text": list(inputs.emitted),
                    "windows": [
                        {"name": w.name, "rect": [*w.pos, *w.size], "radius": w.window_rounding}
                        for w in imgui.get_current_context().windows
                        if w.active and not w.hidden
                    ],
                }

            save("workspace")
            inputs.click("projection-button")
            save("camera-menu")
            for destination in ("Orthographic", "Perspective"):
                x, y, width, height = ui.hits[f"menu-item-Camera/{destination}"]
                inputs.move((x + width / 2, y + height / 2))
                inputs.event("add_mouse_button_event", 0, True)
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync(2)
                assert app.camera.animating
                blend = app.camera.view().projection_blend()
                assert 0 <= blend <= 1
                frames = [pixels()]
                blends = [blend]
                for index in range(18):
                    inputs.sync(1)
                    blends.append(app.camera.view().projection_blend())
                    frames.append(pixels())
                    if index == 7:
                        assert 0 < blends[-1] < 1
                        save(f"projection-{destination.lower()}-middle")
                assert not app.camera.animating
                assert blends[-1] == (1 if destination == "Orthographic" else 0)
                changes = np.diff(blends)
                assert (
                    np.all(changes >= 0) if destination == "Orthographic" else np.all(changes <= 0)
                )
                if theme == "light" and scale == 1:
                    frames[0].save(
                        target / f"projection-{destination.lower()}.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=[500, *([17] * (len(frames) - 2)), 500],
                        loop=0,
                    )
                save(f"projection-{destination.lower()}-end")
                inputs.click("projection-button")
            inputs.escape()
            inputs.click("menu-Help")
            save("text-menu")
            inputs.escape()
            inputs.click("menu-Window")
            save("window-menu")
            inputs.escape()
            workspace.menus.dispatch("dialog:About Mojive")
            inputs.sync()
            save("about")
            inputs.escape()
            inputs.hover("tool-rotate")
            save("shortcut-tooltip")
            x0, y0, x1, y1 = observed["status"].metric_rect
            inputs.move(((x0 + x1) / 2, (y0 + y1) / 2), frames=45)
            save("status-tooltip")

            workspace.activate("Assets")
            inputs.sync()
            if "##asset-type" not in ui.hits:
                inputs.click("section-" + app.localizer.text("Model assets"))
            assert "##asset-type" in ui.hits, sorted(ui.hits)
            _reveal(inputs, "##asset-type", "Panel content")
            inputs.click("##asset-type")
            save("dropdown")
            inputs.escape()
            node = next(n for n in session.nodes if n.name == "geom1")
            assert session.submit(cmd.SelectNode(node.node_id)).ok
            workspace.inspector_tab = "Material"
            inputs.sync()
            _reveal(inputs, "material-color", "Inspector content")
            inputs.click("material-color")
            inputs.click("color-editor-format-1")
            save("color-popup")
            inputs.escape()
            workspace.inspector_tab = "Properties"

            workspace.activate("Pose")
            for name, tool in (("06_precision_hinge", "rotate"), ("07_precision_slide", "move")):
                node = next(n for n in session.nodes if n.name == name)
                assert session.submit(cmd.SelectNode(node.node_id)).ok
                app.tool = tool
                inputs.sync()
                camera = app._camera_view()
                center = app.gizmo._frame.position.copy()
                viewer.set_camera(
                    replace(camera, eye=camera.eye + center - camera.target, target=center)
                )
                inputs.sync()
                gizmo = app.gizmo
                inputs.move(gizmo.joint_limit_hits[0].tick_end, frames=90)
                assert gizmo.joint_precision_visible
                save(f"{name}-limit")
                # PNG compression can outlast the 300 ms reveal grace. Resume
                # the existing tick hover before crossing onto the rail.
                inputs.sync(2)
                rail = gizmo._joint_precision
                inputs.move(rail.current)
                inputs.event("add_mouse_button_event", 0, True)
                inputs.move(rail.start + (rail.end - rail.start) * 0.75)
                assert gizmo._joint_precision_active and gizmo.value_label
                assert observed["rail_theme"] is ui.control_theme()
                label = next(item for item in inputs.emitted if item["text"] == gizmo.value_label)
                style = ui.component_style("input_number")
                assert label["fontPixels"] == pytest.approx(style.get("fontSize", 16) * scale)
                assert label["color"] == pytest.approx(tuple(v / 255 for v in ui.palette("text")))
                save(f"{name}-drag")
                density = imgui.get_io().display_framebuffer_scale
                expected_thumb = tuple(
                    round(v * 255) for v in ui.control_theme().primary_bright[:3]
                )

                def check_thumb(
                    gizmo=gizmo, density=density, pixels=pixels, expected_thumb=expected_thumb
                ):
                    cx, cy = gizmo._joint_precision.current
                    actual = pixels().getpixel((round(cx * density.x), round(cy * density.y)))[:3]
                    assert actual == pytest.approx(expected_thumb, abs=2)

                check_thumb()
                x0, y0, x1, y1 = observed["readout"]
                rx0, ry0, rx1, ry1 = rail.panel_rect
                density = imgui.get_io().display_framebuffer_scale
                pixels().crop(
                    (
                        (min(x0, rx0) - 8 * scale) * density.x,
                        (min(y0, ry0) - 8 * scale) * density.y,
                        (max(x1, rx1) + 8 * scale) * density.x,
                        (max(y1, ry1) + 8 * scale) * density.y,
                    )
                ).save(target / f"{name}-detail.png")
                # The thumb must keep its own ink above both colored limit ticks.
                for endpoint in (rail.start, rail.end):
                    inputs.move(endpoint)
                    assert gizmo._joint_precision_active
                    check_thumb()
                save(f"{name}-endpoint")
                inputs.escape()
                inputs.event("add_mouse_button_event", 0, False)
                assert not gizmo.using
            (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_theme_icon_family_at_production_sizes(backend_name, tmp_path, scale):
    from mojive.app.ui.window import create_window
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui.theme import THEME
    from mojive.ui.window import WindowConfig
    from mojive.ui.workspace.widgets import UI

    config = WindowConfig(
        width=700,
        height=450,
        ui_scale=scale,
        show_on_start=False,
        vsync=False,
        docking=False,
        ini_path="",
        samples=0,
    )
    with create_window(config, backend_name) as window:
        ui = UI(window, portable_fonts=True)
        runtime = DesignRuntime(SpecificationStore(tmp_path / "icons.json"))
        ui.design = DesignPresentation(ui, runtime)
        for theme in ("dark", "light"):
            select_theme(runtime, theme, persist=False)
            runtime.apply_surface(window, scale, THEME)
            for _ in range(3):
                window.begin_frame()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size(imgui.get_io().display_size)
                imgui.begin("Theme icon family", flags=imgui.WindowFlags_.no_decoration)
                ui.text(20 * scale, 14 * scale, "Theme icons / shared production geometry", size=16)
                for row, name in enumerate(("sun", "moon", "settings", "inspect")):
                    y = (64 + row * 76) * scale
                    ui.text(20 * scale, y, name)
                    for column, size in enumerate((14, 18, 20, 24, 56)):
                        x = (130 + column * 80) * scale
                        ui.icon(
                            name,
                            x,
                            y + (20 - size) * scale / 2,
                            size,
                            ui.palette("text"),
                            color_resolved=True,
                        )
                        if row == 0:
                            ui.text(x, 38 * scale, str(size), size=12, translate=False)
                imgui.end()
                pixels = window.end_frame(readback=True)
            target = (
                OUTPUT.parents[1]
                / "light-theme-audit-2026-10-25"
                / f"{backend_name}-{scale:g}"
                / theme
            )
            target.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(pixels)[::-1].copy()).save(target / "theme-icons.png")


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_theme_pose_slider_states(backend_name, monkeypatch, tmp_path, scale):
    from mojive.ui.panels import value_cards

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    observed = {}
    draw_rail = value_cards.draw_value_rail

    def record_rail(draw, left, right, position, radius, theme, ui_scale, **kwargs):
        # The first scalar joint is the topmost rail in the Pose panel.
        observed[round(left[1], 2)] = (position, left[1], radius, theme, kwargs)
        return draw_rail(draw, left, right, position, radius, theme, ui_scale, **kwargs)

    monkeypatch.setattr(value_cards, "draw_value_rail", record_rail)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, workspace = viewer.app, viewer.app.workspace
        app.set_language("zh_CN")
        workspace.activate("Pose")
        inputs.sync(10)
        for theme in ("dark", "light"):
            select_theme(app.design, theme, persist=False)
            inputs.move((0, 0))
            observed.clear()
            inputs.sync(5)
            hits = {
                key: box
                for key, box in workspace.ui.hits.items()
                if key.startswith("##pose-joint-") and key.endswith("-track")
            }
            assert hits
            key = min(hits, key=lambda name: hits[name][1])
            x, y, width, height = hits[key]
            target = (
                OUTPUT.parents[1]
                / "pose-slider-states-2026-10-25"
                / f"{backend_name}-{scale:g}"
                / theme
            )
            target.mkdir(parents=True, exist_ok=True)
            report = {}
            for state in ("idle", "hover", "held", "released"):
                if state == "hover":
                    inputs.move((x + width / 2, y + height / 2))
                elif state == "held":
                    inputs.event("add_mouse_button_event", 0, True)
                    inputs.move((x + width * 0.65, y + height / 2))
                elif state == "released":
                    inputs.event("add_mouse_button_event", 0, False)
                    inputs.move((0, 0))
                capture(viewer, target / (state + ".png"))
                cx, cy, radius, rail_theme, flags = observed[round(y + height / 2, 2)]
                assert flags["pressed"] == (state == "held")
                assert flags["hovered"] == (state in ("hover", "held"))
                expected = tuple(
                    round(v * 255)
                    for v in (
                        rail_theme.primary_bright
                        if flags["pressed"] or flags["hovered"]
                        else rail_theme.primary
                    )[:3]
                )
                density = imgui.get_io().display_framebuffer_scale
                pixels = Image.open(target / (state + ".png"))
                # Check a disk, not just its center: no white inner fill or ring.
                for dx, dy in ((0, 0), (-0.6, 0), (0.6, 0), (0, -0.6), (0, 0.6)):
                    sample = pixels.getpixel(
                        (
                            round((cx + dx * radius) * density.x),
                            round((cy + dy * radius) * density.y),
                        )
                    )[:3]
                    assert sample == pytest.approx(expected, abs=2)
                window = imgui.internal.find_window_by_name("Pose")
                left = window.pos.x
                pixels.crop(
                    (
                        round(left * density.x),
                        round((y - 52 * scale) * density.y),
                        round((left + window.size.x) * density.x),
                        round((y + 140 * scale) * density.y),
                    )
                ).save(target / (state + "-detail.png"))
                report[state] = {"thumb": [cx, cy, radius], "rgb": expected, **flags}
            (target / "report.json").write_text(json.dumps(report, indent=2))
