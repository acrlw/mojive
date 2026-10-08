"""Capture actual native popup stacks and validate their rendered geometry."""

from __future__ import annotations

import json

from imgui_bundle import imgui

PATHS = (
    ("File",),
    ("Edit",),
    ("Add",),
    ("View",),
    ("Simulate",),
    ("Window",),
    ("Help",),
    ("Window", "Workspace"),
    ("Window", "Panels"),
    ("Window", "Inspector position"),
    ("View", "Projection"),
    ("View", "Transform tool"),
    ("View", "Gizmo orientation"),
    ("Camera",),
    ("Shading",),
    ("Playback",),
    ("Playback", "Playback speed"),
    ("Display settings",),
    ("Color",),
    ("Dialog",),
)


def capture_menus(window, study, destination, frames):
    destination.mkdir(parents=True, exist_ok=True)
    frames(window, study, destination / "closed.png")
    states = {}
    for path in PATHS:
        study.menus.preview(path)
        if path == ("Color",):
            study.inspector_tab = "Material"
        if path == ("Dialog",):
            study.menus.dialog_pending = "Interaction guide"
        slug = "-".join(path).lower().replace(" ", "-")
        frames(window, study, destination / (slug + ".png"))
        states[slug] = json.loads(json.dumps(study.menus.observed))
        assert "/".join(path) in states[slug], ("Popup not rendered", path, states[slug])
        vp = imgui.get_main_viewport()
        for popup in states[slug].values():
            x, y, w, h = popup["rect"]
            assert x >= vp.pos.x and y >= vp.pos.y, ("Popup outside viewport", popup)
            assert x + w <= vp.pos.x + vp.size.x + 1, ("Popup exceeds width", popup)
            assert y + h <= vp.pos.y + vp.size.y + 1, ("Popup exceeds height", popup)
            if popup["rows"] and vp.size.y >= 600:
                assert popup["scroll_max"] == 0, ("Unnecessary menu scrollbar", popup)
            for control, (cx, cy, cw, ch) in popup.get("controls", {}).items():
                assert x <= cx and y <= cy and cx + cw <= x + w and cy + ch <= y + h, (
                    "Popup control is clipped",
                    control,
                    popup,
                )
            for row in popup["rows"]:
                if not row["visible"]:
                    continue
                tx, _ty, tw, _th = row["label_rect"]
                assert tx + tw + 8 * study.s <= row["tail_x"], ("Overlapping menu columns", row)
                assert abs(row["rect"][3] - 30 * study.s) < 2, ("Inconsistent row height", row)
    study.menus.preview(())
    frames(window, study)
    actions = verify_actions(study)
    report = {
        "imgui": imgui.get_version(),
        "viewport": list(imgui.get_main_viewport().size),
        "scale": study.s,
        "states": states,
        "actions": actions,
        "checks": "All native popup stacks rendered; popup bounds, row heights and column clearance passed.",
        "method": "Real ImGui menu stacks opened through popup APIs; no synthetic pointer input.",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print(report["checks"])


def verify_actions(study):
    """Exercise effects that a menu row delegates to the document and viewport."""
    menus, preview = study.menus, study.preview
    original = preview.entity["hidden"]
    menus.dispatch("visibility")
    assert preview.entity["hidden"] != original
    menus.dispatch("undo")
    assert preview.entity["hidden"] == original
    menus.dispatch("redo")
    assert preview.entity["hidden"] != original
    menus.dispatch("undo")
    menus.dispatch("projection:orthographic")
    assert preview.camera.orthographic
    menus.dispatch("projection:perspective")
    assert not preview.camera.orthographic
    menus.dispatch("shading:Wireframe")
    menus.dispatch("shading:Solid")
    menus.dispatch("start")
    menus.dispatch("step:1")
    assert abs(study.time - 1 / 60) < 1e-9
    menus.dispatch("step:-1")
    menus.dispatch("step:-1")
    assert study.time == 0
    menus.dispatch("speed:0.5")
    menus.dispatch("play")
    menus.dispatch("step:1")
    assert study.playing and study.speed == 0.5 and study.time == 0
    menus.dispatch("play")
    menus.dispatch("loop")
    assert not study.loop
    menus.dispatch("loop")
    return "Visibility with Undo/Redo, projection, backend shading and playback actions passed."
