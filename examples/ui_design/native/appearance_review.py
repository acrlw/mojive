"""Capture native editor pages and check their visible control geometry."""

from __future__ import annotations

import json
from pathlib import Path

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.viewport_widgets import localized_viewport_labels

from .settings_panel import CATEGORIES


def _window_rect(name):
    window = imgui.internal.find_window_by_name(name)
    assert window is not None and window.active, ("Review window is not visible", name)
    return [window.pos.x, window.pos.y, window.size.x, window.size.y]


def _check_controls(controls, owner, *, tolerance=1.0):
    """Check one peer group; parent controls are checked separately."""
    x, y, w, h = owner
    checked = []
    for key, rect in controls.items():
        cx, cy, cw, ch = rect
        assert cw > 0 and ch > 0, ("Empty control", key, rect)
        assert (
            cx >= x - tolerance
            and cy >= y - tolerance
            and cx + cw <= x + w + tolerance
            and cy + ch <= y + h + tolerance
        ), ("Control exceeds its window", key, rect, owner)
        for other_key, (ox, oy, ow, oh) in checked:
            assert (
                cx + cw <= ox + tolerance
                or ox + ow <= cx + tolerance
                or cy + ch <= oy + tolerance
                or oy + oh <= cy + tolerance
            ), ("Peer controls overlap", key, other_key, rect, (ox, oy, ow, oh))
        checked.append((key, rect))
    return len(checked)


def _settings_size(study, width, height):
    window = imgui.internal.find_window_by_name("Settings")
    assert window is not None and window.dock_id == 0, "Settings must float for size review"
    viewport = imgui.get_main_viewport()
    s = study.s
    width = min(width * s, viewport.size.x - 32 * s)
    height = min(height * s, viewport.size.y - 32 * s)
    imgui.internal.set_window_size(window, (width, height))
    imgui.internal.set_window_pos(
        window,
        (
            viewport.pos.x + (viewport.size.x - width) / 2,
            viewport.pos.y + (viewport.size.y - height) / 2,
        ),
    )


def capture_appearance(window, study, output, frames):
    """Render real Study pages without retaining scene or preference changes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames(window, study)
    app, session = study.preview, study.preview.session
    original = {
        "layout": imgui.save_ini_settings_to_memory(),
        "open": dict(study.open),
        "workspace": study.workspace,
        "inspector_tab": study.inspector_tab,
        "pinned": study.pinned,
        "tool": app.tool,
        "category": study.settings._category,
        "language": app.localizer.language,
        "viewport_labels": app._viewport_labels,
        "selection": session.selected_node.node_id if session.selected_node is not None else None,
        "node_count": len(session.nodes),
    }
    states = {}
    created = False

    def set_language(language):
        app.localizer.set_language(language, persist=False)
        app._viewport_labels = localized_viewport_labels(app.localizer.text)

    def capture(name, *, inspector_fields=False, settings=False):
        snapshot = frames(window, study, output / f"{name}.png")
        controls = {key: list(rect) for key, rect in study.ui.hits.items()}
        checks = {}
        if inspector_fields:
            entity = app.entity
            numeric = {
                key: controls[key]
                for component in ("position", "rotation", "scale")
                for axis in "XYZ"
                if (key := f"##{entity['id']}{component}{axis}") in controls
            }
            assert len(numeric) == 9, ("Editable transform fields are missing", numeric)
            checks["inspector_fields"] = _check_controls(
                numeric, _window_rect("Inspector"), tolerance=study.s
            )
        if settings:
            navigation = {
                f"settings-{category}": controls[f"settings-{category}"]
                for category in CATEGORIES
                if f"settings-{category}" in controls
            }
            assert len(navigation) == len(CATEGORIES), (
                "Settings categories are missing",
                navigation,
            )
            owner = _window_rect("Settings")
            checks["settings_categories"] = _check_controls(navigation, owner, tolerance=study.s)
            viewport = imgui.get_main_viewport()
            _check_controls(
                {"Settings": owner},
                [viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y],
                tolerance=study.s,
            )
        states[name] = {
            "snapshot": snapshot,
            "language": app.localizer.language.value,
            "controls": controls,
            "checks": checks,
        }

    def restore_selection():
        result = session.submit(
            cmd.SelectNode(original["selection"])
            if original["selection"] is not None
            else cmd.Select(0)
        )
        assert result.ok, result.message

    try:
        set_language("en")
        study.open["Settings"] = False
        study.pinned = None
        study.inspector_tab = "Properties"
        study.preset("Edit")
        capture("edit")

        study.menus.dispatch("create:Box")
        created = len(session.nodes) > original["node_count"]
        assert created, ("Temporary box was not created", session.last_message)
        frames(window, study)
        entity = app.entity
        assert entity is not None and entity["posable"] and entity["scalable"], entity
        app.tool = "move"
        capture("inspector-editable", inspector_fields=True)
        result = session.submit(cmd.Undo())
        assert result.ok, result.message
        created = False
        assert len(session.nodes) == original["node_count"], "Temporary box was not removed"
        restore_selection()
        app.tool = original["tool"]

        for panel in ("Control", "Assets", "Pose"):
            study.activate(panel)
            capture(panel.lower())
        study.activate("Scene")
        frames(window, study)
        study.activate("Settings")
        study.new_panel = "Settings"
        frames(window, study)
        _settings_size(study, 820, 620)
        for language, suffix in (("en", "en"), ("zh_CN", "zh")):
            set_language(language)
            for category in CATEGORIES:
                study.settings.show_category(category)
                slug = category.lower().replace(" ", "-")
                capture(f"settings-{slug}-{suffix}", settings=True)
        study.settings.show_category("General")
        _settings_size(study, 420, 600)
        capture("settings-narrow-zh", settings=True)
    finally:
        if created:
            result = session.submit(cmd.Undo())
            assert result.ok, result.message
        restore_selection()
        app.localizer.set_language(original["language"], persist=False)
        app._viewport_labels = original["viewport_labels"]
        app.tool = original["tool"]
        study.open.update(original["open"])
        study.workspace = original["workspace"]
        study.inspector_tab = original["inspector_tab"]
        study.pinned = original["pinned"]
        study.settings.show_category(original["category"])
        imgui.load_ini_settings_from_memory(original["layout"])
        frames(window, study)

    report = {
        "imgui": imgui.get_version(),
        "scale": study.s,
        "states": states,
        "checks": "Viewport overlays, Settings navigation and editable transform field bounds passed.",
        "method": "Real Study pages and Session authoring; language changes remain in memory.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(report["checks"], output)
