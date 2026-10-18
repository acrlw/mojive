"""Capture section/body alignment in the Guide's real production presenters."""

import json
from copy import deepcopy
from math import floor

from imgui_bundle import imgui as g
from PIL import Image

from mojive import commands as cmd
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.section_columns import current_section_column
from mojive.ui.workspace.settings_panel import CATEGORIES


def alignment_geometry(records, text_records):
    """Associate text with its actual native window owner, independently of row Y."""
    columns = [record for record in records if record["kind"] == "section_column"]
    sections = [record for record in records if record["kind"] == "section_header"]
    by_window = {}
    for column in columns:
        by_window.setdefault(column["windowId"], []).append(column)
    errors, captions = [], []
    for section in sections:
        owner = section.get("columnOwner")
        if owner is None:
            continue
        body = section["bodyColumn"]
        candidates = by_window.get(section["windowId"], ())
        epsilon = native_epsilon(*section["hit"], *body)
        if not any(
            column["key"] == owner
            and abs(column["content"][0] - body[0]) <= epsilon
            and abs(column["content"][2] - body[2]) <= epsilon
            for column in candidates
        ):
            errors.append(section["key"] + ": no matching native section column")
        if abs(section["labelRect"][0] - body[0]) > epsilon:
            errors.append(section["key"] + ": heading and body origins differ")
    for record in records:
        if record["kind"] == "label" and "form_label" in record["parts"]:
            rect = record["content"]
        elif record["kind"] in ("switch", "checkbox") and record.get("labelRect"):
            rect = record["labelRect"]
        else:
            continue
        if record.get("windowId") not in by_window:
            continue
        owner, body = record.get("columnOwner"), record.get("bodyColumn")
        if owner is None or body is None:
            errors.append(record["key"] + ": body label has no record-time column owner")
            continue
        candidates = [
            column
            for column in by_window.get(record.get("windowId"), ())
            if column["key"] == owner
            and all(
                abs(actual - expected) <= native_epsilon(*body, *column["content"])
                for actual, expected in zip(body, column["content"], strict=True)
            )
        ]
        if not candidates:
            errors.append(record["key"] + ": no matching record-time section column")
            continue
        column = candidates[-1]
        epsilon = native_epsilon(*rect, *column["content"])
        captions.append(
            {
                "key": record["key"],
                "windowId": record["windowId"],
                "columnOwner": column["key"],
                "content": list(rect),
                "column": list(column["content"]),
                "parts": record["parts"],
                "clip": list(record["clip"]),
            }
        )
        if rect[0] < column["content"][0] - epsilon:
            errors.append(record["key"] + ": body label protrudes left of its column")
    for record in text_records:
        epsilon = native_epsilon(record["x"], *record["column"])
        if record["x"] < record["column"][0] - epsilon:
            errors.append(record["text"] + ": painted text protrudes left of its column")
    pose_cards, pose_labels = [], []
    for card in records:
        if not card["key"].startswith("joint-card-") or not card.get("bodyColumn"):
            continue
        body, heading = card["bodyColumn"], card.get("headingContentColumn")
        epsilon = native_epsilon(*body, *card["hit"])
        if heading is not None and (
            abs(heading[0] - body[0]) > epsilon or abs(heading[2] - body[2]) > epsilon
        ):
            errors.append(card["key"] + ": pose heading and body columns differ")
        pose_cards.append(card)
        joint_id = card["key"].removeprefix("joint-card-")
        prefixes = (f"pose-summary-label-{joint_id}-", f"##pose-joint-{joint_id}-")
        for label in records:
            if (
                label.get("windowId") != card.get("windowId")
                or label["kind"] != "label"
                or not label["key"].startswith(prefixes)
            ):
                continue
            rect = label.get("content", label.get("labelRect", label["hit"]))
            pose_labels.append({"key": label["key"], "content": rect, "bodyColumn": body})
            if rect[0] < body[0] - native_epsilon(*rect, *body):
                errors.append(label["key"] + ": pose label protrudes left of its body column")
    status = [
        record
        for record in records
        if record["key"] in ("application-status_physics", "application-status_render")
    ]
    for record in status:
        value, label, content = record["valueTextRect"], record["labelRect"], record["content"]
        epsilon = native_epsilon(*value, *content)
        if abs(record["unitX"] - value[0] - value[2] - record["unitGap"]) > epsilon:
            errors.append(record["key"] + ": unit is detached from its displayed value")
        if abs(record["unitRight"] + record["trailingReserve"] - content[0] - content[2]) > epsilon:
            errors.append(record["key"] + ": trailing value reserve exceeds its content budget")
        if record["compact"] and abs(value[0] - label[0] - label[2] - record["labelGap"]) > epsilon:
            errors.append(record["key"] + ": value does not follow its label gap")
    return {
        "columns": columns,
        "sections": sections,
        "bodyLabels": captions,
        "paintedText": text_records,
        "poseCards": pose_cards,
        "poseLabels": pose_labels,
        "statusReadouts": status,
        "menuRows": [record for record in records if "checkRect" in record],
        "menuTriggers": [record for record in records if "menu_trigger" in record["parts"]],
        "errors": errors,
    }


def capture_alignment(v):
    """Save native panel states and require common heading/body leading edges."""
    previous = v.page, deepcopy(v.values), deepcopy(v.selected), v.animate, v.clock
    observed = v.runtime.observe_components
    process, queued = v.window._input.process_inputs, []
    original_text, painted = v.ui.text, []
    design = primitives = original_record = original_primitive_record = None
    report = {
        "scale": v.s,
        "method": "Production native windows, section ownership and actual painted text origins",
        "captures": {},
        "scrollCoverage": {},
        "errors": [],
    }
    target = v.output / "alignment-geometry.json"
    selection, pose_state, scrolls = None, None, {}

    def record_with_owner(original):
        def record(*args, **geometry):
            column = current_section_column(v.ui)
            if column is not None:
                geometry.setdefault("columnOwner", "section-column/" + str(column.window_id))
                geometry.setdefault("bodyColumn", list(column.content.tuple()))
            return original(*args, **geometry)

        return record

    def inputs():
        process()
        for method, args in queued:
            getattr(g.get_io(), method)(*args)
        queued.clear()
        g.get_io().delta_time = 1 / 60

    def text(x, y, value, *args, **options):
        column = current_section_column(v.ui)
        if column is not None:
            painted.append(
                {
                    "text": str(value),
                    "x": x,
                    "y": y,
                    "windowId": column.window_id,
                    "columnOwner": "section-column/" + str(column.window_id),
                    "column": list(column.content.tuple()),
                }
            )
        return original_text(x, y, value, *args, **options)

    def frame(*, readback=False):
        painted.clear()
        return v.frame(readback=readback)

    def event(method, *args):
        queued.append((method, args))
        frame()

    def click(key):
        x, y, width, height = v.hits.get(key, v.ui.hits.get(key))
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        for _ in range(3):
            frame()

    def windows():
        return [
            window
            for window in g.get_current_context().windows
            if window.active
            and (
                "Native specimen/" + v.page in window.name or "Guide page/" + v.page in window.name
            )
        ]

    def show(page):
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        v.page = page
        event("add_mouse_pos_event", 0, 0)
        for _ in range(3):
            frame()
        for window in windows():
            scrolls.setdefault(window.id_, (window.scroll.x, window.scroll.y))
            g.internal.set_scroll_y(window, 0)
        for _ in range(4):
            frame()

    def owner_windows(geometry):
        identities = {
            record["windowId"] for record in (*geometry["columns"], *geometry["poseCards"])
        }
        return [window for window in windows() if window.id_ in identities]

    def scroll_geometry(window):
        clip, inner = window.inner_clip_rect, window.inner_rect
        return {
            "windowId": window.id_,
            "name": window.name,
            "scrollY": window.scroll.y,
            "scrollMaxY": window.scroll_max.y,
            "innerY": [inner.min.y, inner.max.y],
            "clipY": [clip.min.y, clip.max.y],
            "visibleHeight": max(0, clip.max.y - clip.min.y),
            "contentInterval": [
                window.scroll.y + clip.min.y - inner.min.y,
                window.scroll.y + clip.max.y - inner.min.y,
            ],
        }

    def reveal_owner(window):
        """Reveal the specimen once before scrolling its actual native content owner."""
        trace, ancestor = [], window.parent_window
        while ancestor is not None:
            delta = max(0, window.inner_rect.max.y - ancestor.inner_clip_rect.max.y)
            target_y = min(ancestor.scroll_max.y, ancestor.scroll.y + delta)
            if target_y > ancestor.scroll.y + native_epsilon(target_y, ancestor.scroll.y):
                scrolls.setdefault(ancestor.id_, (ancestor.scroll.x, ancestor.scroll.y))
                trace.append({**scroll_geometry(ancestor), "requestedY": target_y})
                g.internal.set_scroll_y(ancestor, target_y)
                for _ in range(4):
                    frame()
            ancestor = ancestor.parent_window
        return trace

    def visible_row(rect, clip):
        epsilon = native_epsilon(*rect, *clip)
        return clip[1] - epsilon <= rect[1] and rect[1] + rect[3] <= clip[3] + epsilon

    def save(name, *, required=True, bottom=False):
        for _ in range(4):
            pixels = frame(readback=True)
        Image.fromarray(pixels[::-1]).save(v.output / (name + ".png"))
        records = deepcopy([*v.runtime.observed, *v.ui.design.primitives.observed])
        geometry = alignment_geometry(records, deepcopy(painted))
        if required and not (geometry["columns"] or geometry["poseCards"]):
            geometry["errors"].append("No production section or pose content column was observed")
        report["captures"][name] = {
            "image": name + ".png",
            "page": v.page,
            "pixelSize": [pixels.shape[1], pixels.shape[0]],
            "scrollOwners": [scroll_geometry(window) for window in owner_windows(geometry)],
            **geometry,
        }
        report["errors"].extend(name + ": " + error for error in geometry["errors"])
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if bottom:
            owners = owner_windows(geometry)
            assert len(owners) == 1, "A specimen must have one native content scroll owner"
            owner, tiles = owners[0], [name]
            ancestor_trace = reveal_owner(owner)
            if ancestor_trace:
                tiles.append(name + "-tile-00")
                save(tiles[-1], required=required)
            index = 1
            while owner.scroll.y < owner.scroll_max.y - native_epsilon(owner.scroll_max.y):
                visible = owner.inner_clip_rect.max.y - owner.inner_clip_rect.min.y
                assert visible > 1, "The specimen content owner is not visible"
                target_y = min(owner.scroll_max.y, owner.scroll.y + max(1, floor(visible * 0.8)))
                previous_y = owner.scroll.y
                g.internal.set_scroll_y(owner, target_y)
                ending = target_y == owner.scroll_max.y
                tiles.append(name + ("-bottom" if ending else f"-tile-{index:02d}"))
                save(tiles[-1], required=required)
                assert owner.scroll.y > previous_y, "Native content scrolling did not advance"
                index += 1
            captures = [report["captures"][tile] for tile in tiles]
            expected = {record["key"] for capture in captures for record in capture["sections"]}
            visible_sections = {
                record["key"]
                for capture in captures
                for record in capture["sections"]
                if visible_row(record["labelRect"], record["clip"])
            }
            missing = sorted(expected - visible_sections)
            if missing:
                report["errors"].append(
                    name + ": section headings have no visible tile: " + str(missing)
                )
            body_keys = {record["key"] for capture in captures for record in capture["bodyLabels"]}
            visible_body = {
                record["key"]
                for capture in captures
                for record in capture["bodyLabels"]
                if visible_row(record["content"], record["clip"])
            }
            missing_body = sorted(body_keys - visible_body)
            if missing_body:
                report["errors"].append(
                    name + ": body labels have no visible tile: " + str(missing_body)
                )
            report["scrollCoverage"][name] = {
                "windowId": owner.id_,
                "name": owner.name,
                "stepFraction": 0.8,
                "ancestorReveal": ancestor_trace,
                "tiles": tiles,
                "positions": [capture["scrollOwners"] for capture in captures],
                "sectionKeys": sorted(expected),
                "visibleSectionKeys": sorted(visible_sections),
                "missingSectionKeys": missing,
                "bodyLabelKeys": sorted(body_keys),
                "visibleBodyLabelKeys": sorted(visible_body),
                "missingBodyLabelKeys": missing_body,
            }
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def select(name):
        node = next(node for node in v.sample.app.session.nodes if node.name == name)
        result = v.sample.app.session.submit(cmd.SelectNode(node.node_id))
        assert result.ok, result.message

    v.window._input.process_inputs, v.ui.text = inputs, text
    v.runtime.observe_components, v.animate = True, False
    try:
        frame()
        # GuideWorkspace installs the production presentation during its first
        # frame; observe that instance, rather than the Guide's initial painter.
        design, primitives = v.ui.design, v.ui.design.primitives
        original_record, original_primitive_record = design.record, primitives.record
        design.record = record_with_owner(original_record)
        primitives.record = record_with_owner(original_primitive_record)
        session, workspace = v.sample.app.session, v.sample.app.workspace
        selection = session.selected_node.node_id if session.selected_node is not None else None
        for index, category in enumerate(CATEGORIES):
            v.selected["Settings navigation"] = index
            show("Settings")
            save("alignment-settings-" + category.lower().replace(" ", "-"), bottom=True)
        for size in ("Wide", "Narrow"):
            v.values["inspector-size"] = size
            for index, (tab, node) in enumerate(
                (
                    ("properties", "hinge_body"),
                    ("physics", "hinge_body"),
                    ("material", "free_box"),
                    ("joint", "hinge_limited"),
                )
            ):
                select(node)
                v.selected["Inspector tabs"] = index
                show("Inspector")
                save(f"alignment-inspector-{tab}-{size.lower()}", bottom=True)
        for page in ("Cameras", "Layers", "Control", "Assets"):
            v.values["camera-size"] = "Wide"
            show(page)
            save("alignment-" + page.lower(), bottom=True)
        assert session.submit(cmd.Select(0)).ok
        v.values["pose-size"] = "Wide"
        show("Pose")
        editor = workspace.navigation.pose_editor
        pose_state = editor, deepcopy(editor._expanded), deepcopy(editor._selection)
        joint = next(joint for joint in session.joints if joint.type == "free")
        expand = f"pose-joint-{joint.joint_id}-expand"
        summary = f"pose-summary-{joint.joint_id}-position-X"
        if summary not in v.ui.hits:
            click(expand)
        save("alignment-pose-folded", bottom=True)
        show("Pose")
        click(expand)
        save("alignment-pose-expanded-wide", bottom=True)
        v.values["pose-size"] = "Narrow"
        show("Pose")
        save("alignment-pose-expanded-narrow", bottom=True)
        show("Status")
        save("alignment-status", required=False)
        show("Menus")
        click("sample-menu-File")
        save("alignment-menu-file", required=False)
        for name, capture in report["captures"].items():
            if name.startswith("alignment-inspector-material-"):
                # The native material shortcut paints its caption directly;
                # RGBA fields have axis labels rather than form-label records.
                captions = [text for text in capture["paintedText"] if text["text"] == "Base color"]
                channels = {
                    text["text"]
                    for text in capture["paintedText"]
                    if text["text"] in ("R", "G", "B", "A")
                }
                capture["rawBodyCaptions"] = captions
                capture["rawChannelLabels"] = sorted(channels)
                if not captions or channels != set("RGBA"):
                    report["errors"].append(
                        name + ": native material caption or RGBA labels missing"
                    )
                for caption in captions:
                    if abs(caption["x"] - caption["column"][0]) > native_epsilon(
                        caption["x"], *caption["column"]
                    ):
                        report["errors"].append(
                            name + ": material caption differs from body origin"
                        )
                continue
            if (
                name.startswith(("alignment-settings-", "alignment-inspector-"))
                and not capture["bodyLabels"]
            ):
                report["errors"].append(name + ": no production body labels were observed")
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        assert not report["errors"], "Native section alignment failed; see " + str(target)
        return report
    finally:
        try:
            if v.sample is not None:
                event("add_key_event", g.Key.escape, True)
                event("add_key_event", g.Key.escape, False)
                command = cmd.SelectNode(selection) if selection is not None else cmd.Select(0)
                v.sample.app.session.submit(command)
            if pose_state is not None:
                editor, expanded, selected = pose_state
                editor._expanded.clear()
                editor._expanded.update(expanded)
                editor._selection.clear()
                editor._selection.update(selected)
            for window in g.get_current_context().windows:
                if window.id_ in scrolls:
                    x, y = scrolls[window.id_]
                    g.internal.set_scroll_x(window, x)
                    g.internal.set_scroll_y(window, y)
        finally:
            v.window._input.process_inputs, v.ui.text = process, original_text
            if design is not None:
                design.record, primitives.record = original_record, original_primitive_record
            v.page, values, selected, v.animate, v.clock = previous
            v.values.clear()
            v.values.update(values)
            v.selected.clear()
            v.selected.update(selected)
            v.runtime.observe_components = observed
