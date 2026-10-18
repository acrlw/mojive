"""Observe clipped production instances by scrolling their real native owner."""

from copy import deepcopy

import numpy as np
from imgui_bundle import imgui

from mojive.tools.design_audit_evidence import identity

RECTANGLES = frozenset(
    (
        "hit",
        "frame",
        "content",
        "outer",
        "layoutRect",
        "spacingRect",
        "textBlockRect",
        "marginBox",
        "textRect",
        "titleRect",
        "closeRect",
        "iconRect",
        "numericRect",
        "labelRect",
        "trackRect",
        "thumbRect",
        "destination",
    )
)
LEAF_CONTAINERS = frozenset(
    ("stat_row", "output_row", "material", "sensor", "menu_item", "filter", "badge", "lane_label")
)


def scroll_positions(workbench):
    return {
        window.id_: (window.scroll.x, window.scroll.y)
        for window in imgui.get_current_context().windows
        if window.active and not workbench._editor_window(window)
    }


def translate_observation(record, delta):
    """Retain screen geometry separately; compare effects in the original scroll coordinates."""
    normalized = deepcopy(record)
    for field in RECTANGLES:
        rectangle = normalized.get(field)
        if isinstance(rectangle, (tuple, list)) and len(rectangle) == 4:
            normalized[field] = [rectangle[0] + delta[0], rectangle[1] + delta[1], *rectangle[2:]]
    if "fields" in normalized:
        normalized["fields"] = [
            [child[0] + delta[0], child[1] + delta[1], *child[2:]] for child in normalized["fields"]
        ]
    return normalized


def partial_keys(state):
    """A visible background sliver is insufficient evidence for its clipped text."""
    keys = set()
    for record in state["records"]:
        if identity(record) in state.get("recoveredKeys", ()):
            continue
        if record.get("kind") in ("container", "group", "table", "textarea") and not (
            LEAF_CONTAINERS.intersection(record.get("parts", ()))
        ):
            continue
        rectangle = record.get("content") or record.get("trackRect")
        if rectangle is None and record.get("kind") in (
            "track",
            "range",
            "switch",
            "checkbox",
            "button",
            "search",
            "input_number",
            "input_text",
            "axis_input",
        ):
            rectangle = record["hit"]
        if rectangle is None:
            continue
        if rectangle[1] < record["clip"][1] - 0.5 or (
            rectangle[1] + rectangle[3] > record["clip"][3] + 0.5
        ):
            keys.add(identity(record))
    return keys


def recover_instances(audit, before, after, effect, *, additional_targets=()):
    """Revisit every lost baseline row without treating a scroll offset as parameter motion."""
    targets = set(effect["removedKeys"]) | set(effect["missingTextKeys"]) | set(additional_targets)
    if not targets:
        return []
    original = {identity(record): record for record in before["records"]}
    raw = {identity(record): record for record in after["rawRecords"]}
    original_scroll = scroll_positions(audit.bench)
    recovery = []
    previous_recovery = getattr(audit, "recovering_visible_instances", False)
    audit.recovering_visible_instances = True
    try:
        for key in sorted(targets):
            record = original[key]
            current = raw.get(key)
            window = next(
                (
                    window
                    for window in imgui.get_current_context().windows
                    if window.active and window.id_ == record.get("windowId")
                ),
                None,
            )
            evidence = {"instance": key, "visibleBeforeScroll": False}
            if window is None:
                evidence.update(reason="生产窗口已关闭，无法通过滚动再次观察。")
                recovery.append(evidence)
                continue
            node_id = record["key"].rsplit("-", 1)[-1]
            navigation = getattr(audit.viewer.app.workspace, "navigation", None)
            virtual_row = "Scene entities" in window.name and node_id.isdecimal()
            scanned = None
            if virtual_row and navigation is not None:
                node_id = int(node_id)
                if not any(node.node_id == node_id for node in audit.viewer.session.nodes):
                    evidence.update(reason="对应场景节点已不存在。")
                    recovery.append(evidence)
                    continue
                navigation._scroll_scene_node = node_id
            elif current is not None and window.scroll_max.y > 0:
                # All submitted rectangles came from the initial scroll state.
                # Earlier recovered instances must not add their new offset
                # to the next item's original screen position.
                target = original_scroll[window.id_][1] + current["hit"][1] - current["clip"][1] - 8
                imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, target)))
            elif window.scroll_max.y > 0:
                # Virtual lists intentionally omit offscreen records. Search
                # by the observed identity through the real native scroll
                # range instead of guessing a joint's new height or position.
                pitch = max(64, (window.clip_rect.max.y - window.clip_rect.min.y) * 0.6)
                start = original_scroll[window.id_][1]
                offsets = [start, min(window.scroll_max.y, start + pitch), max(0, start - pitch)]
                offsets.extend(np.arange(0, window.scroll_max.y, pitch).tolist())
                offsets.append(window.scroll_max.y)
                evidence["searchedScrollOffsets"] = []
                for offset in dict.fromkeys(offsets):
                    imgui.internal.set_scroll_y(window, offset)
                    audit.input.frames(2)
                    observed, stable, _ = audit.settled_snapshot()
                    evidence["searchedScrollOffsets"].append(offset)
                    actual = next(
                        (item for item in observed["records"] if identity(item) == key), None
                    )
                    if actual is not None and key not in partial_keys(observed):
                        scanned = observed, stable
                        break
                if scanned is None:
                    evidence.update(reason="遍历真实原生滚动范围后，该实例仍未完整显示。")
                    recovery.append(evidence)
                    continue
            else:
                evidence.update(reason="没有可验证的原生滚动入口或此实例的提交记录。")
                recovery.append(evidence)
                continue
            if scanned is None:
                audit.input.frames(3)
                observed, stable, _ = audit.settled_snapshot()
            else:
                observed, stable = scanned
            actual = next((item for item in observed["records"] if identity(item) == key), None)
            evidence.update(stable=stable, scrollBefore=original_scroll.get(window.id_))
            if actual is None:
                evidence.update(reason="实际滚动后该实例仍未进入可见区域。")
                recovery.append(evidence)
                continue
            current_scroll = scroll_positions(audit.bench).get(window.id_, (0, 0))
            previous_scroll = before.get("productionScroll", original_scroll).get(
                window.id_, (0, 0)
            )
            delta = tuple(current_scroll[i] - previous_scroll[i] for i in (0, 1))
            evidence.update(
                visibleBeforeScroll=True,
                physicalGeometry=actual,
                scrollObserved=current_scroll,
                normalizationOffset=delta,
            )
            normalized = translate_observation(actual, delta)
            after["records"] = [item for item in after["records"] if identity(item) != key] + [
                normalized
            ]
            after["allRecords"] = [
                item for item in after["allRecords"] if identity(item) != key
            ] + [normalized]
            vertices = observed["instanceVertices"][key].copy()
            vertices[:, :2] += delta
            after["instanceVertices"][key] = vertices
            after["drawSources"][key] = observed["drawSources"][key]
            recovery.append(evidence)
            after.setdefault("recoveredKeys", set()).add(key)
    finally:
        audit.recovering_visible_instances = previous_recovery
        for window in imgui.get_current_context().windows:
            if window.active and window.id_ in original_scroll:
                x, y = original_scroll[window.id_]
                imgui.internal.set_scroll_x(window, x)
                imgui.internal.set_scroll_y(window, y)
        audit.input.frames(3)
    after["vertices"] = (
        np.concatenate(
            [after["instanceVertices"][key] for key in sorted(after["instanceVertices"])]
        )
        if after["instanceVertices"]
        else np.empty((0, 6))
    )
    return recovery
