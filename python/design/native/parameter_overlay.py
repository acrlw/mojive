"""Visualize an edited parameter against the production geometry it controls."""

import itertools


def _rect(value):
    return tuple(float(v) for v in value) if value is not None and len(value) == 4 else None


def _edges(outer, inner, sides):
    x, y, w, h = outer
    ix, iy, iw, ih = inner
    areas = {
        "Left": (x, y, ix - x, h),
        "Right": (ix + iw, y, x + w - ix - iw, h),
        "Top": (x, y, w, iy - y),
        "Bottom": (x, iy + ih, w, y + h - iy - ih),
    }
    return [areas[side] for side in sides]


def _gaps(children, horizontal):
    """Measure adjacent children only; wrapped rows use their union as one item."""
    rects = sorted(children, key=lambda r: (round(r[1]), r[0]))
    rows = []
    for r in rects:
        if rows and r[1] < max(v[1] + v[3] for v in rows[-1]) - 0.5:
            rows[-1].append(r)
        else:
            rows.append([r])
    if horizontal:
        return [
            (
                a[0] + a[2],
                min(a[1], b[1]),
                b[0] - a[0] - a[2],
                max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1]),
            )
            for row in rows
            for a, b in itertools.pairwise(sorted(row))
            if b[0] >= a[0] + a[2] - 0.5
        ]
    bounds = [
        (
            min(r[0] for r in row),
            min(r[1] for r in row),
            max(r[0] + r[2] for r in row),
            max(r[1] + r[3] for r in row),
        )
        for row in rows
    ]
    return [
        (min(a[0], b[0]), a[3], max(a[2], b[2]) - min(a[0], b[0]), b[1] - a[3])
        for a, b in itertools.pairwise(bounds)
    ]


def parameter_regions(record, field):
    """Return measured affected rectangles; never synthesize a nominal gap."""
    exact = record.get("fieldRegions", {}).get(field)
    if exact is not None:
        return (
            [_rect(r) for r in exact]
            if exact and isinstance(exact[0], (tuple, list))
            else [_rect(exact)]
        )
    frame = _rect(record.get("frameRect")) or _rect(record["hit"])
    content = _rect(record.get("content")) or frame
    outer = _rect(record.get("outer") or record.get("marginBox")) or frame
    if field.startswith(("padding", "margin")):
        prefix = "padding" if field.startswith("padding") else "margin"
        suffix = field[len(prefix) :]
        sides = {"X": ("Left", "Right"), "Y": ("Top", "Bottom")}.get(suffix, (suffix,))
        if all(side in ("Left", "Right", "Top", "Bottom") for side in sides):
            return (
                _edges(frame, content, sides)
                if prefix == "padding"
                else _edges(outer, frame, sides)
            )
    if field == "rowGap" and record.get("spacingRect") is not None:
        return [_rect(record["spacingRect"])]
    if field in ("gap", "columnGap", "rowGap", "labelGap", "headerGap"):
        children = [_rect(r) for r in record.get("fields", ())]
        if not children:
            children = [
                _rect(record.get(key))
                for key in (
                    "iconRect",
                    "textRect",
                    "labelRect",
                    "fieldRect",
                    "switchRect",
                    "unitRect",
                    "shortcutRect",
                )
                if record.get(key) is not None
            ]
        if children:
            if record.get("kind") in ("form_row", "vector_row"):
                if field in ("columnGap", "labelGap"):
                    return [] if record.get("layout") == "top" else _gaps(children[:2], True)
                if field == "rowGap" and record.get("layout") == "left":
                    return []
            horizontal = field not in ("rowGap", "headerGap")
            areas = _gaps(children, horizontal)
            if field == "gap":
                areas += _gaps(children, False)
            return areas
        return []
    if field in ("left", "right", "top", "bottom") and record.get("anchorRect"):
        anchor = _rect(record["anchorRect"])
        return _edges(anchor, outer, (field.title(),))
    if field.startswith("switch"):
        return [_rect(record.get("switchRect")) or frame]
    if field == "iconSize":
        return [_rect(record.get("iconRect")) or content]
    if field in ("labelWidth", "labelTop"):
        return [_rect(record.get("labelRect")) or content]
    if field in ("fontSize", "textAlign", "lineHeight", "lineHeightMode"):
        return [_rect(record.get("textRect") or record.get("labelRect")) or content]
    return [frame]


def draw_parameter_overlay(draw, records, field, scale, excluded=None):
    from imgui_bundle import imgui as g

    from mojive.design.native.workbench import NativeWorkbench

    fill = g.get_color_u32((1.0, 0.68, 0.28, 0.23))
    stroke = g.get_color_u32((1.0, 0.72, 0.34, 0.95))
    regions = []
    for record in records:
        field = record.get("parameterField", field)
        for region in parameter_regions(record, field):
            if region is None or region[2] < 0 or region[3] < 0:
                continue
            x, y, w, h = region
            # Keep a zero-width anchor visible inside the native border clip.
            left, top, right, bottom = record["clip"]
            if w == 0:
                x = max(left, min(x, right - scale))
            if h == 0:
                y = max(top, min(y, bottom - scale))
            visible_rect = (x, y, max(scale, w), max(scale, h))
            for sx, sy, sw, sh in NativeWorkbench.visible_native_rects(
                visible_rect, record["clip"], record.get("overlayExclusion", excluded)
            ):
                draw.push_clip_rect((sx, sy), (sx + sw, sy + sh), True)
                draw.add_rect_filled((x, y), (x + max(w, scale), y + max(h, scale)), fill)
                draw.add_rect(
                    (x, y), (x + max(w, scale), y + max(h, scale)), stroke, thickness=scale
                )
                draw.pop_clip_rect()
                regions.append(
                    {
                        "key": record["key"],
                        "field": field,
                        "rect": list(region),
                        "visibleRect": [sx, sy, sw, sh],
                    }
                )
    return regions


def global_parameter_records(records, path, runtime):
    """Resolve shared fields through catalogue defaults and visible native owners."""
    if not path or path[0] == "parts":
        return []
    if path[0] == "layout":
        owners = {
            "application_height": (("titlebar",), "height"),
            "sidebar_width": (("rail",), "width"),
            "scene_width": (("panel-Scene",), "width"),
            "inspector_width": (("panel-Inspector",), "width"),
            "bottom_height": (
                ("panel-Timeline", "panel-Output", "panel-Statistics", "panel-Plot"),
                "height",
            ),
        }
        names, field = owners[path[-1]]
        return [
            {**record, "parameterField": field}
            for record in records
            if any(name in record["parts"] or record["key"] == name for name in names)
        ]
    if path[0] == "native":
        return [
            {**record, "parameterField": "radius"}
            for record in records
            if "radius" in record.get("fieldValues", {})
        ]
    token = path[-1]
    family = path[1] if path[0] == "components" else None
    if token == "font_family" or token.endswith("_color"):
        return [
            {**record, "parameterField": "fontSize" if token == "font_family" else "color"}
            for record in records
            if _record_family(record, family)
        ]

    def follows_token(part, field, visited=()):
        key = (part, field)
        if key in visited or field in runtime.spec.get("parts", {}).get(part, {}):
            return False
        rule = runtime.catalog.get(part, {}).get("fields", {}).get(field, {})
        default = rule.get("default")
        if not isinstance(default, dict):
            return False
        if "token" in default:
            return default["token"] == token
        if "field" in default:
            return follows_token(default.get("part", part), default["field"], (*visited, key))
        return False

    fields = {
        part: [field for field in spec["fields"] if follows_token(part, field)]
        for part, spec in runtime.catalog.items()
    }
    result = []
    for record in records:
        if not _record_family(record, family):
            continue
        matches = dict.fromkeys(field for part in record["parts"] for field in fields.get(part, ()))
        for field in matches:
            if record.get("styleOrigins", {}).get(field) == "preferences":
                continue
            result.append({**record, "parameterField": field})
    return result


def _record_family(record, family):
    if family is None:
        return True
    if record.get("styleFamily") is not None:
        return record["styleFamily"] == family
    from imgui_bundle import imgui as g

    window = g.internal.find_window_by_id(record.get("windowId", 0))
    if window is None:
        return False
    while window.parent_window is not None:
        window = window.parent_window
    root = window.name.split("###")[-1]
    return root in {
        "inspector": ("Inspector",),
        "timeline": ("Timeline",),
        "settings": ("Settings",),
        "output": ("Output",),
        "pose": ("Pose",),
        "navigation": ("Scene", "Cameras"),
        "viewport": ("Viewport",),
        "chrome": ("Application",),
        "menus": ("Application",),
    }.get(family, ())


def _window_owners(window):
    """Include child-window owners without crossing an independent popup boundary."""
    from imgui_bundle import imgui as g

    owners = set()
    while window is not None:
        owners.add(window.id_)
        if window.flags & g.WindowFlags_.popup:
            break
        window = window.parent_window
    return owners


def complete_child_geometry(targets, records):
    """Recover direct observed children for containers without a custom layout record."""
    from imgui_bundle import imgui as g

    owners = {}

    def belongs_to_window(record, target):
        record_id, target_id = record.get("windowId"), target.get("windowId")
        if target_id is None or record_id == target_id:
            return True
        if record_id not in owners:
            window = (
                g.internal.find_window_by_id(record_id)
                if record_id is not None and g.get_current_context() is not None
                else None
            )
            owners[record_id] = _window_owners(window)
        return target_id in owners[record_id]

    result = []
    for target in targets:
        if target.get("fields") or any(
            target.get(key) for key in ("iconRect", "textRect", "labelRect")
        ):
            result.append(target)
            continue
        x, y, w, h = target.get("content", target["hit"])
        candidates = {}
        for record in records:
            # Popups overlap their parent windows on screen; geometry alone cannot
            # establish ownership of the controls behind an open popup.
            if not belongs_to_window(record, target):
                continue
            rect = record["hit"]
            rx, ry, rw, rh = rect
            if record["key"] == target["key"] or rw * rh >= w * h * 0.95 or rw <= 0 or rh <= 0:
                continue
            if (
                x - 0.5 <= rx
                and y - 0.5 <= ry
                and rx + rw <= x + w + 0.5
                and ry + rh <= y + h + 0.5
            ):
                candidates[tuple(rect)] = rect
        children = []
        for rect in sorted(candidates, key=lambda r: -r[2] * r[3]):
            rx, ry, rw, rh = rect
            if any(
                a <= rx and b <= ry and rx + rw <= a + c and ry + rh <= b + d
                for a, b, c, d in children
            ):
                continue
            children.append(rect)
        result.append({**target, "fields": children, "inferredChildren": True})
    return result
