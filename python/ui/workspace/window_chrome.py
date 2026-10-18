"""Style native window titles and docking tabs within their actual draw boundary."""

from contextlib import contextmanager, nullcontext
from math import floor, trunc

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.panel_geometry import floating_panel_geometry


def native_child_extent(origin, extent):
    """Keep a child's native endpoint when BeginChild truncates its origin."""
    endpoint = origin + extent
    integral_endpoint = round(endpoint)
    if abs(endpoint - integral_endpoint) <= native_epsilon(origin, extent, endpoint):
        endpoint = integral_endpoint
    return trunc(endpoint) - trunc(origin)


def fit_native_content_width(window):
    """Keep native width queries within this window's own paintable viewport."""
    # ImGui floors the inner clip edge after allocating a fractional scrollbar
    # slot. Its work/content rectangles retain that fraction, which can allocate
    # a full-width control into an unpaintable strip. Do not use InnerClipRect:
    # its ancestor intersection would make controls reflow as the parent scrolls.
    right = floor(window.inner_rect.max.x - window.window_border_size * 0.5)
    content_right = right - window.scroll.x - window.window_padding.x
    for rect in (window.work_rect, window.parent_work_rect, window.content_region_rect):
        rect.max.x = min(rect.max.x, content_right)
    return right


def surface_ink(ui, part, role):
    """Resolve a surface role before its independent component color override."""
    style = ui.component_style(part)
    return ui.style_color(style, "surfaceColor", ui.ink(rgba8(role), style), inherit_alpha=False)


@contextmanager
def panel_surface(ui, *, docked, name=None, floating=None):
    """Separate floating windows from docked content without outlining every panel."""
    floating = not docked if floating is None else floating
    geometry = floating_panel_geometry(ui, name)
    background = surface_ink(ui, "panel", "floating" if floating else "background")
    border = ui.ink(rgba8("border"))
    imgui.push_style_color(imgui.Col_.window_bg, tuple(value / 255 for value in background))
    imgui.push_style_color(imgui.Col_.border, tuple(value / 255 for value in border))
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0 if docked else ui.s)
    rounding = geometry.radius if floating else 0
    if docked and floating:
        root = floating_dock_root(imgui.internal.find_window_by_name(name)) if name else None
        rounding = (
            root.host_window.window_rounding
            if root is not None and root.host_window is not None
            else imgui.get_style().window_rounding
        )
    imgui.push_style_var(imgui.StyleVar_.window_rounding, rounding)
    if floating:
        imgui.push_style_var(imgui.StyleVar_.window_padding, (geometry.padding, geometry.padding))
    try:
        yield background
    finally:
        imgui.pop_style_var(3 if floating else 2)
        imgui.pop_style_color(2)


def floating_dock_root(window):
    """A tabbed floating group is docked to its own host, outside the main dockspace."""
    if window is None or window.dock_node is None:
        return None
    root = imgui.internal.dock_node_get_root_node(window.dock_node)
    return root if root.is_floating_node() else None


@contextmanager
def panel_content_corners(ui, name, window):
    """Use the actual group host when deriving a floated member's child corners."""
    if window.dock_id and floating_dock_root(window) is None:
        yield
        return
    radius = max(0, window.window_rounding - max(window.window_padding))
    parts = (
        ("timeline_body",)
        if name == "Timeline"
        else (
            "button",
            "icon_button",
            "select",
            "input_text",
            "input_number",
            "textarea",
            "settings_nav",
            "segment",
            "tab",
        )
    )
    with ui.corner_scope(radius, parts):
        yield


def native_window_geometry(window):
    """Observe the painted native owner separately from its component content box."""
    root = floating_dock_root(window)
    host = root.host_window if root is not None else None
    title, content = window.title_bar_rect(), window.work_rect
    return {
        "nativeWindowRadius": window.window_rounding,
        "nativeWindowPadding": list(window.window_padding),
        "nativeTitleBarRect": [title.min.x, title.min.y, title.get_width(), title.get_height()],
        "nativeContentRect": [
            content.min.x,
            content.min.y,
            content.get_width(),
            content.get_height(),
        ],
        "floatingDockHostId": host.id_ if host is not None else None,
        "floatingDockHostRadius": host.window_rounding if host is not None else None,
        "floatingDockHostRect": [*host.pos, *host.size] if host is not None else None,
        "nativeRadiusPolicy": "shared-floating-host"
        if root is not None
        else "docked"
        if window.dock_id
        else "individual-floating-window",
    }


@contextmanager
def floating_dock_hosts(workspace):
    """Style hosts created by NewFrame, before any panel's Begin can style itself.

    ImGui consumes undocking requests before creating hosts in one internal pass;
    its pending requests are not exposed by the Python binding. All floating
    hosts therefore share the largest open panel's radius, including panels
    still in the main dockspace. This also covers the first undocked frame.
    Individual windows retain exact geometry; the main dockspace stays square.
    """
    from mojive.design.groups import component_groups, group_scope
    from mojive.design.native.presentation import PANEL_FAMILIES

    ui = workspace.ui

    def member_radius(name):
        group = group_scope(name) if name in component_groups() else nullcontext()
        family = ui.design.family_scope(PANEL_FAMILIES.get(name)) if ui.design else nullcontext()
        with group, family:
            return floating_panel_geometry(ui, name).radius

    radius = None
    for name, opened in workspace.open.items():
        if not opened:
            continue
        candidate = member_radius(name)
        radius = candidate if radius is None else max(radius, candidate)
    for panel in workspace.app.panels.panels:
        if panel.name in workspace.open or not (panel.enabled and panel.open):
            continue
        candidate = member_radius(panel.name)
        radius = candidate if radius is None else max(radius, candidate)
    if radius is None:
        radius = floating_panel_geometry(ui).radius
    imgui.push_style_var(imgui.StyleVar_.window_rounding, radius)
    try:
        yield radius
    finally:
        imgui.pop_style_var()


@contextmanager
def native_chrome(ui, part):
    style, scale = ui.component_style(part), ui.s
    current = imgui.get_style()
    padding = tuple(
        style.get("padding" + axis, value / scale) * scale
        for axis, value in zip(("X", "Y"), current.frame_padding, strict=True)
    )
    tint = ui.style_color(
        style,
        "textColor",
        tuple(round(channel * 255) for channel in imgui.get_style_color_vec4(imgui.Col_.text)),
    )
    imgui.push_font(
        ui.resolve_font(style), style.get("fontSize", imgui.get_font_size() / scale) * scale
    )
    imgui.push_style_color(imgui.Col_.text, tuple(channel / 255 for channel in tint))
    imgui.push_style_var(imgui.StyleVar_.frame_padding, padding)
    if part == "dock_tabs":
        imgui.push_style_var(
            imgui.StyleVar_.item_inner_spacing,
            (
                style.get("gap", current.item_inner_spacing.x / scale) * scale,
                current.item_inner_spacing.y,
            ),
        )
    try:
        with part_scope(part, inherit=False):
            yield
    finally:
        imgui.pop_style_var(2 if part == "dock_tabs" else 1)
        imgui.pop_style_color()
        imgui.pop_font()


def draw_dockspace(workspace):
    ui = workspace.ui
    with native_chrome(ui, "dock_tabs"):
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0)
        imgui.push_style_var(imgui.StyleVar_.child_rounding, 0)
        try:
            workspace.window.begin_dockspace()
        finally:
            imgui.pop_style_var(2)
        if ui.design is None or not ui.design.runtime.observes(("dock_tabs",)):
            return
        visited = set()
        for window in imgui.get_current_context().windows:
            node = window.dock_node
            if (
                node is None
                or node.id_ in visited
                or node.is_no_tab_bar()
                or node.is_hidden_tab_bar()
            ):
                continue
            visited.add(node.id_)
            bar, host = node.tab_bar, node.host_window
            if (
                bar is None
                or host is None
                or not len(bar.tabs)
                or bar.curr_frame_visible != imgui.get_frame_count()
            ):
                continue
            rect = bar.bar_rect
            if rect.max.y <= rect.min.y:
                continue
            ui.design.record(
                "dock-tabs/" + str(node.id_),
                ("dock_tabs",),
                (rect.min.x, rect.min.y, rect.max.x - rect.min.x, rect.max.y - rect.min.y),
                "navigation",
                draw=host.draw_list,
                windowId=host.id_,
                drawListSource="window",
                nativeFontSize=imgui.get_font_size(),
                fields=[
                    (rect.min.x + tab.offset, rect.min.y, tab.width, rect.max.y - rect.min.y)
                    for tab in bar.tabs
                ],
            )
