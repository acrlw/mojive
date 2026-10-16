"""Native interaction plates and independent scrollable specimen panels."""

from imgui_bundle import imgui as g

from mojive.design.native.geometry import Rect
from mojive.ui.interaction_paint import ItemPaint, reference_paint

STATE_PARTS = (
    "button",
    "icon_button",
    "input_text",
    "input_number",
    "axis_input",
    "select",
    "select_option",
    "menu_item",
    "checkbox",
    "switch",
    "range",
    "input_shell",
)


def state_matrix(book, surface):
    ui, s = surface.ui, surface.s
    columns = ("常态", "悬停", "按下", "禁用")
    paints = (
        ItemPaint(),
        ItemPaint(hovered=True),
        ItemPaint(active=True),
        ItemPaint(disabled=True),
    )
    part = book.state_part
    if g.begin_table("reference-state-matrix", 4, g.TableFlags_.sizing_stretch_same):
        for index, (label, paint) in enumerate(zip(columns, paints, strict=True)):
            g.table_next_column()
            g.text_unformatted(label)
            x, y = g.get_cursor_screen_pos()
            width = max(1, g.get_content_region_avail().x - 12 * s)
            w, h = book.specimens.measure(ui, part, width)
            g.push_id("state-" + str(index))
            g.begin_disabled(paint.disabled)
            try:
                with reference_paint(paint):
                    book.specimens.paint(ui, part, Rect(x + (width - w) / 2, y + 16 * s, w, h))
            finally:
                g.end_disabled()
                g.pop_id()
            g.set_cursor_screen_pos((x, y))
            g.dummy((width, h + 40 * s))
        g.end_table()


def scroll_panels(book, surface):
    from mojive.ui.scrollbar_visibility import NativeScrollbarVisibility

    _ui, s = surface.ui, surface.s
    visibility = book.specimens.objects.setdefault("scrollbars", NativeScrollbarVisibility())
    visibility.update()
    width = (g.get_content_region_avail().x - 20 * s) / 2
    with visibility.scope():
        for index in range(2):
            if index:
                g.same_line()
            name = "Reference scroll " + str(index)
            with visibility.panel_scope(name, window_name=name):
                g.begin_child(name, (width, 240 * s), g.ChildFlags_.borders)
                visibility.bind_current_window()
                try:
                    for row in range(20):
                        g.text_unformatted(
                            ("场景节点" if index == 0 else "材质属性") + f" {row + 1:02d}"
                        )
                        g.dummy((1, 8 * s))
                finally:
                    g.end_child()
