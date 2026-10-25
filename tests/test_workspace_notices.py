"""Operation notices and interactive file receipts share complete, centered paint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.theme import THEME
from mojive.ui.workspace.notices import draw_center_notice, draw_file_receipt, draw_operation_notice
from tests.test_component_appearance import appearance as appearance
from tests.test_workspace_tooltips import native_state

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _app(ui, *, width=1100):
    return SimpleNamespace(
        workspace=SimpleNamespace(ui=ui, s=ui.s),
        localizer=SimpleNamespace(text=str),
        theme=THEME,
        _viewport_rect=(20 * ui.s, 20 * ui.s, width * ui.s, 500 * ui.s),
        _status_path_bounds=None,
        _status_notice_bounds=None,
    )


def test_center_notice_translates_complete_template_before_layout_and_paint(
    appearance, monkeypatch
):
    ui, _runtime, _update, frame = appearance
    localizer = Localizer()
    ui.translate = localizer.text
    app = _app(ui)
    source = "Drop a .mojive.json scene here\nOpen a scene or add an object to begin."
    painted, measured = [], []
    paint, measure = ImguiDraw2D.text, ui.measure

    def text(draw, position, color, value, **options):
        painted.append(value)
        return paint(draw, position, color, value, **options)

    def width(value, *args, **options):
        measured.append(value)
        return measure(value, *args, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    monkeypatch.setattr(ui, "measure", width)
    for language in (Language.ENGLISH, Language.SIMPLIFIED_CHINESE, Language.ENGLISH):
        localizer.set_language(language, persist=False)
        painted.clear()
        measured.clear()
        frame(lambda: draw_center_notice(app, ImguiDraw2D(), source))
        lines = localizer.text(source).splitlines()
        assert all(line in painted and line in measured for line in lines)
        if language is Language.SIMPLIFIED_CHINESE:
            assert all(line not in painted and line not in measured for line in source.splitlines())


@pytest.mark.parametrize("stacked", (False, True))
@pytest.mark.parametrize("radius", (None, 0, 32))
def test_receipt_surface_keeps_native_corners_and_inline_text_uses_action_row_center(
    appearance, monkeypatch, stacked, radius
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    if radius is not None:
        update({"file_receipt": {"radius": radius}})
    app = _app(ui, width=360 if stacked else 1100)
    path = Path("/tmp/shot.png")
    caption = "Saved screenshot to " + str(path)
    message = SimpleNamespace(text=caption, copy_text=str(path), level="info")
    surfaces, text_rows = [], []
    rectangle, text = ui.rect, ui.text

    def paint_surface(x, y, width, height, tint, r=None, **options):
        window = imgui.internal.get_current_window()
        if window.name == "##viewport-file-status" and not surfaces:
            draw = options.get("draw") or ui.draw
            surfaces.append(
                {
                    "frame": (x, y, width, height),
                    "clip": (*draw.get_clip_rect_min(), *draw.get_clip_rect_max()),
                    "host": (*window.pos, *window.size),
                    "border": window.window_border_size,
                    "rounding": window.window_rounding,
                }
            )
        return rectangle(x, y, width, height, tint, r, **options)

    def paint_text(x, y, value, tint, size=14, **options):
        text_rows.append((value, y + size * ui.s / 2))
        return text(x, y, value, tint, size, **options)

    monkeypatch.setattr(ui, "rect", paint_surface)
    monkeypatch.setattr(ui, "text", paint_text)

    def draw():
        surfaces.clear()
        text_rows.clear()
        runtime.observed.clear()
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 3.25 * ui.s)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 44 * ui.s)
        before = native_state()
        try:
            draw_file_receipt(app, ImguiDraw2D(), message, path)
            assert native_state() == before
        finally:
            imgui.pop_style_var(2)

    for _ in range(3):
        frame(draw)
    plate = next(record for record in runtime.observed if record["key"] == "file-receipt")
    surface = surfaces[0]
    assert surface["border"] == surface["rounding"] == 0
    assert surface["frame"] == pytest.approx(plate["hit"])
    vx, vy, vw, vh = app._viewport_rect
    assert surface["clip"] == pytest.approx((vx, vy, vx + vw, vy + vh))
    assert plate["surfaceClip"] == surface["clip"]
    x, y, width, height = plate["hit"]
    # The real command's viewport scissor retains both outward AA fringes,
    # even when the transparent native host truncates its fractional dimensions.
    assert surface["clip"][0] < x - 1 < x + width + 1 < surface["clip"][2]
    assert surface["clip"][1] < y - 1 < y + height + 1 < surface["clip"][3]
    window = imgui.internal.find_window_by_name("##viewport-file-status")
    commands = [
        command
        for command in window.draw_list.cmd_buffer
        if command.elem_count and tuple(command.clip_rect) == surface["clip"]
    ]
    assert commands
    assert len(set(plate["padding"])) == 1 and plate["padding"][0] > 0
    assert plate["childRadius"] == pytest.approx(max(0, plate["radius"] - plate["padding"][0]))
    assert plate["inline"] is not stacked
    text_rect = plate["textRect"]
    actions = plate["actionRects"]
    assert len(actions) == 2
    assert max(rect[0] + rect[2] for rect in actions) == pytest.approx(
        x + width - plate["padding"][2]
    )
    assert all(key in ui.hits for key in ("receipt-open-folder", "receipt-copy-path"))
    if stacked:
        assert min(rect[1] for rect in actions) >= text_rect[1] + text_rect[3]
    else:
        center = text_rect[1] + text_rect[3] / 2
        assert all(rect[1] + rect[3] / 2 == pytest.approx(center) for rect in actions)
        # Check the actual text calls as well as allocation metadata.
        for value, text_center in text_rows:
            if value in ("Saved screenshot to shot.png", "Open folder", "Copy path"):
                assert text_center == pytest.approx(center, abs=1)


@pytest.mark.parametrize("level", ("info", "warning", "error"))
def test_message_and_file_receipt_use_same_severity_surface_without_removing_actions(
    appearance, monkeypatch, level
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    app, path = _app(ui), Path("/tmp/shot.png")
    message = SimpleNamespace(text="Saved screenshot", copy_text=str(path), level=level)
    paints = []
    rectangle = ui.rect

    def surface(x, y, width, height, tint, r=None, **options):
        paints.append((tint, r))
        return rectangle(x, y, width, height, tint, r, **options)

    monkeypatch.setattr(ui, "rect", surface)

    def draw():
        paints.clear()
        draw_operation_notice(app, ImguiDraw2D(), message)
        operation = paints[0][0]
        paints.clear()
        draw_file_receipt(app, ImguiDraw2D(), message, path)
        assert paints[0][0] == operation

    for _ in range(3):
        frame(draw)
    assert "receipt-open-folder" in ui.hits and "receipt-copy-path" in ui.hits


def test_receipt_actions_remain_real_native_click_targets(appearance, monkeypatch):
    from mojive.ui import files

    ui, _runtime, _update, frame = appearance
    app, path = _app(ui), Path("/tmp/shot.png")
    message = SimpleNamespace(text="Saved screenshot", copy_text=str(path), level="info")
    revealed, copied = [], []
    monkeypatch.setattr(files, "reveal_path", revealed.append)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)

    def draw():
        draw_file_receipt(app, ImguiDraw2D(), message, path)

    def event(name, *args):
        getattr(imgui.get_io(), name)(*args)
        frame(draw)

    for _ in range(3):
        frame(draw)
    for key in ("receipt-open-folder", "receipt-copy-path"):
        x, y, width, height = ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
    assert revealed == [path]
    assert copied == [str(path)]


def test_undo_notice_uses_native_click_and_rejects_stale_edits(appearance):
    from mojive.ui.workspace.notices import notice_can_undo

    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    app = _app(ui)
    submitted = []
    app.session = SimpleNamespace(
        document_id="scene",
        document_revision=3,
        can_undo=True,
        editing=False,
        submit=submitted.append,
    )
    app.model_edits = SimpleNamespace(active=False)
    app.gizmo = SimpleNamespace(using=False)
    app.refresh_document = lambda: None
    message = SimpleNamespace(text="Translate transform", level="info", undo=("scene", 3))

    def draw():
        ui.design.begin_frame()
        draw_operation_notice(app, ImguiDraw2D(), message)

    for _ in range(3):
        frame(draw)
    notice = next(r for r in runtime.observed if r["key"] == "operation-notice")
    action = next(r for r in ui.design.primitives.observed if r["key"] == "notice-undo")
    assert action["icon"] == "undo" and not action["clippedLabel"]
    x, y, width, height = ui.hits["notice-undo"]
    nx, ny, nw, nh = notice["hit"]
    inset = y - ny
    assert (nx + nw - x - width, ny + nh - y - height) == pytest.approx((inset, inset))
    assert action["fieldValues"]["radius"] * ui.s == pytest.approx(max(0, notice["radius"] - inset))
    text_x, _text_y, text_width, _text_height = notice["fields"][1]
    assert x - text_x - text_width == pytest.approx(16 * ui.s)
    assert nh == pytest.approx(36 * ui.s)
    assert notice["radius"] == pytest.approx(8 * ui.s)
    for event, args in [
        ("add_mouse_pos_event", (x + width / 2, y + height / 2)),
        ("add_mouse_button_event", (0, True)),
        ("add_mouse_button_event", (0, False)),
    ]:
        getattr(imgui.get_io(), event)(*args)
        frame(draw)
    assert len(submitted) == 1 and type(submitted[0]).__name__ == "Undo"
    message.undo = None
    message.text = "Undo Translate transform"
    for _ in range(3):
        frame(draw)
    result = next(r for r in runtime.observed if r["key"] == "operation-notice")
    assert (*result["hit"][:2], result["hit"][3], result["radius"]) == pytest.approx(
        (nx, ny, nh, notice["radius"])
    )
    assert not any(r["key"] == "notice-undo" for r in ui.design.primitives.observed)
    message.undo = ("scene", 3)
    app.session.document_revision = 4
    assert not notice_can_undo(app, message)
    app.session.document_revision = 3
    app.gizmo.using = True
    assert not notice_can_undo(app, message)
    app.gizmo.using = False
    app.model_edits.active = True
    assert not notice_can_undo(app, message)
