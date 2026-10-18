"""The production timeline preserves semantic groups across width and typography changes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2.25))
def toolbar(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1800 * request.param, 1000 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    session = Session(StaticSceneAdapter(Scene()))
    ctx = PanelContext(session, NullBackend(), style_scale=ui.s)
    workspace = SimpleNamespace(ui=ui, s=ui.s)
    workspace.menus = Menus(workspace, ())
    panel, editor = NativeTimelineToolbar(workspace), TimelineEditor()
    state = SimpleNamespace(ui=ui, runtime=runtime, editor=editor, width=1000)

    def paint():
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((state.width * ui.s + 16, 900 * ui.s))
        imgui.begin("Timeline groups", flags=imgui.WindowFlags_.no_title_bar)
        ui.design.begin_frame()
        runtime.observed.clear()
        ui.hits.clear()
        with group_scope("Timeline"):
            panel.draw_compact_toolbar(editor, ctx, ())
        state.records = {
            row["key"]: row for row in (*runtime.observed, *ui.design.primitives.observed)
        }
        state.bottom = imgui.get_cursor_screen_pos().y
        imgui.end()
        imgui.render()

    state.paint = paint
    state.localize = lambda language: setattr(ui, "translate", Localizer(language).text)
    try:
        for _ in range(3):
            paint()
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


def _inside(outer, inner):
    x, y, w, h = outer
    ix, iy, iw, ih = inner
    return x - 0.1 <= ix and y - 0.1 <= iy and ix + iw <= x + w + 0.1 and iy + ih <= y + h + 0.1


@pytest.mark.parametrize("width", (260, 340, 620, 1100, 1600))
@pytest.mark.parametrize("language", (Language.ENGLISH, Language.SIMPLIFIED_CHINESE))
def test_real_toolbar_keeps_range_pair_and_whole_groups_inside_their_container(
    toolbar, width, language
):
    state, ui = toolbar, toolbar.ui
    state.width = width
    state.localize(language)
    state.paint()
    groups = tuple(
        ui.hits["timeline-group-" + name] for name in ("transport", "capture", "range", "view")
    )
    for index, group in enumerate(groups):
        assert _inside(state.records["timeline-toolbar"]["content"], group)
        for other in groups[index + 1 :]:
            assert group[0] + group[2] <= other[0] + 0.1 or group[1] + group[3] <= other[1] + 0.1
    for key in ("first", "previous", "play-pause", "next", "last", "time"):
        assert _inside(groups[0], ui.hits["timeline-" + key])
    for key in ("record", "take", "snapshot", "video"):
        assert _inside(groups[1], ui.hits["timeline-" + key])
    start, end = (state.records["range-bounds/##timeline-range-" + key] for key in ("start", "end"))
    for record in (start, end):
        assert _inside(groups[2], record["outer"])
    assert start["fieldRect"][1] == pytest.approx(end["fieldRect"][1])
    assert start["fieldRect"][0] + start["fieldRect"][2] < end["fieldRect"][0]
    assert state.bottom >= max(y + h for _, y, _, h in groups)


def test_dynamic_numbers_do_not_resize_toolbar_groups(toolbar):
    state = toolbar
    state.width = 340
    state.paint()
    before = dict(state.ui.hits)
    state.editor.playhead = 12345.6789
    state.editor.view_start, state.editor.view_end = -1234.5, 12345.6
    state.paint()
    for key in ("transport", "capture", "range", "view"):
        assert state.ui.hits["timeline-group-" + key] == before["timeline-group-" + key]
    for key in ("timeline-time", "##timeline-range-start", "##timeline-range-end"):
        assert state.ui.hits[key] == before[key]


def test_toolbar_padding_and_numeric_typography_keep_the_range_pair_together(toolbar):
    state = toolbar
    state.width = 420
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["timeline_toolbar"] = {
        "paddingLeft": 17,
        "paddingRight": 23,
        "paddingTop": 9,
        "paddingBottom": 11,
        "rowGap": 13,
        "columnGap": 11,
    }
    spec["parts"]["range_bounds_input_number"] = {"fontSize": 21}
    state.runtime.update_spec(spec)
    state.paint()
    container = state.records["timeline-toolbar"]
    for key in ("transport", "capture", "range", "view"):
        assert _inside(container["content"], state.ui.hits["timeline-group-" + key])
    assert container["padding"] == pytest.approx(tuple(v * state.ui.s for v in (17, 9, 23, 11)))
    start, end = (state.ui.hits["##timeline-range-" + key] for key in ("start", "end"))
    assert _inside(state.ui.hits["timeline-group-range"], start)
    assert _inside(state.ui.hits["timeline-group-range"], end)
    assert start[1] == pytest.approx(end[1])
    assert start[3] == pytest.approx(end[3])
