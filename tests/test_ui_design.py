"""CPU checks for the design study's output ownership and review lifecycle."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native import docking_study, fonts, reference_menus


def test_portable_fonts_work_without_system_fonts_or_cached_downloads(monkeypatch):
    is_file = fonts.Path.is_file

    def without_sf(path):
        return False if str(path).startswith("/System/Library/Fonts/") else is_file(path)

    def no_primary(**kwargs):
        assert kwargs["allow_download"] is False
        raise FileNotFoundError("No system or cached primary font")

    monkeypatch.setattr(fonts.Path, "is_file", without_sf)
    monkeypatch.setattr(fonts, "resolve_font_sources", no_primary)
    resolved = fonts.resolve_fonts()
    assert all(is_file(fonts.Path(source.path)) for source, _ratio in resolved)
    assert resolved[2][0].label == "Inconsolata"


def test_portable_fonts_can_be_selected_on_macos():
    resolved = fonts.resolve_fonts(portable=True)
    assert all(source.path != "/System/Library/Fonts/SFNS.ttf" for source, _ratio in resolved)
    assert resolved[0][0].label == "Roboto Regular"


@pytest.fixture
def document_menus(tmp_path):
    from mojive import Scene
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.session import Session
    from mojive.session.model_edits import ModelEditDraft
    from mojive.ui.app import ViewerApp
    from mojive.ui.theme import THEME

    scene = Scene()
    scene.box(name="body", position=(0, 0, 0))
    session = Session(StaticSceneAdapter(scene))
    node = next(n for n in session.nodes if n.name == "body")
    app = ViewerApp.__new__(ViewerApp)
    app.session = session
    app.model_edits = ModelEditDraft(session)
    app.live_model_updates = False
    app.localizer = SimpleNamespace(text=str)
    app.window = SimpleNamespace(style_scale=1)
    app.theme = THEME
    app._pending_document_action = None
    app._model_load_queue = []
    output = tmp_path / "chosen-output"
    output.mkdir()
    study = SimpleNamespace(s=1, ui=None, output_directory=output, preview=app)
    menus = reference_menus.Menus(study, ())
    try:
        yield menus, app, node, output
    finally:
        session.release()


def _choose_unsaved_action(monkeypatch, decision):
    from mojive.ui.app import menus as viewer_menus

    monkeypatch.setattr(viewer_menus, "_prepare_modal", lambda *args: None)
    monkeypatch.setattr(
        viewer_menus,
        "_equal_modal_buttons",
        lambda *args, **kwargs: tuple(
            decision == action for action in ("cancel", "discard", "save")
        ),
    )
    monkeypatch.setattr(viewer_menus.imgui, "begin_popup_modal", lambda *args: (True, None))
    monkeypatch.setattr(viewer_menus.imgui, "is_key_pressed", lambda *args: False)
    for name in (
        "open_popup",
        "text",
        "text_wrapped",
        "spacing",
        "close_current_popup",
        "end_popup",
    ):
        monkeypatch.setattr(viewer_menus.imgui, name, lambda *args: None)


def test_document_actions_use_the_study_output_directory(document_menus, tmp_path, monkeypatch):
    from mojive import commands as cmd
    from mojive.scene.queries import node_world_pose

    menus, app, node, output = document_menus
    session = app.session
    session.submit(cmd.SetPose(node.node_id, np.array([1, 2, 3]), np.eye(3)))
    menus.dispatch("file:save")
    menus.dispatch("file:export")
    assert all(
        (output / name).is_file()
        for name in (reference_menus.SAVED_SCENE, reference_menus.EXPORTED_SCENE)
    )
    assert session.last_message == f"Saved {reference_menus.EXPORTED_SCENE}"
    assert app._model_drop_notice == session.last_message
    session.submit(cmd.SetPose(node.node_id, np.array([9, 9, 9]), np.eye(3)))
    menus.dispatch("file:load")
    assert app._pending_document_action == ("open_scene", output / reference_menus.SAVED_SCENE)
    assert not app._model_load_queue
    _choose_unsaved_action(monkeypatch, "discard")
    app._draw_unsaved_changes()
    job = app._model_load_queue.pop()
    assert job.action == "open" and isinstance(job.command, cmd.OpenScene)
    assert app._load_model(job.command).ok
    node = next(n for n in session.nodes if n.name == "body")
    assert node_world_pose(session, node)[0].tolist() == [1, 2, 3]
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("action", ("save", "export"))
@pytest.mark.parametrize("in_frame", (False, True))
def test_document_save_refuses_pending_dimensions(document_menus, action, in_frame):
    from contextlib import nullcontext

    from mojive import commands as cmd
    from mojive.session.model_edits import model_edit_scope

    menus, app, node, output = document_menus
    before = app.session._source.geom_size.copy()
    assert app.model_edits.stage(cmd.SetScale(node.node_id, np.array([2.0, 1.0, 1.0]))).ok
    scope = model_edit_scope(app.session, app._intercept_model_edit) if in_frame else nullcontext()
    with scope:
        menus.dispatch("file:" + action)
    assert not list(output.iterdir())
    assert app.session.last_message == "Apply or discard pending model edits first"
    assert app.model_edits.active and app.session.dirty
    np.testing.assert_allclose(app.session._source.geom_size, before)


@pytest.mark.parametrize("edit", ("pose", "dimensions"))
def test_open_saved_scene_cancel_keeps_unsaved_work(document_menus, monkeypatch, edit):
    from mojive import commands as cmd
    from mojive.scene.queries import node_world_pose

    menus, app, node, output = document_menus
    menus.dispatch("file:save")
    if edit == "pose":
        assert app.session.submit(cmd.SetPose(node.node_id, np.array([9, 9, 9]), np.eye(3))).ok
    else:
        assert app.model_edits.stage(cmd.SetScale(node.node_id, np.array([2.0, 1.0, 1.0]))).ok
    before = app.session.source.geom_size.copy()
    document_id = app.session.document_id
    menus.dispatch("file:load")
    assert app._pending_document_action == ("open_scene", output / reference_menus.SAVED_SCENE)
    assert not app._model_load_queue
    _choose_unsaved_action(monkeypatch, "cancel")
    app._draw_unsaved_changes()
    assert app._pending_document_action is None and not app._model_load_queue
    assert app.session.document_id == document_id and app.session.dirty
    np.testing.assert_allclose(app.session.source.geom_size, before)
    if edit == "pose":
        np.testing.assert_allclose(node_world_pose(app.session, node)[0], [9, 9, 9])
    else:
        assert app.model_edits.active


def test_clean_open_saved_scene_queues_document_loading(document_menus):
    from mojive import commands as cmd

    menus, app, _node, output = document_menus
    menus.dispatch("file:save")
    document_id = app.session.document_id
    menus.dispatch("file:load")
    assert app._pending_document_action is None
    assert app.session.document_id == document_id
    job = app._model_load_queue.pop()
    assert job.action == "open" and isinstance(job.command, cmd.OpenScene)
    assert job.path == (output / reference_menus.SAVED_SCENE).resolve()


def test_document_save_reports_writer_failure(document_menus, monkeypatch):
    menus, app, _node, output = document_menus

    def fail_save(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(app.session.adapter, "save_scene", fail_save)
    menus.dispatch("file:save")
    assert not list(output.iterdir())
    assert app.session.last_message == "Disk full"
    assert app._model_load_error == "Disk full" and app._show_model_load_error


def _layout_snapshot():
    return {
        "Scene": {"dock": 1, "rect": [0, 0, 100, 600]},
        "Inspector": {"dock": 2, "rect": [700, 0, 100, 600]},
        "Viewport": {"dock": 3, "rect": [100, 0, 600, 400]},
        "Timeline": {"dock": 4, "rect": [100, 400, 600, 200]},
        "viewport_content": [100, 0, 600, 400],
        "overlays": [[110, 10, 30, 30], [650, 10, 30, 30]],
    }


@pytest.mark.parametrize("review_density", (1, 2))
@pytest.mark.parametrize("desktop_density", (1, 2))
def test_fresh_context_restore_keeps_custom_output_and_review_density(
    tmp_path, monkeypatch, review_density, desktop_density
):
    windows, studies, measurements = [], [], []
    io = SimpleNamespace(display_size=None, display_framebuffer_scale=None)
    monkeypatch.setattr(docking_study.imgui, "get_io", lambda: io)
    monkeypatch.setattr(docking_study.imgui, "get_version", lambda: "review-test")
    monkeypatch.setattr(docking_study.imgui, "save_ini_settings_to_memory", lambda: "saved layout")
    monkeypatch.setattr(
        docking_study.imgui,
        "get_main_viewport",
        lambda: SimpleNamespace(
            work_pos=SimpleNamespace(x=0, y=0),
            work_size=SimpleNamespace(x=io.display_size[0], y=io.display_size[1]),
        ),
    )

    class Window:
        def __init__(self, config, backend):
            self.size_pixels = (config.width * desktop_density, config.height * desktop_density)

            def process_inputs():
                io.display_size = (config.width, config.height)
                io.display_framebuffer_scale = (desktop_density, desktop_density)

            self._input = SimpleNamespace(process_inputs=process_inputs)
            self.closed = False
            windows.append(self)

        def close(self):
            self.closed = True

    class Study:
        def __init__(
            self, window, backend, restore=None, *, output_directory, portable_fonts=False
        ):
            self.output_directory = output_directory
            self.restore = restore
            self.mode = "dock"
            self.open = {}
            self.closed = False
            studies.append(self)

        def activate(self, _name):
            pass

        def toggle_timeline(self):
            self.open["Timeline"] = True

        def command(self, action):
            self.mode = action

        def close(self):
            self.closed = True

    def render(window, study, path=None):
        window._input.process_inputs()
        measurements.append((io.display_size, io.display_framebuffer_scale))
        if path is not None:
            assert path.parent == study.output_directory
        snapshot = _layout_snapshot()
        if study.mode == "float":
            snapshot["Inspector"]["dock"] = 0
        elif study.mode == "group":
            snapshot["Inspector"]["dock"] = snapshot["Scene"]["dock"]
        return snapshot

    monkeypatch.setattr(docking_study, "create_window", Window)
    monkeypatch.setattr(docking_study, "Study", Study)
    monkeypatch.setattr(docking_study, "frames", render)
    output = tmp_path / "custom-output"
    docking_study.main(
        [
            "--output",
            str(output),
            "--review-density",
            str(review_density),
            "--width",
            "800",
            "--height",
            "600",
        ]
    )
    assert len(windows) == len(studies) == 2
    assert all(window.closed for window in windows)
    assert all(study.closed and study.output_directory == output for study in studies)
    assert studies[1].restore == "saved layout"
    density = review_density * desktop_density
    assert all(value == ((800, 600), (density, density)) for value in measurements)
    report = json.loads((output / "report.json").read_text())
    assert "restored" in report["states"]
