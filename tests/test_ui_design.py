"""CPU checks for the design study's output ownership and review lifecycle."""

import json
from copy import deepcopy
from types import SimpleNamespace

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


def test_document_actions_use_the_study_output_directory(tmp_path, monkeypatch):
    document = {
        "entities": [{"id": "body", "position": [0, 0, 0]}],
        "pose": {"hinge": 0},
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.json").write_text(json.dumps(document))
    monkeypatch.setattr(reference_menus, "ROOT", source)

    class Preview:
        def __init__(self, *_args):
            self.document = deepcopy(document)
            self.entities = self.document["entities"]

        def close(self):
            pass

    monkeypatch.setattr(docking_study, "Preview", Preview)
    monkeypatch.setattr(docking_study, "UI", lambda _window, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(docking_study.imgui, "get_style", SimpleNamespace)
    restored = []
    monkeypatch.setattr(docking_study.imgui, "load_ini_settings_from_memory", restored.append)
    output = tmp_path / "chosen-output"
    study = docking_study.Study(
        SimpleNamespace(style_scale=1), restore="saved layout", output_directory=output
    )
    try:
        assert restored == ["saved layout"]
        study.preview.entities[0]["position"] = [1, 2, 3]
        study.menus.dispatch("file:save")
        study.menus.dispatch("file:export")
        for name in ("saved-scene.json", "exported-scene.json"):
            assert json.loads((output / name).read_text())["entities"][0]["position"] == [1, 2, 3]
        study.preview.entities[0]["position"] = [9, 9, 9]
        study.menus.dispatch("file:load")
        assert study.preview.entities[0]["position"] == [1, 2, 3]
        assert study.preview.entities is study.preview.document["entities"]
        assert list(source.iterdir()) == [source / "document.json"]
        assert json.loads((source / "document.json").read_text()) == document
    finally:
        study.close()


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
