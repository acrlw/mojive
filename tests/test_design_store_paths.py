"""User design preferences remain independent of generated captures and launchers."""

import os
import shutil
from copy import deepcopy

from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore, default_spec_path


def test_default_design_uses_the_editor_settings_directory(tmp_path, monkeypatch):
    settings = tmp_path / "user-config" / "settings.json"
    monkeypatch.delenv("MOJIVE_DESIGN_SPEC", raising=False)
    monkeypatch.setenv("MOJIVE_SETTINGS", str(settings))
    store = SpecificationStore()
    assert default_spec_path() == settings.with_name("design-spec.json")
    assert store.saved_path == settings.with_name("design-spec.json")
    assert store.live_path == settings.with_name("design-spec.live.json")
    assert not settings.parent.exists()


def test_explicit_design_paths_override_the_settings_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "shared" / "custom.json"))
    assert SpecificationStore().saved_path == tmp_path / "shared" / "custom.json"
    explicit = SpecificationStore(tmp_path / "other.json", tmp_path / "draft.json")
    assert explicit.saved_path == tmp_path / "other.json"
    assert explicit.live_path == tmp_path / "draft.json"


def test_generated_output_is_neither_a_fallback_nor_a_design_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOJIVE_DESIGN_SPEC", raising=False)
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "config" / "settings.json"))
    legacy = SpecificationStore(tmp_path / "output/design-workbench/current-spec.json")
    old = deepcopy(DEFAULT)
    old["parts"]["button"]["height"] = 44
    legacy.write(old, save=True)
    store = SpecificationStore()
    assert store.latest()["spec"] == DEFAULT
    user = deepcopy(DEFAULT)
    user["parts"]["button"]["height"] = 40
    store.write(user, save=True)
    shutil.rmtree(tmp_path / "output")
    assert SpecificationStore().latest()["spec"] == user


def test_readme_capture_isolates_settings_but_preserves_the_users_design(tmp_path, monkeypatch):
    from PIL import Image

    from mojive.tools import build_readme_media

    settings = tmp_path / "config/settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(settings))
    monkeypatch.delenv("MOJIVE_DESIGN_SPEC", raising=False)
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_RENDERER", "opengl")
    expected = default_spec_path()

    def capture_editor(output, width, height, **_options):
        assert default_spec_path() == expected
        assert os.environ["MOJIVE_SETTINGS"] != str(settings)
        for name in ("hero.png", "joint-authoring.png"):
            Image.new("RGB", (width, height)).save(output / name)
        return {}

    def capture_rendering(output, width, height):
        Image.new("RGB", (width, height)).save(output / "rendering.png")

    monkeypatch.setattr(build_readme_media, "_capture_editor", capture_editor)
    monkeypatch.setattr(build_readme_media, "_capture_rendering", capture_rendering)
    assert (
        build_readme_media.main(
            ["--output", str(tmp_path / "captures"), "--width", "32", "--height", "24"]
        )
        == 0
    )
