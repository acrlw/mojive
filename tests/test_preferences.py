"""Preference persistence, shared ownership and legacy localization entry points."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

import pytest

from mojive.ui.localization import Language, Localizer
from mojive.ui.preferences import Preferences, settings_path


@pytest.mark.parametrize("content", [None, "{broken", "[]", "null", "false", '"text"'])
def test_missing_or_invalid_settings_start_with_empty_preferences(tmp_path, content):
    path = tmp_path / "settings.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    preferences = Preferences.load(path)
    assert preferences.values == {}
    assert preferences.path == path
    assert preferences.get("missing", 42) == 42


def test_environment_path_round_trip_preserves_unrelated_preferences(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(path))
    preferences = Preferences.load()
    assert settings_path() == path
    preferences.update({"extension": {"label": "保留"}, "status_metric": "steps"})
    preferences.update({"status_metric": "time"})
    assert Preferences.load().values == {
        "extension": {"label": "保留"},
        "status_metric": "time",
    }
    assert "保留" in path.read_text(encoding="utf-8")
    preferences.update({"status_metric": "steps"}, persist=False)
    assert preferences.get("status_metric") == "steps"
    assert Preferences.load().get("status_metric") == "time"


def test_localizer_and_editor_share_one_preference_store(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJIVE_LANGUAGE", raising=False)
    preferences = Preferences(tmp_path / "settings.json", {"status_metric": "steps"})
    localizer = Localizer.from_preferences(preferences)
    values = localizer.preferences
    assert values is preferences.values
    preferences.update({"status_metric": "time"})
    assert localizer.preference("status_metric") == "time"
    localizer.set_preferences({"take_pause_at_end": False})
    assert preferences.get("take_pause_at_end") is False
    assert values is localizer.preferences
    localizer.set_language("zh_CN")
    assert preferences.get("language") == "zh_CN"
    assert Localizer.from_preferences(Preferences.load(preferences.path)).text("Settings") == "设置"


def test_legacy_constructor_and_mutable_aliases_keep_their_behavior(tmp_path):
    values = {"status_metric": "steps"}
    localizer = Localizer(Language.SIMPLIFIED_CHINESE, tmp_path / "first.json", values)
    assert localizer.preferences is values
    localizer.set_preferences({"view_selection_padding": 2}, persist=False)
    assert values["view_selection_padding"] == 2
    assert not localizer.path.exists()
    localizer.path = tmp_path / "second.json"
    replacement = {"status_metric": "time"}
    localizer.preferences = replacement
    localizer.set_language("en")
    assert localizer.preferences is replacement
    assert json.loads(localizer.path.read_text()) == {"status_metric": "time", "language": "en"}


def test_localizer_preserves_dataclass_value_operations(tmp_path):
    values = {"status_metric": "steps"}
    path = tmp_path / "settings.json"
    localizer = Localizer(Language.ENGLISH, path, values)
    assert is_dataclass(localizer)
    assert [field.name for field in fields(localizer)] == ["language", "path", "preferences"]
    assert localizer == Localizer(Language.ENGLISH, path, dict(values))
    assert repr(localizer) == (
        f"Localizer(language={Language.ENGLISH!r}, path={path!r}, preferences={values!r})"
    )
    copied = replace(localizer, language=Language.SIMPLIFIED_CHINESE)
    assert copied.preferences is values
    assert copied.path == path
    assert copied != localizer
    assert copied.text("Settings") == "设置"
    snapshot = asdict(localizer)
    assert snapshot == {"language": Language.ENGLISH, "path": path, "preferences": values}
    assert snapshot["preferences"] is not values


@pytest.mark.parametrize("requested, expected", [(None, "zh_CN"), ("en-US", "en"), ("bad", "en")])
def test_language_environment_overrides_saved_language_without_rewriting_it(
    tmp_path, monkeypatch, requested, expected
):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(path))
    if requested is None:
        monkeypatch.delenv("MOJIVE_LANGUAGE", raising=False)
    else:
        monkeypatch.setenv("MOJIVE_LANGUAGE", requested)
    Preferences(path).update({"language": "zh_CN"})
    localizer = Localizer.load()
    assert localizer.language.value == expected
    assert localizer.preference("language") == "zh_CN"
    localizer.set_language("en", persist=False)
    assert Preferences.load(path).get("language") == "zh_CN"


def test_failed_save_retains_previous_file_and_shared_values(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    preferences = Preferences(path)
    preferences.update({"status_metric": "steps"})
    values = preferences.values

    def reject_replace(self, target):
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="replacement failed"):
        preferences.update({"status_metric": "time"})
    assert preferences.values is values
    assert values == {"status_metric": "steps"}
    assert Preferences.load(path).values == values
    assert list(tmp_path.iterdir()) == [path]


def test_failed_language_save_keeps_language_preferences_and_file_consistent(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    preferences = Preferences(path)
    preferences.update({"language": "en"})
    monkeypatch.delenv("MOJIVE_LANGUAGE", raising=False)
    localizer = Localizer.from_preferences(preferences)

    def reject_replace(self, target):
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="replacement failed"):
        localizer.set_language("zh_CN")
    assert localizer.language is Language.ENGLISH
    assert localizer.text("Settings") == "Settings"
    assert preferences.get("language") == "en"
    assert Preferences.load(path).get("language") == "en"
    localizer.set_language("zh_CN", persist=False)
    assert localizer.text("Settings") == "设置"
    assert preferences.get("language") == "en"
    assert Preferences.load(path).get("language") == "en"


def test_unserializable_value_does_not_modify_saved_or_in_memory_state(tmp_path):
    preferences = Preferences(tmp_path / "settings.json", {"status_metric": "steps"})
    with pytest.raises(TypeError):
        preferences.update({"unsupported": object()})
    assert preferences.values == {"status_metric": "steps"}
    assert not preferences.path.exists()
