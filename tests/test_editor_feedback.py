"""Editor feedback regressions without opening desktop applications."""

from pathlib import Path

import pytest
from imgui_bundle import imgui

from mojive.ui import files, theme
from mojive.ui.panels.inspector import vector_layout


def test_native_checkbox_selected_background_uses_the_theme():
    context = imgui.create_context()
    try:
        theme.apply(imgui)
        color = imgui.get_style_color_vec4(imgui.Col_.checkbox_selected_bg)
        assert tuple(color) == pytest.approx(theme.THEME.bg_frame_active)
    finally:
        imgui.destroy_context(context)


def test_vector_reflow_keeps_each_layout_until_there_is_room_to_expand():
    label, axes, scale = 70, 210, 1
    assert vector_layout(285, label, axes, scale) == 0
    assert vector_layout(275, label, axes, scale, 0) == 1
    assert vector_layout(285, label, axes, scale, 1) == 1
    assert vector_layout(310, label, axes, scale, 1) == 0
    assert vector_layout(205, label, axes, scale, 1) == 2
    assert vector_layout(215, label, axes, scale, 2) == 2
    assert vector_layout(240, label, axes, scale, 2) == 1
    for width in (205, 215, 240, 275, 285, 310):
        for previous in range(3):
            assert vector_layout(
                width * 2.5, label * 2.5, axes * 2.5, 2.5, previous
            ) == vector_layout(width, label, axes, 1, previous)


def test_output_paths_preserve_spaces_unicode_and_trailing_message_text(tmp_path):
    path = tmp_path / "录制片段 with spaces.mp4"
    path.touch()
    assert files.message_path(f"视频已保存到 {path}") == path
    assert files.message_path(f'Saved "{path}" in 1.2 seconds') == path
    assert files.message_path("Saved recording", str(path)) == path
    assert files.message_path("Rotate transform") is None
    assert files.message_path(str(path.with_name("missing.mp4"))) is None
    folder = tmp_path / "exports (1)"
    folder.mkdir()
    assert files.message_path(f"Saved files to {folder}") == folder


def test_linux_file_reveal_selects_the_file_and_falls_back_to_its_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(files.sys, "platform", "linux")
    monkeypatch.setattr(files.subprocess, "run", lambda command, **kwargs: calls.append(command))
    path = tmp_path / "file with spaces.mp4"
    files._reveal(path)
    assert calls[0][-3:] == ["org.freedesktop.FileManager1.ShowItems", f"['{path.as_uri()}']", ""]

    def unsupported(command, **kwargs):
        calls.append(command)
        if command[0] == "gdbus":
            raise FileNotFoundError

    monkeypatch.setattr(files.subprocess, "run", unsupported)
    files._reveal(path)
    assert calls[-1] == ["xdg-open", str(path.parent)]


@pytest.mark.parametrize(
    "platform,command", [("darwin", ["open", "-R"]), ("win32", ["explorer", "/select,"])]
)
def test_file_reveal_uses_native_selection_commands(monkeypatch, platform, command):
    calls = []
    monkeypatch.setattr(files.sys, "platform", platform)
    monkeypatch.setattr(files.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    path = Path("file with spaces.mp4")
    files._reveal(path)
    assert calls == [[*command, str(path)]]
