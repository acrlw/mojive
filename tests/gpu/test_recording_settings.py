"""Recording controls select, persist, and retain independent encoding values."""

import pytest
from imgui_bundle import imgui

from mojive import RecordingConfig, ViewerConfig, build
from mojive.scene.assets import resolve
from mojive.tools.keyframe_timeline import show_settings
from mojive.tools.ui_runtime import _click, _item_center, _item_rect

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("scale", (1, 2.5))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_encoding_controls_switch_modes_and_persist_without_losing_values(
    tmp_path, monkeypatch, scale, language
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with build(
        resolve("joint_types"),
        paused=True,
        vsync=False,
        width=min(3200, round(1280 * scale)),
        height=min(1800, round(800 * scale)),
        show_window=False,
        config=ViewerConfig(recording=RecordingConfig()),
    ) as viewer:
        viewer.app.set_language(language)
        show_settings(viewer)

        def choose(control, label):
            _click(viewer, _item_center(viewer, "begin_combo", control))
            _click(viewer, _item_center(viewer, "selectable", viewer.app.localizer.text(label)))

        lo, hi = _item_rect(viewer, "slider_int", "##recording_crf")
        _click(viewer, (lo[0] + (hi[0] - lo[0]) * 0.3, (lo[1] + hi[1]) * 0.5))
        crf = viewer.app.recording_config.crf
        assert 0 < crf < 25
        choose("##recording_rate_control", "Target bitrate")
        with pytest.raises(AssertionError):
            _item_rect(viewer, "slider_int", "##recording_crf")
        lo, hi = _item_rect(viewer, "input_float", "##recording_bitrate")
        _click(viewer, (lo[0] + 20 * scale, (lo[1] + hi[1]) * 0.5))
        io = imgui.get_io()
        modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        io.add_key_event(modifier, True)
        io.add_key_event(imgui.Key.a, True)
        viewer.sync()
        io.add_key_event(imgui.Key.a, False)
        io.add_key_event(modifier, False)
        io.add_input_characters_utf8("6.5")
        viewer.sync()
        io.add_key_event(imgui.Key.enter, True)
        viewer.sync()
        io.add_key_event(imgui.Key.enter, False)
        viewer.sync()
        choose("##recording_preset", "Slow")
        choose("##recording_pixel_format", "Full chroma (4:4:4)")
        config = viewer.app.recording_config
        assert config.rate_control == "bitrate" and config.bitrate_mbps == 6.5
        assert config.encoder_preset == "slow" and config.pixel_format == "yuv444p"
        saved = viewer.app.localizer.preference("recording")
        assert RecordingConfig.from_mapping(saved) == config
        choose("##recording_rate_control", "Quality priority")
        assert viewer.app.recording_config.crf == crf
        assert viewer.app.recording_config.bitrate_mbps == 6.5
        with pytest.raises(AssertionError):
            _item_rect(viewer, "input_float", "##recording_bitrate")
        for title, function, control in (
            ("Quality (CRF)", "slider_int", "##recording_crf"),
            ("Encoding speed", "begin_combo", "##recording_preset"),
            ("Color sampling", "begin_combo", "##recording_pixel_format"),
        ):
            lo, hi = _item_rect(viewer, function, control)
            label_lo, label_hi = _item_rect(viewer, "text", viewer.app.localizer.text(title))
            assert label_hi[0] < lo[0]
            assert label_lo[1] < hi[1] and label_hi[1] > lo[1]
