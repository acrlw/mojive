"""The recording acceptance scenario operates real controls in clipped HiDPI docks."""

import pytest

from mojive.app.composition import build
from mojive.config import PanelConfig, RecordingConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.recording_layers import _run

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("language,scale", [("en", 1.0), ("zh_CN", 2.25)])
def test_layers_acceptance_scrolls_controls_and_records_visible_changes(
    tmp_path, monkeypatch, language, scale
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_LANGUAGE", language)
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    config = ViewerConfig(
        panels={"layers": PanelConfig(open=True), "inspector": PanelConfig(open=False)},
        recording=RecordingConfig(copy_to_clipboard=False),
    )
    with build(
        resolve("joint_gizmo"),
        config=config,
        vsync=False,
        show_window=False,
        width=1600,
        height=1000,
    ) as viewer:
        _run(viewer, tmp_path)
