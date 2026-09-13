from mojive.tools.ui_runtime import _capture_size


def test_runtime_gallery_preserves_workspace_at_extreme_ui_scale(monkeypatch) -> None:
    monkeypatch.delenv("MOJIVE_UI_SCALE", raising=False)
    assert _capture_size(1920, 1080) == (1920, 1080)

    monkeypatch.setenv("MOJIVE_UI_SCALE", "2")
    assert _capture_size(1920, 1080) == (1920, 1080)

    monkeypatch.setenv("MOJIVE_UI_SCALE", "4")
    assert _capture_size(1920, 1080) == (3840, 2160)
