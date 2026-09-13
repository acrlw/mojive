import numpy as np
from PIL import Image

from mojive.tools.mouse_hint_icons import export_icons


def test_mouse_icon_export_writes_black_and_true_transparent_shell_variants(tmp_path) -> None:
    paths = export_icons(tmp_path, 128)

    assert len(paths) == 12
    assert all(path.is_file() for path in paths)
    for button in ("left", "right", "wheel"):
        black = Image.open(tmp_path / f"mouse-{button}-black-shell.png").convert("RGBA")
        transparent = Image.open(tmp_path / f"mouse-{button}-transparent-shell.png").convert("RGBA")
        assert black.size == transparent.size == (128, 128)
        black_pixels = np.asarray(black)
        transparent_pixels = np.asarray(transparent)
        assert np.any(np.all(black_pixels[..., :3] == 0, axis=2) & (black_pixels[..., 3] > 128))
        assert not np.any(
            np.all(transparent_pixels[..., :3] == 0, axis=2) & (transparent_pixels[..., 3] > 128)
        )
        assert transparent.getchannel("A").getextrema() == (0, 255)
