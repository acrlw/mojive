from __future__ import annotations

import numpy as np
import pytest

from forge_viewer.tools.tool_icons import render_tool_shell_icon


@pytest.mark.parametrize(
    ("kind", "space"),
    (("move", "world"), ("frame", "world"), ("frame", "body")),
)
def test_tool_shell_debug_variants_share_the_production_core(kind, space):
    transparent = np.asarray(render_tool_shell_icon(256, kind, space))
    black = np.asarray(render_tool_shell_icon(256, kind, space, shell=(0, 0, 0, 255)))

    transparent_core = (transparent[..., 3] > 192) & np.all(
        transparent[..., :3] > 192,
        axis=-1,
    )
    black_core = (black[..., 3] > 192) & np.all(black[..., :3] > 192, axis=-1)
    black_shell = (black[..., 3] > 192) & np.all(black[..., :3] < 32, axis=-1)

    # Every opaque white pixel still belongs to the production silhouette.
    # Frame variants additionally overlay the center-dot outline on the three
    # shafts so the origin remains independently legible.
    assert not np.any(black_core & ~transparent_core)
    replaced_core = transparent_core & ~black_core
    if kind == "move":
        assert np.count_nonzero(replaced_core) < (np.count_nonzero(transparent_core) * 0.02)
    else:
        # Sample the right-hand sector, away from all three shafts: only the
        # black variant exposes the center dot's independent shell there.
        assert transparent[128, 143, 3] <= 4
        assert black[128, 143, 3] == 255
        assert np.all(black[128, 143, :3] < 8)
        # The same shell is genuinely transparent where the upper shaft meets
        # the dot; it is not repainted with an assumed capsule background.
        assert transparent[111, 128, 3] <= 4
        assert black[111, 128, 3] == 255
        assert np.all(black[111, 128, :3] < 8)
    assert np.count_nonzero(black_shell) > 0
    assert np.count_nonzero(black[..., 3] > 128) > np.count_nonzero(transparent[..., 3] > 128)
    assert not np.any((transparent[..., 3] > 192) & np.all(transparent[..., :3] < 32, axis=-1))
