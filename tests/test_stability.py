from __future__ import annotations

from pathlib import Path

import pytest

from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.tools.stability import run_stability

pytest.importorskip("mujoco")
pytestmark = [pytest.mark.integration, pytest.mark.physics]


def test_headless_frame_and_large_model_lifecycle_stability(tmp_path: Path) -> None:
    report = run_stability(
        tmp_path / "stability.json",
        frames=100,
        cycles=3,
        bodies=16,
        max_growth_mb=4.0,
    )

    assert report["passed"]
    assert all(report["checks"].values())


def test_mujoco_release_drops_model_specs_and_large_frame_buffers(tmp_path: Path) -> None:
    model = tmp_path / "large.xml"
    model.write_text(
        "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>",
        encoding="utf-8",
    )
    adapter = MuJoCoAdapter(model)
    assert len(adapter._geom_xpos_buf) == 1

    adapter.release()

    assert adapter.model is None and adapter.data is None
    assert adapter._root_spec is None and not adapter._attached_models
    assert adapter._geom_xpos_buf.shape == (0, 3)
    assert adapter._body_xmat_buf.shape == (0, 3, 3)
