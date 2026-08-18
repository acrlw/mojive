"""Real GPU capture through the local control service."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.control_rpc import ControlService

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


def test_control_service_captures_rgb_depth_and_segmentation(tmp_path):
    asset = Path("assets/test_scene.xml").resolve()
    service = ControlService(MuJoCoAdapter(asset), asset)
    try:
        rgb = service.dispatch(
            "capture",
            {"mode": "rgb", "width": 128, "height": 96, "output": str(tmp_path / "rgb.png")},
        )
        depth = service.dispatch(
            "capture",
            {"mode": "depth", "width": 128, "height": 96, "output": str(tmp_path / "depth.npy")},
        )
        segmentation = service.dispatch(
            "capture",
            {
                "mode": "segmentation",
                "width": 128,
                "height": 96,
                "output": str(tmp_path / "segmentation.npy"),
            },
        )
    finally:
        service.close()

    assert Image.open(rgb["path"]).size == (128, 96)
    assert np.load(depth["path"]).shape == (96, 128)
    assert np.load(segmentation["path"]).shape == (96, 128, 2)
