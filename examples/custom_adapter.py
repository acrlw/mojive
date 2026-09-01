"""Implement a small simulation adapter without MuJoCo."""

from __future__ import annotations

import numpy as np

from mojive import AdapterCaps, FrameNeeds, Scene, SceneAdapterBase, build_from_adapter
from mojive.adapters.base import SceneFrame, SceneSource


class OscillatorAdapter(SceneAdapterBase):
    """Move one authored object using an adapter-owned simulation clock."""

    caps = AdapterCaps(name="oscillator", simulation=True, write_pose=True)

    def __init__(self) -> None:
        self.scene = Scene()
        self.scene.plane(name="floor", size=(4.0, 4.0, 0.04))
        self.marker = self.scene.sphere(
            name="marker",
            size=(0.3, 0.3, 0.3),
            position=(0.0, 0.0, 0.5),
            color=(0.2, 0.55, 0.95, 1.0),
        )
        self._time = 0.0
        self._step = 0

    @property
    def structure_revision(self) -> int:
        """Forward structural edits made to the authored scene."""
        return self.scene.structure_revision

    def scene_source(self) -> SceneSource:
        """Return stable geometry, material, light, and hierarchy data."""
        return self.scene.source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        """Return dynamic poses for the current simulation time."""
        frame = self.scene.frame
        frame.time = self._time
        frame.step = self._step
        return frame

    def timestep(self) -> float:
        """Use a 240 Hz simulation step."""
        return 1.0 / 240.0

    def step(self, count: int = 1) -> None:
        """Advance the oscillator and update its scene pose."""
        count = max(0, int(count))
        self._step += count
        self._time += count * self.timestep()
        self.marker.set_pose(
            (
                np.sin(self._time * 2.0),
                0.5 * np.cos(self._time * 2.0),
                0.65,
            )
        )

    def reset(self) -> None:
        """Restore the initial simulation state."""
        self._time = 0.0
        self._step = 0
        self.marker.set_pose((0.0, 0.5, 0.65))


def main() -> None:
    """Run the custom adapter in the standard interactive viewer."""
    viewer = build_from_adapter(OscillatorAdapter(), title="Mojive custom adapter")
    try:
        viewer.run()
    finally:
        viewer.release()


if __name__ == "__main__":
    main()
