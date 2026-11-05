"""Export the native capsule profile for independent Electron geometry checks."""
import json
import math
from pathlib import Path

import numpy as np

from mojive.geometry2d.curves import (
    _cap_geometry,
    _fit_cap_smoothing,
    _turn_end,
    capsule_end_padding,
    smooth_capsule_points,
    turn_curvature,
)

cases = []
for width, height in ((120, 28), (54, 24), (30, 28), (28, 28), (28, 120)):
    for smoothing in (0, 0.382, 0.6, 1):
        long, short = max(width, height), min(width, height)
        fitted = _fit_cap_smoothing(long / (2 * short), smoothing)
        scale = short / _turn_end(math.pi, fitted)[1]
        cap, depth = _cap_geometry(short, fitted, 0.025)
        ramp, total = fitted * math.pi, (1 + fitted) * math.pi
        breaks = sorted({0, ramp, total / 2, math.pi, total})
        distances = [0.0]
        for lo, hi in zip(breaks, breaks[1:]):
            count = max(1, math.ceil((hi - lo) / math.sqrt(8 * 0.025 / scale)))
            distances.extend(np.linspace(lo, hi, count + 1)[1:])
        angles, _, _ = turn_curvature(distances, math.pi, fitted)
        points = cap + (long - depth, short / 2)
        normals = np.column_stack((np.sin(angles), -np.cos(angles)))
        if height > width:
            points = np.column_stack((short - points[:, 1], points[:, 0]))
            normals = np.column_stack((-normals[:, 1], normals[:, 0]))
        contour = np.asarray(smooth_capsule_points(0, 0, width, height, smoothing))
        cases.append({
            "width": width, "height": height, "smoothing": smoothing,
            "effectiveSmoothing": fitted, "minimumRadius": scale, "depth": depth,
            "opticalEndPadding": capsule_end_padding(width, height, smoothing),
            "bounds": [*contour.min(axis=0), *contour.max(axis=0)],
            "samples": np.column_stack((points, normals)).tolist(),
        })
output = Path(__file__).resolve().parents[1] / "tests/fixtures/capsule-native.json"
output.write_text(json.dumps({"source": "python/geometry2d/curves.py", "cases": cases}) + "\n")
print(f"Exported {len(cases)} native capsule references.")
