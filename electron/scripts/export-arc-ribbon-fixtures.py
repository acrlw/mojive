"""Export native strip topology, including near-closed authored hinge ranges.

Run with .venv/bin/python electron/scripts/export-arc-ribbon-fixtures.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from mojive.geometry2d.curves import arc_ribbon_mesh

ROOT = Path(__file__).resolve().parents[2]
destination = Path(__file__).resolve().parents[1] / "tests/fixtures/arc-ribbon-native.json"
cases = []
for height in (68.64, 42.0, 5.0, 0.7):
    for span in (6.28318, 5.8, -5.8, np.pi):
        angles = np.linspace(-3.14159, -3.14159 + span, 129)
        points = np.column_stack((450 + 68.64 * np.cos(angles), 300 + height * np.sin(angles)))
        vertices, indices = arc_ribbon_mesh(points, None, None, 3.5, round_caps=True, smoothing=0)
        cases.append({
            "height": height, "span": float(span), "points": points.tolist(),
            "width": 3.5, "vertices": vertices.tolist(), "indices": list(indices),
        })
source = ROOT / "python/geometry2d/curves.py"
destination.write_text(json.dumps({
    "source": {"path": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    "cases": cases,
}, separators=(",", ":")) + "\n")
