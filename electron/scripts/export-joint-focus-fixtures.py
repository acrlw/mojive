"""Export production joint focus direction references; run from repo with .venv/bin/python."""
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from mojive.ui.camera import (
    elevated_focus_view_direction, closest_perpendicular_view_direction,
    oblique_axis_view_directions,
)
from mojive.ui.app.navigation import _Navigation

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / 'tests/fixtures/joint-focus-native.json'

def main():
    cases = []
    for axis in ([0,0,1], [1,0,0], [.2,.8,-.3], [0,0,-1]):
        for offset in ([3,-2,1], [-1,-2,-3], [0,0,5], [0,0,-5], [1e-12,0,0]):
            for markers in ([], [[1,0,0], [0,1,0], [.3,-.8,.1]]):
                fallback = [1,0,0]
                candidates = oblique_axis_view_directions(axis,offset,fallback,35,marker_directions=markers)
                navigation = SimpleNamespace(
                    camera=SimpleNamespace(direction=lambda: np.array([1,0,0],np.float64)),
                    session=SimpleNamespace(adapter=SimpleNamespace(caps=SimpleNamespace(raycast=False))),
                )
                chosen = _Navigation._least_occluded_joint_direction(
                    navigation,np.zeros(3),.2,candidates,SimpleNamespace(body_index=1),np.array(offset,np.float64),
                    preferred_up=(0,0,1),minimum_elevation_degrees=30)
                cases.append(dict(axis=axis,offset=offset,fallback=fallback,markers=markers,
                    elevated=elevated_focus_view_direction(offset,fallback).tolist(),
                    perpendicular=closest_perpendicular_view_direction(axis,offset,fallback).tolist(),
                    candidates=[value.tolist() for value in candidates],chosen=chosen.tolist()))
    result = dict(schema=1,source={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        for path in ['python/ui/camera.py','python/ui/app/navigation.py','python/ui/app/support.py']},
        tolerance=2e-7,cases=cases)
    DESTINATION.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
