"""Export independent production geometry references for the Electron port's numerical tests."""
import json
from pathlib import Path
import numpy as np
from mojive.geometry2d.curves import smooth_lollipop_points, smooth_affine_corners, arc_ribbon_mesh, capped_polyline_points
from mojive.ui.viewcube import _back_alpha
from mojive.interaction.gizmo import axis_arrow_polygon, dimension_axis_polygon, project_rotation_arc, world_scale
from mojive.types import CameraView
OUT = Path(__file__).resolve().parents[1] / 'tests'
r = {'source': 'python/geometry2d/curves.py and python/ui/viewcube.py', 'curves': [], 'fade': [], 'strokes': [], 'g3lines': [], 'railSegments': []}
for distance in (0, 4, 9.5, 9.50001, 9.6, 10, 10.5, 11.49, 11.5, 15, 24, 29):
    r['curves'].append({'distance': distance, 'radius': 9.5, 'width': 2, 'points': smooth_lollipop_points(distance, 9.5, 2)})
for depth in (-1, 0, .5, .72, .78, .84, .9, .96, 1):
    r['fade'].append([depth, _back_alpha(depth)])
paths = []
for height in (64, 24, 8, 3, .7):
    angles = np.linspace(0, np.pi, 33)
    paths.append((np.column_stack((68*np.cos(angles)+100, height*np.sin(angles)+100)), 3.5))
paths.extend((np.asarray(points), 2.5) for points in ([[0, 0], [20, 0]], [[4, 8], [17, 36]], [[1, 1], [1.1, 1]]))
for points, width in paths:
    r['strokes'].append({'points': points.tolist(), 'width': width, 'outline': arc_ribbon_mesh(points, None, None, width, round_caps=True, smoothing=0)[0].tolist()})
for start, end, width in [((0,0),(14,0),3.5), ((2,4),(19,31),2), ((0,0),(.1,0),3.5)]:
    for round_start,round_end in [(True,True),(False,True),(True,False),(False,False)]:
        r['g3lines'].append({'start':start,'end':end,'width':width,'roundStart':round_start,'roundEnd':round_end,'outline':capped_polyline_points((start,end),width,round_start=round_start,round_end=round_end)})
for round_start, round_end in [(True,False),(False,False),(False,True)]:
    points=((10,20),(30,20));width=4
    r['railSegments'].append({'points':points,'width':width,'roundStart':round_start,'roundEnd':round_end,'outline':arc_ribbon_mesh(points,None,None,width,round_start=round_start,round_end=round_end,smoothing=0)[0].tolist()})
(OUT / 'viewcube-native-reference.json').write_text(json.dumps(r))
r = {'source': 'python/interaction/gizmo.py and python/geometry2d/curves.py', 'arrows': [], 'roundedArrows': [], 'dimensions': [], 'planes': [], 'arcs': []}
for end in ((88, 0), (0, 88), (17.55, -32.83), (1, 1), (6, 0), (1e-5, 0)):
    for key, function in [('arrows', axis_arrow_polygon), ('dimensions', dimension_axis_polygon)]:
        r[key].append({'start': [0, 0], 'end': end, 'points': function((0, 0), end, 1).tolist()})
for end in ((88,0),(17.55,-32.83),(6,0),(1,1),(1e-5,0)):
    r['roundedArrows'].append({'start':[0,0],'end':end,'points':axis_arrow_polygon((0,0),end,1,round_tail=True).tolist()})
for points in (((0, 0), (25, 0), (25, 25), (0, 25)), ((13.2, 15.8), (25.3, 16.2), (18.4, 18.3), (9.1, 17.6)), ((10, 4), (10.01, 4.08), (13, 20), (12.99, 19.92))):
    r['planes'].append({'points': points, 'rounded': smooth_affine_corners(points, 2).tolist()})
for eye, origin, orthographic in [((2, -3, 1), (.4, 0, .3), False), ((0, -3, .03), (0, 0, 0), False), ((.15, -.2, .05), (.02, 0, 0), False), ((2, -3, 1), (1.4, .7, -.2), True), ((2, 0, 0), (0, 0, 0), False)]:
    camera = CameraView(eye=np.array(eye), target=np.zeros(3), up=np.array([0, 0, 1]), aspect=1.5, orthographic=orthographic)
    for axis in range(3):
        scale = world_scale(camera, origin, 600)
        r['arcs'].append({'eye': eye, 'origin': origin, 'orthographic': orthographic, 'axis': axis, 'radius': scale*.78, 'view': camera.view_matrix().tolist(), 'projection': camera.proj_matrix().tolist(), 'points': project_rotation_arc(camera, origin, np.eye(3), scale, axis, (0, 0, 900, 600))[:, :2].tolist()})
(OUT / 'gizmo-native-reference.json').write_text(json.dumps(r))
print('Exported production geometry references.')
