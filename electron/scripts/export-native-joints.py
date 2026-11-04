"""Export scalar-joint range, tick and precision-rail geometry from production."""
from pathlib import Path
import json
import numpy as np
from mojive.types import CameraView
from mojive.interaction.gizmo import world_scale
from mojive.ui.gizmo.joint_ranges import _JointRanges
from mojive.ui.gizmo.state import _JointRangeState
from mojive.ui.gizmo.projection import _basis_from_z
owner=_JointRanges();result=[]
for kind,lower,upper,value in [('hinge',-.05,.05,.01),('hinge',-2,2,.3),('hinge',-4,4,.3),('slide',-.01,.01,.003),('slide',-.7,.8,.1)]:
 for eye in ((2,-3,2),(0,-3,.4)):
  camera=CameraView(eye=np.array(eye),target=np.zeros(3),up=np.array([0,0,1]),aspect=1.5)
  axis=np.array([0,0,1]) if kind=='hinge' else np.array([1,0,0]);basis=_basis_from_z(axis)
  origin=np.array([.2,0,.1]);position=origin if kind=='hinge' else origin+axis*value
  state=_JointRangeState(kind,value,lower,upper,1,1)
  projection=owner._hinge_range_projection(camera,(0,0,900,600),1,state,position,basis) if kind=='hinge' else owner._slide_range_projection(camera,(0,0,900,600),1,state,position,basis)
  rail=owner._joint_precision_projection((0,0,900,600),1,state,projection)
  fields={}
  if projection is not None:
   for name in ('alpha','allowed','complement','full_range','current_tick','lower_tick','upper_tick','lower','current','upper'):
    if not hasattr(projection,name):continue
    item=getattr(projection,name)
    fields[name]=item.tolist() if hasattr(item,'tolist') else [x.tolist() for x in item] if isinstance(item,tuple) else item
  result.append({'kind':kind,'lower':lower,'upper':upper,'value':value,'eye':eye,'origin':origin.tolist(),'axis':axis.tolist(),'scale':world_scale(camera,position,600),'view':camera.view_matrix().tolist(),'projection':camera.proj_matrix().tolist(),'expected':fields,'rail':None if rail is None else {'start':rail.start.tolist(),'end':rail.end.tolist(),'current':rail.current.tolist(),'source':rail.source.tolist(),'rect':rail.panel_rect}})
Path(__file__).resolve().parents[1].joinpath('tests/joint-native-reference.json').write_text(json.dumps(result))
