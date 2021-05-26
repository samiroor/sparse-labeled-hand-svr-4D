import os
import torch

from  data.datasets.torch_handmodel import Torch_HandModel as HandModel


handmodel = HandModel(
    mano_root='misc/mano_v1_2/models/',
    ref_side='right',
    annot_metric='m',
    use_mean_pose=False,
    )

os.makedirs('misc/?', exist_ok=True)
outmesh_path = 'misc/?/template_right_in_meter.obj'
J,V,F = handmodel(torch.zeros((1,48)))

de = lambda x: x.detach().cpu().numpy()
J,V,F = de(J[0]),de(V[0]),de(F)

with open( outmesh_path, 'w') as fp:
  for v in V:
    fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
  for f in F+1: # Faces are 1-based, not 0-based in obj files 
    fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

