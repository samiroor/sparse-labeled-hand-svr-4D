#from psbody.mesh import Mesh

from . import mesh_sampling


def decimate_mesh(
    template_file,
    ds_factors,
    ):

  mesh = Mesh(filename=template_file)
  _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh,
        ds_factors,
        )
  return {
       'vertices': V,
       'face': F,
       'adj': A,
       'down_transform': D,
       'up_transform': U,
       }


if __name__ == '__main__':
  import pickle
  ds_factors = [2,2,2,2]
  template_file = 'misc/spiral/template.obj'
  downsample_file = 'misc/spiral/downsample_2222.pkl'
  downsample_dict = decimate_mesh(
        template_file,
        ds_factors,
        )
  with open(downsample_file, 'wb') as fp:
    pickle.dump(downsample_dict, fp)
 
