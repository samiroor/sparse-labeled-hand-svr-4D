#from psbody.mesh import Mesh

from .mesh_sampling import generate_transform_matrices


def decimate_mesh(
    template_file,
    ds_factors,
    ):

  mesh = Mesh(filename=template_file)
  _, A, D, U, F, V = generate_transform_matrices(
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
