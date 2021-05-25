import os
import pickle
import numpy as np
from psbody.mesh import Mesh

import torch
import torch.nn as nn

from .graphutils import GraphCNN, GraphMesh, mesh_sampling


class GraphHead(nn.Module):

  def __init__(self,
               h_dim=512,
               block_num=4,
               ds_factors=[2],
               reference_mesh_file='misc/graph/template.obj',
               downsample_file='misc/graph/downsample{}.pkl',
               mean_file='?mean.npy',
               std_file='?std.npy',
               device=None,
               **kwargs):
    """
    Args:
      h_dim (int): dimension of input to Graph head.
      block_num (int): number of Graph Residual Blocks.
      ds_factors (list): list of decimation factors to downsample reference mesh.
      reference_mesh_file (str): directory of a reference mesh topology.
      downsample_method (str): type of downsampling.
      downsample_directory (str): directory of downsampled mesh topologies.
      mean_file (str): mean of train meshes; used for mesh normalization.
      std_file (str): std of train meshes; used for mesh normalization.
      device (torch.device):
    """

    super(GraphHead, self).__init__()

    self.h_dim = h_dim
    self.block_num = block_num

    self.ds_factors = ds_factors
    self.downsample_file = (downsample_file).format('_'.join(map(str,ds_factors)))

    self.mean_file = mean_file
    self.std_file = std_file
    self.reference_mesh_file = reference_mesh_file

    self.mean=None
    self.std=None
    self.device = device

    self._rebuild()


  def _rebuild(self):

    self.gcnn = None
    self._sample_mesh = None
    self.reference_mesh = Mesh(filename=self.reference_mesh_file)

    # normalizing reference mesh
    if os.path.exists(self.mean_file) and os.path.exists(self.std_file):
      print("Normalizing reference mesh using provided mean&std files!")
      self.mean = np.load(self.mean_file)
      self.std = np.load(self.std_file)

      v_nrm = (self.reference_mesh.v - self.mean) / self.std
      self.reference_mesh = Mesh(v=v_nrm, f=self.reference_mesh.f)

      self.mean = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
      self.std = torch.tensor(self.std, dtype=torch.float32, device=self.device)

    # downsampling reference mesh
    if not os.path.exists(self.downsample_file):
      print("Generating Transform Matrices ..")
      M,A,D,U,F = mesh_sampling.generate_transform_matrices(self.reference_mesh, self.ds_factors)
      with open(self.downsample_file, 'wb') as fp:
        M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
        pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)

    # graph mesh object 
    self._sample_mesh = GraphMesh(
        downsample_file=self.downsample_file,
        num_downsampling=0,
        nsize=1, #TODO
        ref_faces=torch.tensor(self.reference_mesh.f.astype(np.int32), dtype=torch.int32),
        ref_vertices=torch.tensor(self.reference_mesh.v, dtype=torch.float32),
        device=self.device,
        )

    # graph cnn
    self.gcnn = GraphCNN(
        in_dim=self.h_dim,
        grb_num=self.block_num,
        A=self._sample_mesh.adjmat,
        ref_vertices=self._sample_mesh.ref_vertices.t(),
        )


  def forward(self, x):
    verts = self.gcnn(x)
    verts = verts.transpose(1,2)

    if self.mean is not None and self.std is not None:
      verts = verts * self.std + self.mean

    return verts
