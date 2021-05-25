"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear

class GraphCNN(nn.Module):

  def __init__(self,
               in_dim,
               grb_num,
               A,
               ref_vertices):
    """
    Args:
      in_dim (int): feature dimension as input to the graph.
      grb_num (int): graph residual blocks number.
      A (torch.Tensor): NxN Adjacency matrix.
      ref_vertices (torch.Tensor): 3xN template vertices location; could be normalized or not!
    Returns:
    """
    super(GraphCNN, self).__init__()
    self.A = A
    self.ref_vertices = ref_vertices
    assert ref_vertices.shape[0] ==3,"Should be transposed!"

    layers = [GraphLinear(3 + in_dim, in_dim // 2)]
    for i in range(grb_num):
      layers.append(GraphResBlock(in_dim // 2, in_dim // 2, self.A))
    layers.extend([GraphResBlock(in_dim // 2, 64, self.A),
                   GraphResBlock(64, 32, self.A),
                   nn.GroupNorm(32 // 8, 32),
                   nn.ReLU(inplace=True),
                   GraphLinear(32, 3)])

    self.layers = nn.Sequential(*layers)

  def forward(self, features):
    #[3,N] => [B,3,N]
    ref_vertices = self.ref_vertices[None].expand(features.shape[0], -1, -1)
    #[B,D] => [B,D,N]
    features = features[:,:,None].expand(-1, -1, ref_vertices.shape[-1])
    inp = torch.cat([ref_vertices, features], dim=1)
    #[B,3,N]
    verts = self.layers(inp)
    return verts
