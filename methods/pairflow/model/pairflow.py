import torch
import torch.nn as nn

from methods.modelutils import project
from data import Fields as fields


class ContextLayer(nn.Module):
  def __init__(self, layers):
    super().__init__()
    self.layers = layers

  def deffuse(self, f1, f2):
    return f1 - f2

  def forward(self, f1, f2):
    f1 = self.layers(f1)
    f2 = self.layers(f2)
    return self.deffuse(f1,f2)


class AttentionContextLayer(ContextLayer):
  pass


class PairFlow(nn.Module):

  def __init__(
      self,
      mesh_predictor,
      context_layer,
      flow,
      **kwargs):
    """
    A model that
     During training:
      1) encodes supervised and unsupervised images to get their
         embeddings (f_s and f_u respectively) using mesh_predictor.
      2) decodes the unsupervised features f_u to get 3d and 2d
         vertices and joints using mesh_predictor.
      3) inputs f_s, f_u to context_layer to create context features (ctx)

      4) uses predicted verts3d_u from f_u and ctx as input to flow to
         deform verts3d_u to verts3d_s
    Args:
      mesh_predictor: (nn.Module) that takes images and outputs
                      predicted mesh as well as encoded features.
      context: (nn.Module) that takes featurs of supervised and
               unsupervised images and outputs the corresponding
               context feature.
      flow: (nn.Module) that takes context features and the
            unsupervised predicted mesh and estimates a mesh
            for the supervised one.
    Returns:
      predicted supervised mesh
    """

    super().__init__()

    assert(mesh_predictor.return_encoder_features)
    self.mesh_predictor = mesh_predictor
    self.context_layer = context_layer
    self.flow = flow

  def forward(self, input_batch):
    # TODO: move these key strings to fields
    supervised_mesh_out = self.mesh_predictor(input_batch['supervised'])
    unsupervised_mesh_out = self.mesh_predictor(input_batch['unsupervised'])

    if self.training:
      context = self.context_layer(supervised_mesh_out['encoder_features'],
                                   unsupervised_mesh_out['encoder_features'])
      supervised_pred = self.flow(unsupervised_mesh_out,
                                  cond=context)
      # TODO create output batch
    else:
      context = self.context_layer(unsupervised_mesh_out['encoder_features'],
                                   supervised_mesh_out['encoder_features'])
      # Use GT verts for supervised mesh for testing
      unsupervised_pred = self.flow(input_batch['supervised'],
                                    cond=context)
      # TODO create output batch


    return output_batch

