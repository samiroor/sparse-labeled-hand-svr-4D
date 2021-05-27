import torch
import torch.nn as nn

from methods.pairflow.model.pairflow import ContextLayer, PairFlow
from methods.pairflow.model.resflow import ResFlow
from methods.pairflow.model.velocity_field import VelocityField
from methods.obman import get_model as obman_get_model
from methods.graph import get_model as graph_get_model

mesh_predictor_dict = {'obman': obman_get_model,
                       'graph': graph_get_model}
flow_dict = {'resflow': ResFlow}

context_layer_dict = {'subtractive': ContextLayer}


def get_model(
    cfg,
    device=None,
    **kwargs,
    ):

  mesh_predictor = mesh_predictor_dict[cfg.mesh_predictor]
  flow = ...
  context_layer = ...

  model = PairFlow(mesh_predictor, context_layer, flow)
