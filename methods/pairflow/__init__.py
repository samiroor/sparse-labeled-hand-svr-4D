import logging

import torch
import torch.nn as nn

from methods.modelutils import layers
from methods.obman import get_model as obman_get_model
from methods.graph import get_model as graph_get_model

from methods.pairflow.model.resflow import ResFlow
from methods.pairflow.model.velocity_field import VelocityField
from methods.pairflow.model.context_head import SubtractContextHead
from methods.pairflow.model.pairflow import PairFlow
from methods.pairflow.trainer import PairFlowTrainer



optimizer_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'LBFGS':torch.optim.LBFGS,
    }

mesh_predictor_dict = {
    'obman': obman_get_model,
    'graph': graph_get_model,
    }


context_base_dict = {
    'fcs': layers.fcs,
    }


context_head_dict = {
    'subtract': SubtractContextHead,
    }


velocity_field_dict = {
    'resblock':VelocityField,
    }

flow_dict = {
    'resflow': ResFlow,
    }


#### model ######


def get_model(
    cfg,
    device=None,
    **kwargs,
    ):

  # shortcuts
  morph_key = cfg.get('morph_key', 'verts3d')
  use_tail_features = cfg.get('use_tail_features', False)

  # mesh predictor
  mesh_predictor = cfg.get('mesh_predictor', 'obman')
  mesh_predictor_kwargs = cfg.mesh_predictor_kwargs
  assert bool(mesh_predictor_kwargs), "Expected a config for mesh_predictor!"

  c_dim = mesh_predictor_kwargs.h_dim if use_tail_features else mesh_predictor_kwargs.c_dim

  # context 
  context_base = cfg.get('context_base', 'fcs')
  context_base_kwargs = cfg.get('context_base_kwargs', {
    'ch_in':c_dim,
    'ch_out':c_dim,
    'ch_hiddens':[c_dim//2],
    'activation':'tanh',
    'dropout':0.0,
    'flatten':False,
    })

  context_head = cfg.get('context_head', 'subtract')
  context_head_kwargs = cfg.get('context_head_kwargs', {})

  # flow
  velocity_field = cfg.get('velocity_field', 'resblock')
  velocity_field_kwargs = cfg.get('velocity_field_kwargs', {
    'c_dim':c_dim,
    'hidden_size':c_dim,
    'activation':'tanh',
    'n_blocks':3,
    'use_positional_encoding':False,
    })

  flow = cfg.get('flow', 'resflow')
  flow_kwargs = cfg.get('flow_kwargs', {'T':3})

  # components
  mesh_predictor = mesh_predictor_dict[mesh_predictor](
      mesh_predictor_kwargs,
      device=device,
      )

  context_base = context_base_dict[context_base](**context_base_kwargs) if context_base is not  None else None
  context_head = context_head_dict[context_head](context_base, **context_head_kwargs)

  velocity_field = velocity_field_dict[velocity_field](**velocity_field_kwargs)
  flow = flow_dict[flow](velocity_field, **flow_kwargs)


  model = PairFlow(
      mesh_predictor=mesh_predictor,
      context_head=context_head,
      flow=flow,
      morph_key=morph_key,
      use_tail_features=use_tail_features,
      **kwargs,
      )


  return  model



#### optimizer ######


def adjust_parameter_groups(model, opt_kwargs, lr):
  mesh_predictor_group = []
  context_head_group = []
  velocity_field_group = []

  for name, p in model.named_parameters():
    if p.requires_grad:
      if 'mesh_predictor' in name:
        mesh_predictor_group.append(p)
      elif 'context_head' in name:
        context_head_group.append(p)
      elif 'velocity_field' in name:
        velocity_field_group.append(p)
      else:
        logging.warn(f"Parameter {name} has default lr")

  groups = [dict(params=mesh_predictor_group,
                 lr=opt_kwargs.get('mesh_predictor_lr', lr)),
            dict(params=context_head_group,
                 lr=opt_kwargs.get('context_head_lr', lr)),
            dict(params=velocity_field_group,
                 lr=opt_kwargs.get('velocity_field_lr', lr))]
  return groups


def get_optimizer(
    opt_name,
    opt_kwargs,
    lr,
    model):
  groups = adjust_parameter_groups(model, opt_kwargs, lr)
  optimizer = optimizer_dict[opt_name](groups, lr=lr,
                                       weight_decay=opt_kwargs.get('weight_decay',0.0))
  return optimizer


###### Trainer #####



def get_trainer(
    cfg,
    model,
    optimizer,
    scheduler,
    metrics_fns,
    device,
    exp_dir,
    adjust_bn_params=True,
    grad_norm_clip=False,
    grad_norm_clip_value=10,
    **kwargs,
    ):

  loss_cfg = cfg.get('loss', {
    'flow':{
      'verts3d':(0.5,'l2'),
      },
    'mesh_predictor':{
      'verts3d':(0.5,'l2'),
      #'objverts3d':(0.5,'l2'),
      #'mano_shapes':(5e-7,'penalty'),
      #'mano_poses':(5e-6,'penalty'),
      }
    })

  trainer = PairFlowTrainer(
      model,
      optimizer,
      scheduler,
      metrics_fns,
      device,
      exp_dir,
      loss_cfg,
      adjust_bn_params=True,
      grad_norm_clip=False,
      grad_norm_clip_value=10,
      **kwargs,
      )

  return trainer

