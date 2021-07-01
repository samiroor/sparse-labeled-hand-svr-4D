import os
import logging

import torch
import torch.nn as nn

from methods.modelutils import layers, attentions
from methods.obman import get_model as obman_get_model
from methods.graph import get_model as graph_get_model

from methods.pairflow.model.resflow import ResFlow
from methods.pairflow.model.rotation_translation_flow import RotationTranslationFlow

from methods.pairflow.model.velocity_field import VelocityField
from methods.pairflow.model.rotation_translation_velocity_field import RotationTranslationVelocityField

from methods.pairflow.model.context_head import OPContextHead

#from methods.pairflow.model.pairflow import PairFlow
from methods.pairflow.trainer import PairFlowTrainer
from methods.pair_disentangle_flow.model.pair_dientangle_flow import PairDistFlow


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
    'fcs': layers.fcs,
    'cross_attention': attentions.MultiHeadCrossAttention,
    }


velocity_field_dict = {
    'resblock':VelocityField,
    'rotation_translation': RotationTranslationVelocityField,
    }

flow_dict = {
    'resflow': ResFlow,
    'rotation_translation':RotationTranslationFlow,
    }


#### model ######

def get_context_head(cfg, c_dim=512):

  context_base = cfg.get('context_base', 'fcs')
  context_base_kwargs = cfg.get('context_base_kwargs', {
    'ch_in':c_dim, # should be equal to mesh predictor featur key out-channels
    'ch_out':c_dim,
    'ch_hiddens':[c_dim],
    'activation':'relu',
    'dropout':0.0,
    'flatten':False,
    })

  context_head = cfg.get('context_head', None)
  context_head_kwargs = cfg.get('context_head_kwargs',{
    "nheads":1,
    "in_dim":c_dim,  # should be equal to context base out-chnannels feature
    "key_dim":c_dim,  # could be changed (optionally)
    "value_dim":c_dim, # should be equal to velocity field c_dim  features 
    "qk_same_proj":True,
    "positional_encoding":False,
    "max_pos_size":100,
    })

  _context_base = context_base_dict[context_base](**context_base_kwargs)\
      if context_base is not  None else None
  _context_head = context_head_dict[context_head](**context_head_kwargs)\
      if context_head is not None else None

  context_head = OPContextHead(
      context_base=_context_base,
      context_head=_context_head,
      op=cfg.get('op', 'subtract'),
      )

  return context_head


def get_velocity_field(cfg):
  velocity_field = cfg.get('velocity_field', 'rotation_translation')
  velocity_field_kwargs = cfg.get('velocity_field_kwargs', {
    "in_dim":51,
    "angular_dim":48,
    "c_dim":512,
    "hidden_size":512,
    "n_blocks":1,
    "cond_mode":"cat",
    "activation":"tanh",
    "rot_mode":"axisang",
    "disentangle":False,
    "temperature_scaling":True,
    #"positional_encoding":False,
    #"positional_encoding_kwargs":{},
    })

  return velocity_field_dict[velocity_field](**velocity_field_kwargs)


def get_flow(cfg, velocity_field):

  flow = cfg.get('flow', 'rotation_translation')
  flow_kwargs = cfg.get('flow_kwargs', {
    "T":1,
    "return_multiTime":False,
    })

  return flow_dict[flow](velocity_field, **flow_kwargs)


def get_model(
    cfg,
    device=None,
    **kwargs,
    ):

  # shortcuts
  morph_key = cfg.get('morph_key', 'mano_params')
  morph_shape = cfg.get('morph_shape', False)
  use_gt_mano_shape = cfg.get('use_gt_mano_shape', True)
  invert_from_gts = cfg.get('invert_from_gts', True)
  infer_from_gts = cfg.get('infer_from_gts', True)

  # mesh predictor
  mesh_predictor = cfg.get('mesh_predictor', 'obman')
  mesh_predictor_kwargs = cfg.mesh_predictor_kwargs
  assert bool(mesh_predictor_kwargs), "Expected a config for mesh_predictor!"

  assert mesh_predictor_kwargs.return_tail_features
  assert mesh_predictor_kwargs.return_encoder_features

  # mesh predictor
  mesh_predictor = mesh_predictor_dict[mesh_predictor](
      mesh_predictor_kwargs,
      device=device,
      )

  # context head
  context_head_root = get_context_head(cfg['context_root'])
  context_head_rel = get_context_head(cfg['context_rel'])

  # flow velocity field
  velocity_field_root = get_velocity_field(cfg['velocity_field_root'])
  velocity_field_rel = get_velocity_field(cfg['velocity_field_rel'])

  # make flow with that velocity field
  flow_root = get_flow(cfg['flow_root'], velocity_field_root)
  flow_rel = get_flow(cfg['flow_rel'], velocity_field_rel)

  # make pairflow
  model = PairDistFlow(
      mesh_predictor=mesh_predictor,
      context_head_root=context_head_root,
      context_head_rel=context_head_rel,
      flow_root=flow_root,
      flow_rel=flow_rel,
      morph_key=morph_key,
      feature_key_root="encoder_features",
      feature_key_rel="tail_features",
      morph_shape=morph_shape,
      use_gt_mano_shape=use_gt_mano_shape,
      invert_from_gts=invert_from_gts,
      infer_from_gts=infer_from_gts,
      **kwargs,
      )

  ### load pretrained ####
  pretrained_file = cfg.get('pretrained_file', None)

  if pretrained_file:
    assert os.path.exists(pretrained_file), f'Expected file {pretrained_file} is not found'
    ckpt_dict = torch.load(pretrained_file)['state_dict']
    ckpt_dict = {k.split('module.')[-1]:v for k,v in ckpt_dict.items()}

    model_dict = model.state_dict()
    for k in model_dict:
      if k not in ckpt_dict:
        logging.warn("Missed pretrained weight: {}".format(k))
    model_dict.update(ckpt_dict)
    model.load_state_dict(model_dict)
    logging.info(f'pretrained weight from {pretrained_file} is loaded.')

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

