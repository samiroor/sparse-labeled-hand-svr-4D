import os
import logging

import torch
from torch import nn

from manopth.manolayer import ManoLayer

from methods.modelutils import encoders, layers
from methods.obman.model.obman import ObMan
from methods.obman.trainer import ObManTrainer



optimizer_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'LBFGS':torch.optim.LBFGS,
    }


encoder_dict = {'resnet18': encoders.Resnet18}
mano_base_dict = {'fcs': layers.fcs}
st_head_dict = {'fcs': layers.fcs}



#### model ######

def get_model(
    cfg,
    device=None,
    **kwargs,
    ):


  # shortcuts
  c_dim = cfg.get('c_dim', 512) # encoder output dim
  h_dim = cfg.get('h_dim', 512) # pose & shape head input dim

  return_base_features = cfg.get('return_base_features', False)
  return_encoder_features = cfg.get('return_encoder_features', False)
  adaptor_path = cfg.get('adaptor_path', None)

  annot_metric = cfg.get('annot_metric', 'm')

  # projection setup
  off_z = cfg.get('off_z', 0.4) # camera distance from object in meter.
  scale_factor = cfg.get('scale_factor', 0.0001)
  trans_factor = cfg.get('trans_factor', 100.0)

  # mano
  ncomps = cfg.get('ncomps', 15)
  use_pca = cfg.get('use_pca', True)
  use_shape = cfg.get('use_shape', True)
  ref_side = cfg.get('ref_side', 'right')
  center_idx = cfg.get('center_idx', 0)
  flat_hand_mean = cfg.get('flat_hand_mean', False)
  root_rot_mode = cfg.get('root_rot_mode','axisang')
  joint_rot_mode = cfg.get('joint_rot_mode','axisang')
  mano_root = cfg.get('mano_root', 'misc/mano_v1_2/models')

  # arch
  encoder = cfg.get('encoder', 'resnet18')
  encoder_kwargs = cfg.get('encoder_kwargs', {})

  mano_base = cfg.get('mano_base', 'fcs')
  mano_base_kwargs = cfg.get('mano_base_kwargs', {
    'ch_in': c_dim,
    'ch_out': h_dim,
    'ch_hiddens':[c_dim],
    'activation':'relu',
    'dropout':0.0,
    'flatten':False,
    })

  st_head = cfg.get('st_head', 'fcs') # scale-trans head
  st_head_kwargs = cfg.get('st_head_kwargs', {
    'ch_in': c_dim,
    'ch_out': 3,
    'ch_hiddens':[c_dim//2],
    'activation':'relu',
    'dropout':0.0,
    'flatten':False,
    })

  obj_st_head = cfg.get('obj_st_head', 'fcs') # rot-scale-trans obj head
  obj_st_head_kwargs = cfg.get('obj_st_head_kwargs', {
    'ch_in': c_dim,
    'ch_out': 6,
    'ch_hiddens':[c_dim//2],
    'activation':'relu',
    'dropout':0.0,
    'flatten':False,
    })

  manolayer = ManoLayer(
      ncomps=ncomps,
      use_pca=use_pca,
      side=ref_side,
      center_idx=center_idx,
      flat_hand_mean=flat_hand_mean,
      root_rot_mode=root_rot_mode,
      joint_rot_mode=joint_rot_mode,
      mano_root=mano_root,
      )

  encoder = encoder_dict[encoder](
      c_dim=c_dim,
      **encoder_kwargs).to(device) if encoder in encoder_dict else None

  mano_base = mano_base_dict[mano_base](
      **mano_base_kwargs
      ).to(device) if mano_base in mano_base_dict else None


  shape_head = nn.Linear(h_dim, 10) if use_shape else None
  if root_rot_mode == "axisang":
    R_head = nn.Linear(h_dim, 3)

  if joint_rot_mode == "axisang":
    pose_head = nn.Linear(h_dim, ncomps)
    if not use_pca: assert ncomps==45, 'Expected 15 joint x 3 axis-angle parameters'

  st_head = st_head_dict[st_head](**st_head_kwargs)
  obj_st_head = st_head_dict[obj_st_head](**obj_st_head_kwargs) if obj_st_head is not None else None

  ### create model ####
  model = ObMan(
      encoder=encoder,
      mano_base=mano_base,
      manolayer=manolayer,
      R_head=R_head,
      pose_head=pose_head,
      shape_head=shape_head,
      st_head=st_head,
      obj_st_head=obj_st_head,
      scale_factor=scale_factor,
      trans_factor=trans_factor,
      off_z=off_z,
      return_base_features=return_base_features,
      return_encoder_features=return_encoder_features,
      adaptor_path=adaptor_path,
      annot_metric=annot_metric,
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


  #### freeze modules ####
  freeze = cfg.get('freeze', {})
  if bool(freeze):

    if freeze.get('whole', False):
      for name, p in model.named_parameters():
        p.requires_grad = False
        logging.info(f'{name} is freezed.')

    else:
      module_dict ={
          'encoder': model.encoder,
          'mano_base': model.mano_base,
          'R_head': model.R_head,
          'pose_head': model.pose_head,
          'shape_head': model.shape_head,
          'st_head':model.st_head,
          'obj_st_head':model.obj_st_head,
          }

      for module_name in freeze:
        if freeze.get(module_name, False):
          for name, p in module_dict[module_name].named_parameters():
            p.requires_grad = False
            logging.info(f'{name} is freezed.')


  return model


#### optimizer ######


def adjust_parameter_groups(model, opt_kwargs, lr):
  encoder_group = []
  st_head_group = []
  mano_group = []

  for name, p in model.named_parameters():
    if p.requires_grad:
      if 'encoder' in name:
        encoder_group.append(p)
      elif 'st_head' in name:
        st_head_group.append(p)
      elif 'mano_base' in name or '_head' in name:
        mano_group.append(p)
      else:
        logging.warn(f"Parameter {name} has default lr")

  groups = [dict(params=encoder_group,
                 lr=opt_kwargs.get('encoder_lr', lr)),
            dict(params=st_head_group,
                 lr=opt_kwargs.get('st_head_lr', lr)),
            dict(params=mano_group,
                 lr=opt_kwargs.get('mano_lr', lr))]
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
    'joints3d':(0.5,'l2'),
    'objverts3d':(0.5,'l2'),
    'mano_shapes':(5e-7,'penalty'),
    'mano_poses':(5e-6,'penalty'),
    })

  trainer = ObManTrainer(
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

