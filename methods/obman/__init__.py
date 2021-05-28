from torch import nn
from manopth.manolayer import ManoLayer


import utils
from methods.modelutils import encoders, layers
from methods.obman.model.obman import ObMan
from methods.obman.trainer import ObManTrainer


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
  model = nn.DataParallel(model).to(device)

  ### load pretrained ####
  pretrained_file = cfg.get('pretained_file', None)
  if pretrained_file:
    utils.load_checkpoint(pretrained_file, model)
    print(f'pretrained weight from {pretrained_file} is loaded.')

  #### freeze modules ####
  for module, part_name in [(model.module.encoder,'encoder'),
                            (model.module.mano_base,'mano_base'),
                            (model.module.R_head,'R_head'),
                            (model.module.pose_head,'pose_head'),
                            (model.module.st_head,'st_head'),
                            (model.module.obj_st_head, 'obj_st_head')]:
    if cfg.freeze.get(part_name, False):
      for name,p in module.named_parameters():
        p.requires_grad = False
        print(f'{name} is freezed.')


  return model



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

