from torch import nn

import utils
from methods.modelutils import encoders, layers
from methods.graph.model.graph import Graph
from methods.graph.model.graph_head import GraphHead
from methods.graph.trainer import GraphTrainer


encoder_dict = {'resnet18': encoders.Resnet18}
graph_base_dict = {'fcs': layers.fcs}
st_head_dict = {'fcs': layers.fcs}



#### model ######

def get_model(
    cfg,
    device=None,
    **kwargs,
    ):

  # shortcuts
  c_dim = cfg.get('c_dim', 512) # encoder output dim
  h_dim = cfg.get('h_dim', 512) # graph head input dim

  return_base_features = cfg.get('return_base_features', False)
  return_encoder_features = cfg.get('return_encoder_features', False)
  adaptor_path = cfg.get('adaptor_path', None)

  annot_metric = cfg.get('annot_metric', 'm')

  # projection setup
  off_z = cfg.get('off_z', 0.4) # camera distance from object in meter.
  scale_factor = cfg.get('scale_factor', 0.0001)
  trans_factor = cfg.get('trans_factor', 100.0)

  # graph
  center_idx = cfg.get('center_idx', 0)
  graph_block_num = cfg.get('graph_block_num',5)
  ds_factors= cfg.get('ds_factors',[2,2,2,2])
  reference_mesh_file = cfg.get('reference_mesh_file', 'misc/graph/template_right_meter.obj')
  downsample_file = cfg.get('downsample_file', 'misc/graph/downsample_{}.pkl')
  mean_file = cfg.get('mean_file', '?_meter.npy')
  std_file = cfg.get('std_file', '?_meter.npy')

  # arch
  encoder = cfg.get('encoder', 'resnet18')
  encoder_kwargs = cfg.get('encoder_kwargs', {})

  graph_base = cfg.get('graph_base', 'fcs')
  graph_base_kwargs = cfg.get('graph_base_kwargs', {
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


  encoder = encoder_dict[encoder](
      c_dim=c_dim,
      **encoder_kwargs).to(device) if encoder in encoder_dict else None

  graph_base = graph_base_dict[graph_base](
      **graph_base_kwargs
      ).to(device) if graph_base in graph_base_dict else None

  st_head = st_head_dict[st_head](**st_head_kwargs)
  obj_st_head = st_head_dict[obj_st_head](**obj_st_head_kwargs) if obj_st_head is not None else None

  graph_head = GraphHead(
      h_dim=h_dim,
      block_num=graph_block_num,
      ds_factors=ds_factors,
      reference_mesh_file=reference_mesh_file,
      downsample_file=downsample_file,
      mean_file=mean_file,
      std_file=std_file,
      device=device,
      )

  ### create model ###
  model = Graph(
      encoder=encoder,
      graph_base=graph_base,
      graph_head=graph_head,
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

  ### load pretrained ###
  pretrained_file = cfg.get('pretained_file', None)
  if pretrained_file:
    utils.load_checkpoint(pretrained_file, model)
    print(f'pretrained weight from {pretrained_file} is loaded.')

  ### freeze modules ###
  for module, part_name in [(model.module.encoder,'encoder'),
                            (model.module.graph_base,'graph_base'),
                            (model.module.graph_head,'graph_head'),
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
    'verts3d':(0.5,'l2'),
    })

  trainer = GraphTrainer(
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

