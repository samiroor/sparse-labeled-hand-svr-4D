import os
import pickle
from torch import nn

from methods.modelutils import encoders, layers

from methods.spiral.model.spiral import Spiral
from methods.spiral.model import spiralutils

from methods.obman import get_trainer



encoder_dict = {'resnet18': encoders.Resnet18}#, 'resnet50': encoders.Resnet50}
spiral_base_dict = {'fcs': layers.fcs}
st_head_dict = {'fcs': layers.fcs}



#### model ######

def get_model(
    cfg,
    device=None,
    **kwargsi,
    ):


  # shortcuts
  c_dim = cfg.get('c_dim', 512) # encoder output dim
  h_dim = cfg.get('h_dim', 512) # spiral head input dim

  scale_factor = cfg.get('scale_factor', 0.0001)
  trans_factor = cfg.get('trans_factor', 100.0)

  return_spiral_features = cfg.get('return_spiral_features', False)
  return_encoder_features = cfg.get('return_encoder_features', False)
  use_adaptor = cfg.get('use_adaptor', True)
  adaptor_path = cfg.get('adaptor_path', None)

  annot_metric = cfg.get('annot_metric', 'm')

  # arch
  encoder = cfg.get('encoder', 'resnet18')
  encoder_kwargs = cfg.get('encoder_kwargs', {})

  spiral_base = cfg.get('mano_base', 'fcs')
  spiral_base_kwargs = cfg.get('mano_base_kwargs', {
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


  downsample_file = cfg.get('downsample_file','misc/spiral/?.pkl')
  template_file = cfg.get('template_mesh','misc/spiral/?.obj')
  center_idx = cfg.get('center_idx', 0)
  ds_factors = cfg.get('ds_factors', [2,2,2,2])
  spiral_length = cfg.get('spiral_length', [8, 8, 8, 8])
  spiral_dilation = cfg.get('spiral_dilation', [1, 1, 1, 1])
  spiral_filters = cfg.get('spiral_filters',[32, 32, 16,16])

  # get component
  encoder = encoder_dict[encoder](
      c_dim=c_dim,
      **encoder_kwargs).to(device) if encoder in encoder_dict else None

  spiral_base = spiral_base_dict[spiral_base](
      **spiral_base_kwargs
      ).to(device) if spiral_base in spiral_base_dict else None

  R_head = nn.Linear(h_dim, 3)

  st_head = st_head_dict[st_head](**st_head_kwargs)
  obj_st_head = st_head_dict[obj_st_head](**obj_st_head_kwargs) if obj_st_head is not None else None


  # downsample keys: 'vertices','face', 'adj', 'down_transform', 'up_transform'
  if not os.path.exists(downsample_file):
    print('Generating transform matrices for mesh decimations ...')
    assert os.path.exists(template_file)
    downsample_dict = spiralutils.decimate_mesh(
        template_file,
        ds_factors,
        )
    with open(downsample_file, 'wb') as fp:
      pickle.dump(downsample_dict, fp)
  else:
    with open(downsample_file, 'rb') as fp:
      downsample_dict = pickle.load(fp, encoding='latin1')

  # get indices to create a spiral for each vertex, outs (Nxspiral_len)
  spiral_indices_list = [
      spiralutils.preprocess_spiral(
        downsample_dict['face'][idx],
        spiral_length[idx],
        downsample_dict['vertices'][idx],
        spiral_dilation[idx]).to(device)
      for idx in range(len(downsample_dict['face']) - 1)
      ]

  down_transform_list = [
      spiralutils.to_sparse(down_transform).to(device)
      for down_transform in downsample_dict['down_transform']
      ]

  up_transform_list = [
    spiralutils.to_sparse(up_transform).to(device)
    for up_transform in downsample_dict['up_transform']
    ]

  # get spiral decoder
  AE = spiralutils.AE(
      in_channels=3,
      out_channels=spiral_filters,
      latent_channels=h_dim,
      spiral_indices=spiral_indices_list,
      down_transform=down_transform_list,
      up_transform=up_transform_list,
      ).to(device) #TODO omit earlier device!
  spiral_decoder = AE.decoder

  # create model
  model = Spiral(
      encoder=encoder,
      spiral_base=spiral_base,
      spiral_decoder=spiral_decoder,
      R_head=R_head,
      st_head=st_head,
      obj_st_head=obj_st_head,
      scale_factor=scale_factor,
      trans_factor=trans_factor,
      return_spiral_features=return_spiral_features,
      return_encoder_features=return_encoder_features,
      use_adaptor=use_adaptor,
      annot_metric=annot_metric,
      )
  model = nn.DataParallel(model).to(device) #check works with multi gpu??
  return model

