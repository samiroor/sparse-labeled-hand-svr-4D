import os
import numpy as np
import commentjson as json
from easydict import EasyDict
import torch

from . import collate
from .standard_fields import Fields
from .datautil import denormalize, quat2rotvec, rotvec2quat, torch_denormalize, torch_normalize

from .datasets.ho3d.ho3d import HO3D
from .datasets.fphab.fphab import FPHAB
from .datasets.interhand26m.interhand import InterHand
from .datasets.interhand26m.mini_interhand_v1 import MiniInterHand_v1

from .batch_samplers import SUBatchSampler
from .singlehand_img_dataset import ShImgDataset
from .supervised_singlehand_img_dataset import SupervisedShImgDataset
from .unsupervised_singlehand_img_dataset import UnsupervisedShImgDataset
from .pair_supervised_unsupervised_singlehand_img_dataset import PairSupUnsupShImgDataset
from .pair_supervised_any_singlehand_img_dataset import PairSupAnyShImgDataset
from .pair_supervised_supervised_singlehand_img_dataset import PairSupSupShImgDataset




data_dict = {
    'ho3d':HO3D,
    'fphab':FPHAB,
    'interhand':InterHand,
    'mini_interhand_v1':MiniInterHand_v1,
    }


dataset_dict = {
    "singlehand_img":ShImgDataset,
    "supervised_singlehand_img":SupervisedShImgDataset,
    "unsupervised_singlehand_img":UnsupervisedShImgDataset,
    "pair_supervised_unsupervised_singlehand_img":PairSupUnsupShImgDataset,
    "pair_supervised_any_singlehand_img":PairSupAnyShImgDataset,
    "pair_supervised_supervised_singlehand_img":PairSupSupShImgDataset,
    }

batch_sampler_dict= {
    "supervised_unsupervised":SUBatchSampler,
    }


def worker_init_fn(worker_id):
  np.random.seed(np.random.get_state()[1][0] + worker_id)



def select_data(
    cfg,
    split='train',
    use_mini_dataset=False,
    mini_dataset_file='*.json',
    mini_dataset={},
    supervise_fraction=1.0,
    add_first_last_frame=True,
    images_features=None,
    mano_root='misc/mano_v1_2/models',
    **kwargs,
    ):

  """
  Read a specific video dataset.
  Args:
    cfg : data specification (Ho3D InterHand etc)
    mini_dataset_file (json file) to read mini_dataset dictionary from that.
    mini_dataset (dict) {'split': video_dir i.e 'parent_folder(train|test|val)/video__name'}.
      For instance you can have {'test': 'train/...'}
  """

  # subset of data
  if use_mini_dataset:
    if not bool(mini_dataset):
      assert os.path.exists(mini_dataset_file), f'Not found "{mini_dataset_file}".'
      with open(mini_dataset_file, 'r') as f:
        subset = EasyDict(json.load(f))
        assert split in subset, f"Expected {split} in {mini_dataset_file}."
        mini_dataset = EasyDict({split: subset[split]})

  # image features
  if images_features is not None:
    feature_file = (images_features).format(split=split)
    assert os.path.exists(feature_file), f'Not found "{feature_file}".'
    with open(feature_file, 'rb') as f:
      images_features = pickle.load(f)

  # read_data
  if cfg.name in data_dict:
    dataset = data_dict[cfg.name](
        cfg,
        split=split,
        use_mini_dataset=use_mini_dataset,
        mini_dataset=mini_dataset,
        supervise_fraction=supervise_fraction,
        add_first_last_frame=add_first_last_frame,
        mano_root=mano_root,
        images_features=images_features,
        **kwargs,
        )
  else:
    raise NotImplementedError(f'Dataset name {cfg.name} is not defined yet')
  return dataset


def get_dataset(
    cfgs,
    data_cfg,
    augmentation_cfg,
    mano_root,
    training,
    **kwargs,
    ):

  """
    Args :
      *cfgs (EasyDict) datasets_config
        - cfg (EasyDict) dataset_config
        - cfg (EasyDict) dataset_config
      *data_cfg (Easydict) data specification
      *augmentation_cfg (Easydict) data specification

  """

  datasets = []
  for dataset_name in cfgs:
    cfg = cfgs[dataset_name]
    if dataset_name in dataset_dict:
      dataset = dataset_dict[dataset_name](
          select_data(
            data_cfg,
            split=cfg.get('split','train'),
            use_mini_dataset=cfg.get('use_mini_dataset', False),
            mini_dataset_file=cfg.get('mini_dataset_file', '*.json'),
            mini_dataset=cfg.get('mini_dataset', {}),
            supervise_fraction=cfg.get('supervise_fraction', 1.0),
            add_first_last_frame=cfg.get('add_first_last_frame',True),
            images_features=cfg.get('images_features',None),
            mano_root=mano_root,
            ),
          labelset=cfg.get('labelset',{}),
          do_augment=cfg.get('do_augment', False),
          augmentation=augmentation_cfg,
          training=training,
          gap=cfg.get('gap',1),
          **kwargs,
          )
    else:
      raise NotImplementedError(f"{dataset_name}")
    datasets.append(dataset)

  if len(datasets) > 1:
    return torch.utils.data.ConcatDataset(datasets)
  else:
    return datasets[0]



def get_dataloaders(
    cfg,
    data_cfg,
    augmentation_cfg,
    mano_root='misc/mano_v1_2/models',
    training=False,
    shuffle=False,
    **kwargs,
    ):

  """
    Args :
      cfg (EasyDict) : dataloader_config
        - batch_size (int)
        - workers (int)
        - datasets (EasyDict)
          * name
          * ...
  """

  dataset = get_dataset(
      cfgs=cfg.get('datasets',{}),
      data_cfg=data_cfg,
      augmentation_cfg=augmentation_cfg,
      mano_root=mano_root,
      training=training,
      **kwargs,
      )
  if 'batch_sampler' in cfg:
    batch_sampler = batch_sampler_dict[cfg.batch_sampler](
        dataset.cumulative_sizes,
        **cfg.batch_sampler_kwargs,
        )
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=cfg.get('workers',4),
                                             collate_fn=collate.meshreg_collate,
                                             worker_init_fn=worker_init_fn,
                                             pin_memory=True,
                                             )
  else:
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.get('batch_size',8),
                                             num_workers=cfg.get('workers',4),
                                             collate_fn=collate.meshreg_collate,
                                             worker_init_fn=worker_init_fn,
                                             shuffle=shuffle,
                                             drop_last=False,
                                             pin_memory=True,
                                             )
  return dataloader

