import numpy as np
import torch

from . import collate
from .standard_fields import Fields
from .datautil import denormalize, quat2rotvec, rotvec2quat, torch_denormalize

from .datasets.ho3d.ho3d import HO3D
from .datasets.fphab.fphab import FPHAB
from .datasets.interhand26m.interhand import InterHand
from .datasets.interhand26m.mini_interhand_v1 import MiniInterHand_v1


from .singlehand_img_dataset import ShImgDataset
from .supervised_singlehand_img_dataset import SupervisedShImgDataset

from .singlehand_seq_dataset import ShSeqDataset
from .flow_singlehand_seq_dataset import FlowShSeqDataset


data_dict = {
    'ho3d':HO3D,
    'fphab':FPHAB,
    'interhand':InterHand,
    'mini_interhand_v1':MiniInterHand_v1,
    }

dataset_dict = {
    "singlehand_img":ShImgDataset,
    "supervised_singlehand_img":SupervisedShImgDataset,
    "singlehand_seq":ShSeqDataset,
    "flow_singlehand_seq":FlowShSeqDataset,
    }


def worker_init_fn(worker_id):
  np.random.seed(np.random.get_state()[1][0] + worker_id)


def select_data(
    cfg,
    mano_root='misc/mano_v1_2/models',
    ):
  """
  Read a specific video dataset.
  Args:
    cfg (EasyDict)
    mano_root (str)
  """
  # subset of data
  split = cfg.split
  use_mini_dataset = cfg.get("use_mini_dataset", False)
  mini_dataset = cfg.get('mini_dataset', {})
  if use_mini_dataset:
    if not bool(mini_dataset):
      subset_file = f"misc/use_mini_dataset/{cfg.name}_minimal.json"
      assert os.path.exists(subset_file), f'Not found "{data_file}".'
      with open(subset_file, 'r') as f:
        subset = EasyDict(json.load(f))
        assert split in subset, f"Expected {split} in {subset_file}."
        cfg.mini_dataset = EasyDict({split: subset[split]})

  # image features
  images_features = cfg.get("images_features", None)
  if images_features:
    feature_file = (images_features).format(split=split)
    assert os.path.exists(feature_file), f'Not found "{feature_file}".'
    with open(feature_file, 'rb') as f:
      images_features = pickle.load(f)

  # read_data
  if cfg.name in data_dict:
    dataset = data_dict[cfg.name](
        cfg,
        split=split,
        mano_root=mano_root,
        images_features=images_features)
  else:
    raise NotImplementedError(f'Dataset name {cfg.name} is not defined yet')
  return dataset


def get_dataset(
    cfgs,
    mano_root,
    training,
    do_augment,
    **kwargs):
  """
    Args :
      *cfgs (EasyDict) datasets_config
        - cfg (EasyDict) dataset_config
        - cfg (EasyDict) dataset_config
  """
  datasets = []
  for _ in cfgs:
    cfg = cfgs[_]
    if cfg.name in dataset_dict:
      dataset = dataset_dict[cfg.name](
          select_data(
            cfg.data,
            mano_root=mano_root,
            ),
          do_augment=do_augment,
          training=training,
          **cfg.data,
          **kwargs)
    else:
      raise NotImplementedError(f"{cfg.name}")
    datasets.append(dataset)

  if len(datasets) > 1:
    return torch.utils.data.ConcatDataset(datasets)
  else:
    return datasets[0]


def get_dataloaders(
    cfg,
    mano_root='misc/mano_v1_2/models',
    shuffle=False,
    training=False,
    do_augment=False,
    **kwargs):
  """
    Args :
      cfg (EasyDict) : dataloader_config
        - batch_size (int)
        - workers (int)
        - datasets (EasyDict)
          * name
          * ...
  """

  dataset = get_dataset(cfg.datasets,
                        mano_root=mano_root,
                        training=training,
                        do_augment=do_augment,
                        **kwargs)
  dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.workers,
                                           collate_fn=collate.meshreg_collate,
                                           worker_init_fn=worker_init_fn,
                                           shuffle=shuffle,
                                           drop_last=False,
                                           pin_memory=True)
  return dataloader
