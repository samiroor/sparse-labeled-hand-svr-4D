import os
import numpy as np
from tqdm import tqdm


import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import distributions as dist

from data import Fields as Fls
from methods.obman import ObManTrainer


class GraphTrainer(ObManTrainer):
  def __init__(
      self,
      model,
      optimizer,
      scheduler=None,
      metrics_fns=None,
      device=None,
      exp_dir=None,
      loss_cfg={},
      adjust_bn_params=True,
      grad_norm_clip=False,
      grad_norm_clip_value=10,
      **kwargs,
      ):

    super().__init__(
        model,
        optimizer,
        scheduler=scheduler,
        metrics_fns=metrics_fns,
        device=device,
        exp_dir=exp_dir,
        loss_cfg=loss_cfg,
        adjust_bn_params=adjust_bn_params,
        grad_norm_clip=grad_norm_clip,
        grad_norm_clip_value=grad_norm_clip_value,
        **kwargs,
        )

    self.std_file = self.model.module.graph_head.std_file
    self.mean_file = self.model.module.graph_head.mean_file

    if not os.path.exists(self.mean_file):
      assert kwargs['mean_std_loader'] is not None
      self.save_train_mesh_mean_std(kwargs['mean_std_loader'])
      self.model.module.graph_head._rebuild()


  def save_train_mesh_mean_std(
      self,
      mean_std_loader,
      ):

    dataloader = iter(mean_std_loader)
    verts3d_rel = list()

    print('gather train meshes!')

    # use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i in range(len(dataloader)):
        batch_samples = next(dataloader)
        verts3d_rel.append(batch_samples[Fls.verts3d_rel].numpy())
        # update tqdm postfix
        t.set_postfix(Iter=i+1)
        t.update()

    verts3d_rel = np.concatenate(verts3d_rel, axis=0)

    mean = np.mean(verts3d_rel, axis=0)
    std = np.std(verts3d_rel, axis=0)

    np.save(self.mean_file, mean)
    np.save(self.std_file, std)

    print(f'- {self.mean_file} saved')
    print(f'- {self.std_file} saved')
    print(f'- train samples number {verts3d_rel.shape[0]}')
