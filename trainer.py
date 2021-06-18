import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch

import utils
from visualize import visualizer
from data import Fields as Fls
from data import torch_denormalize as denormalize



class BaseTrainer(object):
  """
    Base trainer class.
  """

  def train_step(self, *args, **kwargs):
    """
    Performs a training step.
    """
    raise NotImplementedError


  def eval_step(self, *args, **kwargs):
    """
    Performs an evaluation step.
    """
    raise NotImplementedError


  def visualize_step(self, *args, **kwargs):
    """
    Performs  visualization.
    """
    raise NotImplementedError

  def metric_step(self, *args, **kwargs):
    """
    Performs  metric calculation.
    """
    raise NotImplementedError


  def restore_model(
      self,
      restore_file,
      ):
    """
    Restoring model.
    """
    restore_path = os.path.join(self.exp_dir, restore_file)
    assert os.path.exists(restore_path)

    checkpoint = utils.load_checkpoint(
        restore_path,
        self.model,
        self.optimizer,
        self.scheduler,
        allow_partial=False,
        )

    epoch = checkpoint['epoch']
    return epoch


  def save_model(
      self,
      epoch,
      save_file='last.pt',
      ):
    """
    Saving model
    """
    save_path = os.path.join(self.exp_dir, save_file)
    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': self.model.state_dict(),
      'optim_dict' : self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
      },
      save_path,
      )


  def _reshape_step(
      self,
      gt_dict,
      pred_dict,
      multiFrame=False,
      multiIter=False,
      ):
    """
    Reshape dictionaries to have batch-wise elemnts for metrics & visualization step
    """
    gt_dict = gt_dict.copy()
    pred_dict = pred_dict.copy()

    if multiFrame:
      for _dict in (gt_dict, pred_dict):
        for k,v in _dict.items():
          shape_ = v.shape
          _dict[k] = v.reshape(-1,*shape_[2:])

    # pred_dict might be multi-iteration (multiTime) and gt_dict not
    # in this case we just consider of the last iteration outputs.
    if multiIter:
      for k,v in pred_dict.items():
        if k in gt_dict:
          if v.shape[0] != gt_dict[k].shape[0]:
            pred_dict[k] = pred_dict[k][-1]
          else:
            break

    return gt_dict, pred_dict


  def _compute_loss(
      self,
      gt_dict,
      pred_dict,
      loss_cfg,
      multiFrame=False,
      multiIter=False,
      gamma_t=0.4,
      ):

    fns_dict = {
        'l2': torch.nn.functional.mse_loss,
        'l1': lambda x1,x2: torch.abs(x1-x2).mean(),
        'penalty': lambda _,x2: torch.nn.functional.mse_loss(x2, torch.zeros_like(x2))
        }

    _keys =[
        Fls.joints3d,
        Fls.joints3d_rel,
        Fls.joints2d,
        Fls.verts3d,
        Fls.verts3d_rel,
        Fls.verts2d,
        Fls.mano_poses,
        Fls.mano_shapes,
        Fls.objverts3d,
        ]

    loss_dict = {}
    device = self.device
    loss = torch.tensor(0.).to(device)

    for k in _keys:
      if k.split('Fls.')[-1] in loss_cfg:
        lamda_fn = loss_cfg[k.split('Fls.')[-1]]
        lamda = lamda_fn[0]
        fn = fns_dict[lamda_fn[1]]

        loss_val = fn(
            gt_dict[k],
            pred_dict[k],
            )

        loss +=  lamda * loss_val
        loss_dict[k] = loss_val.item()

    return loss, loss_dict


  def _metric_step(
      self,
      gt_dict,
      pred_dict,
      multiFrame=False,
      multiIter=False,
      ):
    """
    Metric preliminary step
    """

    gt_dict, pred_dict = self._reshape_step(
        gt_dict,
        pred_dict,
        multiFrame,
        multiIter,
        )

    metrics_dict = {}
    for metric_fn in self.metrics_fns:
      metrics_dict.update(
          metric_fn(
            gt_dict=gt_dict,
            pred_dict=pred_dict,
            ))
    return metrics_dict


  def _visualize_step(
      self,
      gt_dict,
      pred_dict,
      multiFrame=False,
      multiIter=False,
      display_render=False,
      display_skeleton=False,
      return_render_images=False,
      save_ply=False,
      suffix='',
      vis_dir='figs',
      ):
    """
     Visualize preliminary step
    """

    gt_dict, pred_dict = self._reshape_step(
        gt_dict,
        pred_dict,
        multiFrame,
        multiIter,
        )


    if Fls.images in gt_dict: gt_dict[Fls.images] = denormalize(gt_dict[Fls.images].clone()) #B,c,H,W

    #### rendering mesh ####
    if (Fls.verts3d in pred_dict or Fls.verts3d in gt_dict) and display_render:
      with torch.no_grad():
        render_images = visualizer.render(
            verts_gt=gt_dict.get(Fls.verts3d,None),
            verts_prd=pred_dict.get(Fls.verts3d,None),
            faces=gt_dict.get(Fls.faces, None),
            images=gt_dict.get(Fls.images, None),
            camsintr=gt_dict.get(Fls.camsintr, None),
            )

      if save_ply:
        faces = gt_dict[Fls.faces]
        for dict_, name in zip([gt_dict, pred_dict], ['-GT-','-PRD-']):
          if Fls.verts3d in dict_:
            batch_size = dict_[Fls.verts3d].shape[0]
            for i in range(0, batch_size, batch_size//8):
              visualizer.save_mesh_ply(
                  dict_[Fls.verts3d][i].detach().cpu().numpy(),
                  faces[0].detach().cpu().numpy(),
                  name=os.path.join(vis_dir,f'{suffix}{name}{i}.ply'),
                  )

      if return_render_images:
        return render_images #should be torch.tensor
      else:
        visualizer.save_images(
            render_images.detach().cpu().numpy(),
            save_dir=os.path.join(vis_dir, 'tmp'),
            )
        visualizer.create_video(
            video_file=os.path.join(vis_dir, f'video.mp4'),
            img_folder=os.path.join(vis_dir,'tmp'),
            )

      #### skeleton ####
      if display_skeleton:
        dt = lambda x : x.detach().cpu().numpy() if x is not None else None
        images = dt(gt_dict[Fls.images].permute(0,2,3,1)) if Fls.images in gt_dict else None #B,H,W,c

        joints2d_gt = dt(gt_dict.get(Fls.joints2d, None))
        joints3d_gt = dt(gt_dict.get(Fls.joints3d, None))
        joints2d_prd = dt(pred_dict.get(Fls.joints2d, None))
        joints3d_prd = dt(pred_dict.get(Fls.joints3d, None))

        batch_size = joints2d_gt.shape[0]

        for s_i in range(0, batch_size, batch_size//8):
          w,h = matplotlib.figure.figaspect(0.25)
          fig = plt.figure(figsize=(w,h))

          ax = fig.add_subplot(1, 4, 1)
          if images is not None:ax.imshow(images[s_i])

          if joints2d_gt is not None :
            ax = fig.add_subplot(1, 4, 2)
            ax.imshow(images[s_i])
            visualizer.skeleton2D(ax, joints2d_gt[s_i],title='2D-GT')

          if joints2d_prd is not None:
            ax = fig.add_subplot(1, 4, 3)
            ax.imshow(images[s_i])
            visualizer.skeleton2D(ax, joints2d_prd[s_i],title='2D-PRD')

          if joints3d_prd is not None:
            ax = fig.add_subplot(1, 4, 4, projection='3d')
            visualizer.skeleton3D(ax, joints3d_gt[s_i], color=(0,0,1), title='3D')
            visualizer.skeleton3D(ax, joints3d_prd[s_i], color=(1,0,0), title='3D')

          fig.savefig(os.path.join(vis_dir,f'{suffix}_sample_{s_i}.png'))
          plt.close(fig)
