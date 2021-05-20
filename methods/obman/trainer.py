import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import distributions as dist

import utils
from trainer import BaseTrainer
from visualize import visualizer
from data import Fields as Fls
from data import torch_denormalize as denormalize


class ObManTrainer(BaseTrainer):
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

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_fns = metrics_fns
        self.device = device
        self.exp_dir = exp_dir
        self.loss_cfg = loss_cfg

        self.adjust_bn_params = adjust_bn_params
        self.grad_norm_clip = grad_norm_clip
        self.grad_norm_clip_value = grad_norm_clip_value

        # used for metrics & visualization
        self.verts2joints_fn = kwargs.get('verts2joints_fn', None)
        self.mano_fn = kwargs.get('mano_fn', None)


    def train_step(
        self,
        batch_samples,
        calc_metrics=False,
        have_visualize=False,
        vis_dir='',
        ):
        ''' Performs a train step.

        Args:
            batch_samples (Easydict): data dictionary
        '''

        self.model.train()
        device = self.device

        for k in batch_samples:
          batch_samples[k] = batch_samples[k].float().to(device)

        # adjust batch norm
        if self.adjust_bn_params:
          utils.adjust_bn_params(self.model, 'encoder.')

        self.optimizer.zero_grad()
        output_batch = self.model(batch_samples)
        loss, loss_dict = self.compute_loss(batch_samples, output_batch)
        loss.backward()

        if self.grad_norm_clip:
          clip_grad_norm_(self.model.parameters(), self.grad_norm_clip_value)
        self.optimizer.step()

        train_dict = loss_dict
        train_dict['loss'] = loss.item()

        if calc_metrics:
          train_dict.update(
              self.metric_step(batch_samples, output_batch)
              )

        if have_visualize:
          self.visualize_step(batch_samples, output_batch, vis_dir=vis_dir)

        return train_dict


    def eval_step(
        self,
        batch_samples,
        calc_metrics=False,
        have_visualize=False,
        vis_dir='',
        return_results=False,
        ):

        ''' Performs an evaluation step.

        Args:
            batch_samples (Easydict): data dictionary
        '''
        self.model.eval()
        device = self.device

        for k in batch_samples:
          batch_samples[k] = batch_samples[k].float().to(device)

        eval_dict = {}

        with torch.no_grad():
          output_batch = self.model(batch_samples)
          loss, loss_dict = self.compute_loss(batch_samples, output_batch)
          eval_dict = loss_dict
          eval_dict['loss'] = loss.item()

        if calc_metrics:
          eval_dict.update(
              self.metric_step(batch_samples, output_batch)
            )

        if have_visualize:
          self.visualize_step(batch_samples, output_batch, vis_dir=vis_dir)

        if return_results:
            return eval_dict, output_batch

        return eval_dict


    def compute_loss(
        self,
        batch_samples,
        output_batch,
        ):
        ''' Computes the loss.

        Args:
            batch_samples (Easydict): data dictionary
        '''

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
          if k.split('Fls.')[-1] in self.loss_cfg:
            lamda_fn = self.loss_cfg[k.split('Fls.')[-1]]
            lamda = lamda_fn[0]
            fn = fns_dict[lamda_fn[1]]

            loss_val = fn(
                batch_samples[k],
                output_batch[k],
                )

            loss +=  lamda * loss_val
            loss_dict[k] = loss_val.item()

        return loss, loss_dict


    def visualize_step(
        self,
        batch_samples,
        output_batch=None,
        have_render=True,
        vis_dir='',
        ):
      if output_batch is None:
        _ = self.eval_step(
            batch_samples,
            have_visualize=True,
            vis_dir=vis_dir
            )
        return 0

      os.makedirs(vis_dir, exist_ok=True)

      device = self.device
      for k in  batch_samples:
        batch_samples[k] = batch_samples[k].float().to(device)

      assert Fls.images in batch_samples, 'Expected images in batch'
      batch_samples[Fls.images] = denormalize(batch_samples[Fls.images]) #B,c,H,W
      batch_size = batch_samples[Fls.images].shape[0]

      # rendering mesh
      if Fls.verts3d in output_batch and have_render:
        with torch.no_grad():
          render_images = visualizer.render(
              verts_gt=batch_samples.get(Fls.verts3d,None),
              verts_prd=output_batch.get(Fls.verts3d,None),
              faces=batch_samples.get(Fls.faces, None),
              images=batch_samples.get(Fls.images, None),
              camsintr=batch_samples.get(Fls.camsintr, None),
            )
        visualizer.save_images(
            render_images.detach().cpu().numpy(),
            save_dir=os.path.join(vis_dir, 'tmp'),
            )
        visualizer.create_video(
            video_file=os.path.join(vis_dir, f'video.mp4'),
            img_folder=os.path.join(vis_dir,'tmp'),
            )

      # skeleton
      dt = lambda x : x.detach().cpu().numpy() if x is not None else None
      images = dt(batch_samples[Fls.images].permute(0,2,3,1)) #B,H,W,c
      joints2d_gt = dt(batch_samples.get(Fls.joints2d, None))
      joints3d_gt = dt(batch_samples.get(Fls.joints3d, None))
      joints2d_prd = dt(output_batch.get(Fls.joints2d, None))
      joints3d_prd = dt(output_batch.get(Fls.joints3d, None))

      for s_i in range(0, batch_size, batch_size//8):
        w,h = matplotlib.figure.figaspect(0.25)
        fig = plt.figure(figsize=(w,h))

        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(images[s_i])

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

        fig.savefig(os.path.join(vis_dir,f'sample_{s_i}.png'))
        plt.close(fig)

