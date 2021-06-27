import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch import distributions as dist

import utils
from visualize import visualizer
from trainer import BaseTrainer
from data import Fields as Fls

from methods.modelutils import losses as LOSS


class PairFlowTrainer(BaseTrainer):
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
    assert len(self.loss_cfg.loss_types) > 0, "Expected some loss types!"

    self.adjust_bn_params = adjust_bn_params
    self.grad_norm_clip = grad_norm_clip
    self.grad_norm_clip_value = grad_norm_clip_value

    # used for metrics & visualization
    self.verts2joints_fn = kwargs.get('verts2joints_fn', None)
    self.mano_fn = kwargs.get('mano_fn', None)


  def train_step(
      self,
      batch_samples,
      vis_dir='',
      display=False,
      calc_metrics=False,
      ):
    """
      Performs a train step.
          try to predict supervised via warping unsupervised predictions i.e
          loss signals are defined on supervised ones!
      Args:
          batch_samples tuple of (Easydict): data dictionary
          first item refers to supervised batch, and second item refers to unsupervised ones
    """

    self.model.train()
    device = self.device
    assert len(batch_samples)==2, "Expected supervised & unsupervised batches"

    supervised_batch = batch_samples[0]
    unsupervised_batch = batch_samples[1]

    for k in supervised_batch:
      supervised_batch[k] = supervised_batch[k].float().to(device)
      unsupervised_batch[k] = unsupervised_batch[k].float().to(device)

    # adjust batch norm
    if self.adjust_bn_params:
      utils.adjust_bn_params(self.model, 'encoder.')

    self.optimizer.zero_grad()

    output_batch = self.model(
        {
          'supervised':supervised_batch,
          'unsupervised':unsupervised_batch,
          },
        fwd=True,
        invert=bool(self.loss_cfg.flow_u),
      )

    loss, loss_dict = self.compute_loss(
        (supervised_batch, unsupervised_batch),
        output_batch,
        step_mode='train',
        loss_types=self.loss_cfg.loss_types,
        )

    loss.backward()

    if self.grad_norm_clip:
      clip_grad_norm_(self.model.parameters(), self.grad_norm_clip_value)
    self.optimizer.step()

    train_dict = loss_dict
    train_dict['loss'] = loss.item()

    if calc_metrics:
      train_dict.update(
          self.metric_step(
            (supervised_batch, unsupervised_batch),
            output_batch,
            )
          )

    if display:
      self.visualize_step(
          (supervised_batch, unsupervised_batch),
          output_batch,
          step_mode='train',
          vis_dir=vis_dir,
          )


    return train_dict


  def eval_step(
      self,
      batch_samples,
      vis_dir='',
      display=False,
      calc_metrics=False,
      return_results=False,
      ):

    """
      Performs an evaluation step.
          Try to infer Unsupervised predictions via warping GT supervised annotations!
      Args:
          batch_samples (Easydict): data dictionary
    """

    self.model.eval()
    device = self.device
    assert len(batch_samples)==2, "Expected supervised & unsupervised batches"

    supervised_batch = batch_samples[0]
    unsupervised_batch = batch_samples[1]

    for k in supervised_batch:
      supervised_batch[k] = supervised_batch[k].float().to(device)
      unsupervised_batch[k] = unsupervised_batch[k].float().to(device)

    eval_dict = {}
    with torch.no_grad():

      output_batch = self.model({
        'supervised':supervised_batch,
        'unsupervised':unsupervised_batch,
        },
        fwd=False,
        infer=True,
        )

      loss, loss_dict = self.compute_loss(
        (supervised_batch, unsupervised_batch),
        output_batch,
        step_mode='infer',
        loss_types=['flow_s'],
        )

      eval_dict = loss_dict
      eval_dict['loss'] = loss.item()


    if calc_metrics:
      eval_dict.update(
          self.metric_step(
            (supervised_batch, unsupervised_batch),
            output_batch,
            )
          )

    if display:
      self.visualize_step(
          (supervised_batch, unsupervised_batch),
          output_batch,
          vis_dir=vis_dir,
          step_mode='infer',
          )

    if return_results:
      return eval_dict, output_batch

    return eval_dict


  def compute_loss(
      self,
      batch_samples,
      output_batch,
      loss_types=list(),
      step_mode='train',
      ):
    """
        Performs computing loss corresponding to step mode & config.
        Args:
            *supervised_batch (Easydict): data dictionary
            *unsupervised_batch (Easydict)
            *output_batch (EasyDict)
            *loss_types (list of str): [
              'mesh_predictor_s',
              'mesh_predictor_u', #infer_step
              'flow_s',
              'flow_u',
              'flow_infer_u', #infer_step
              ]
              **'s' means loss on supervised images
              **'u' means loss on unsupervised images
    """
    supervised_batch = batch_samples[0]
    unsupervised_batch = batch_samples[1]

    loss_list = list()

    # mesh predictor loss on Supervised items
    if 'mesh_predictor_s' in loss_types:
      assert self.loss_cfg.get('mesh_predictor_s', False), 'Expected mesh_predictor_s loss config'
      if 'ordinary' in self.loss_cfg.mesh_predictor_s:
        loss_list.append(
            [
              'mesh_predictor_s',
              'ordinary',
              supervised_batch,
              output_batch['mesh_predictor']['supervised'],
              self.loss_cfg.mesh_predictor_s.ordinary,
              ]
            )

    # mesh predictor loss on Unsupervised items
    if 'mesh_predictor_u' in loss_types or 'flow_u' in loss_types:
      assert self.loss_cfg.get('mesh_predictor_u', False) or self.loss_cfg.get('flow_u', False), 'Expected mesh_predictor_u or flow_u loss config'

      if 'tfeq' in self.loss_cfg.get('mesh_predictor_u', {}) or 'tfeq' in self.loss_cfg.get('flow_u', {}):
        x_s = supervised_batch[Fls.images]
        x_u = unsupervised_batch[Fls.images]

        batch_size, _, h, w = x_s[Fls.images].shape
        TFEQ = LOSS.TfEq(
            batch_size=batch_size,
            input_hw=(h,w),
            device=self.device,
            color_augment=self.loss_cfg.get('tfeq_color_augment', False),
            transform_type=self.loss_cfg.get('tfeq_transform_type', 'rotation'),
            )

        TFEQ.set_transform_mats()

        # ftx_s & ftx_u
        tx_s = TFEQ.transform(x_s)
        tx_u = TFEQ.transform(x_u)

        ftx_output = self.model({
          'supervised': tx_s,
          'unsupervised':tx_u,
          },
          fwd=False,
          invert='tfeq' in self.loss_cfg.get('flow_u', {}),
          )

        if 'tfeq' in self.loss_cfg.get('mesh_predictor_u', {}):
          ftx_u = ftx_output['mesh_predictor']['unsupervised']
          tfx_u = TFEQ.transform(output_batch['mesh_predictor']['unsupervised'])
          loss_list.append(
              [
                'mesh_predictor_u',
                'tfeq',
                tfx_u,
                ftx_u,
                self.loss_cfg.mesh_predictor_u.tfeq,
                ]
              )

        if 'tfeq' in self.loss_cfg.get('flow_u', {}):
          ftx_u = ftx_output['flow']['unsupervised']
          tfx_u = TFEQ.transform(output_batch['flow']['unsupervised'])
          loss_list.append(
              [
                'flow_u',
                'tfeq',
                tfx_u,
                ftx_u,
                self.loss_cfg.flow_u.tfeq,
                ]
              )

    ### flow loss on supervised items ###
    if 'flow_s' in loss_types and step_mode!="infer":
      assert self.loss_cfg.get('flow_s', False), 'Expected flow_s loss config'
      if 'ordinary' in self.loss_cfg.flow_s:
        loss_list.append(
            [
              'flow_s',
              'ordinary',
              supervised_batch,
              output_batch['flow']['supervised'],
              self.loss_cfg.flow_s.ordinary,
            ]
          )


    if 'flow_u' in loss_types:
      assert self.loss_cfg.get('flow_u', False),'Expected flow_bkw loss config'
      if 'bkw' in self.loss_cfg.flow_u:
        loss_list.append(
            [
              'flow_u',
              'bkw',
              output_batch['mesh_predictor']['unsupervised'],
              output_batch['flow']['unsupervised'],
              self.loss_cfg.flow_u.bkw,
            ]
          )

    if 'flow_s' in loss_types and step_mode=='infer':
      assert self.loss_cfg.get('flow_s', False),'Expected flow loss config'
      loss_list.append(
          [
            'flow_u',
            'infer_ordinary',
            unsupervised_batch,
            output_batch['flow']['infer_unsupervised'],
            self.loss_cfg.flow_s.ordinary,
            ]
          )


    ### compute loss ###
    loss = torch.tensor(0.).to(self.device)
    loss_dict = dict()

    for prefix, name, gt, prd, cfg in loss_list:
      _loss, _loss_dict = self._compute_loss(gt, prd, cfg)
      loss += _loss
      loss_dict.update(
          {f'{prefix}__loss_{name}_' + k:v for k, v in _loss_dict.items()}
          )

    return loss, loss_dict


  def metric_step(
      self,
      batch_samples,
      output_batch=None,
      vis_dir='',
      ):

    supervised_batch = batch_samples[0]
    unsupervised_batch = batch_samples[1]

    metric_list = list()

    if 'supervised' in output_batch['mesh_predictor']:
      metric_list.append(
          [
            'mesh_predictor_s',
            supervised_batch,
            output_batch['mesh_predictor']['supervised'],
            ]
          )

    if 'unsupervised' in output_batch['mesh_predictor']:
      metric_list.append(
          [
            'mesh_predictor_u',
            unsupervised_batch,
            output_batch['mesh_predictor']['unsupervised'],
            ]
          )

    if 'supervised' in output_batch['flow']:
      metric_list.append(
          [
            'flow_s',
            supervised_batch,
            output_batch['flow']['supervised'],
            ]
          )

    if 'unsupervised' in output_batch['flow']:
      metric_list.append(
          [
            'flow_u',
            unsupervised_batch,
            output_batch['flow']['unsupervised'],
            ]
          )

    if 'infer_unsupervised' in output_batch['flow']:
      metric_list.append(
          [
            'flow_infer_u',
            unsupervised_batch,
            output_batch['flow']['infer_unsupervised'],
            ]
          )


    ### compute metrics ###

    metric_dict = dict()

    for prefix, gt, prd in metric_list:
      _metric_dict = self._metric_step(
          gt,
          prd
          )
      metric_dict.update(
          {f'{prefix}__' + k:v for k, v in _metric_dict.items()}
          )

    return metric_dict


  def visualize_step(
      self,
      batch_samples,
      output_batch=None,
      step_mode='infer',
      vis_dir='',
      save_ply=False,
      ):

    if output_batch is None:
      _ = self.eval_step(
            batch_samples,
            vis_dir=vis_dir,
            display=True,
            )
      return 0

    supervised_batch = batch_samples[0]
    unsupervised_batch = batch_samples[1]

    os.makedirs(vis_dir, exist_ok=True)

    # rendering & saving plys
    mesh_predictor_render_images_s = self._visualize_step(
          supervised_batch,
          output_batch['mesh_predictor']['supervised'],
          return_render_images=True,
          display_render=True,
          save_ply=save_ply,
          suffix='mesh_predictor_s_',
          vis_dir=vis_dir,
          )

    mesh_predictor_render_images_u = self._visualize_step(
        unsupervised_batch,
        output_batch['mesh_predictor']['unsupervised'],
        return_render_images=True,
        display_render=True,
        save_ply=save_ply,
        suffix='mesh_predictor_u_',
        vis_dir=vis_dir,
        )

    title='1st row mesh predictor out on Supervised\n \
           2nd row mesh predictor out on Unsupervised\n'

    if step_mode == 'infer':
      flow_render_images = self._visualize_step(
          unsupervised_batch,
          output_batch['flow']['infer_unsupervised'],
          return_render_images=True,
          display_render=True,
          save_ply=save_ply,
          suffix='flow_infe_u_',
          vis_dir=vis_dir,
          )

      title += '3rd row flow inference out on Unsupervised images i.e \n \
                (Warped GT Supervised meshes into Unsupervised ones based on image evidence!)'

    elif step_mode=="train":
      flow_render_images = self._visualize_step(
          supervised_batch,
          output_batch['flow']['supervised'],
          return_render_images=True,
          display_render=True,
          save_ply=save_ply,
          suffix='flow_s_',
          vis_dir=vis_dir,
          )

      title += '3rd row flow out on Supervised images i.e \n \
               (Warped Unsupervised meshes into Supervised ones)'

    render_images = torch.cat([
      mesh_predictor_render_images_s,
      mesh_predictor_render_images_u,
      flow_render_images,
      ], dim=1)


    # make video
    visualizer.save_images(
        render_images.detach().cpu().numpy(),
        title=title,
        save_dir=os.path.join(vis_dir, 'tmp'),
        figaspect=0.6,
        fontsize=6,
        )

    visualizer.create_video(
        video_file=os.path.join(vis_dir, f'video.mp4'),
        img_folder=os.path.join(vis_dir,'tmp'),
        )
