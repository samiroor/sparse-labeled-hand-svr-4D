import os

import torch
from torch.nn.utils import clip_grad_norm_


import utils
from trainer import BaseTrainer
from data import Fields as Fls


from methods.modelutils import losses as LOSS


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
    assert len(self.loss_cfg.loss_types) > 0, "Expected some loss types!"

    self.adjust_bn_params = adjust_bn_params
    self.grad_norm_clip = grad_norm_clip
    self.grad_norm_clip_value = grad_norm_clip_value

    # just to debug
    self.verts2joints_fn = kwargs.get('verts2joints_fn', None)
    self.mano_fn = kwargs.get('mano_fn', None)


  def train_step(
      self,
      batch_samples,
      vis_dir='',
      display=False,
      calc_metrics=False,
      ):
    # FOR DEBUG: torch.autograd.set_detect_anomaly(True)
    self.model.train()
    device = self.device

    for k in batch_samples:
      batch_samples[k] = batch_samples[k].float().to(device)

    # adjust batch norm
    if self.adjust_bn_params:
      utils.adjust_bn_params(self.model, 'encoder.')

    self.optimizer.zero_grad()
    output_batch = self.model(batch_samples)

    loss, loss_dict = self.compute_loss(
        batch_samples,
        output_batch,
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
          self.metric_step(batch_samples, output_batch)
          )

    if display:
      self.visualize_step(batch_samples, output_batch, vis_dir=vis_dir)

    return train_dict


  def eval_step(
      self,
      batch_samples,
      vis_dir='',
      display=False,
      calc_metrics=False,
      return_results=False,
      ):

    self.model.eval()
    device = self.device

    for k in batch_samples:
      batch_samples[k] = batch_samples[k].float().to(device)

    eval_dict = {}
    with torch.no_grad():

      output_batch = self.model(batch_samples)

      loss, loss_dict = self.compute_loss(
          batch_samples,
          output_batch,
          loss_types=self.loss_cfg.loss_types,
          )

      eval_dict = loss_dict
      eval_dict['loss'] = loss.item()

    if calc_metrics:
      eval_dict.update(
          self.metric_step(batch_samples, output_batch)
        )

    if display:
      self.visualize_step(batch_samples, output_batch, vis_dir=vis_dir)

    if return_results:
      return eval_dict, output_batch

    return eval_dict


  def compute_loss(
      self,
      batch_samples,
      output_batch,
      loss_types=[
        'mesh_predictor_s',
        'mesh_predictor_tfeq_u'],
      ):

    loss_list = list()
    select_fn = lambda dict_, mask : {k : v[mask] for k,v in dict_.items()}

    mask_s = batch_samples[Fls.supervised].bool()

    # mesh predictor loss on Supervised items
    if 'mesh_predictor_s' in loss_types:
      assert bool(self.loss_cfg.mesh_predictor), 'Expected mesh_predictor loss config'
      if any(mask_s):
        loss_list.append(
            [
              'mesh_predictor_s_',
              select_fn(batch_samples, mask_s),
              select_fn(output_batch, mask_s),
              self.loss_cfg.mesh_predictor,
              ]
            )

    # mesh predictor tfeq (transform equivariance) loss on Unsupervised items
    if 'mesh_predictor_tfeq_u' in loss_types:
      assert bool(self.loss_cfg.mesh_predictor_tfeq), 'Expected mesh_predictor_tfeq loss config'
      if any(~mask_s):
        x = select_fn(batch_samples, ~mask_s)
        fx = select_fn(output_batch, ~mask_s)

        batch_size, _, h, w = x[Fls.images].shape
        TFEQ = LOSS.TfEq(
            batch_size=batch_size,
            input_hw=(h,w),
            device=self.device,
            )

        TFEQ.set_transform_mats()

        # ftx
        tx = TFEQ.transform(x)
        ftx = self.model(tx)

        #tfx
        tfx = TFEQ.transform(fx)

        loss_list.append(
            [
              'mesh_predictor_tfeq_u',
              ftx,
              tfx,
              self.loss_cfg.mesh_predictor_tfeq,
              ]
            )

    ### compute loss ###

    loss = torch.tensor(0.).to(self.device)
    loss_dict = dict()

    for prefix, gt, prd, cfg in loss_list:
      _loss, _loss_dict = self._compute_loss(
          gt,
          prd,
          cfg,
          )
      loss += _loss
      loss_dict.update(
          {f'loss_{prefix}' + k:v for k, v in _loss_dict.items()}
          )

    return loss, loss_dict


  def metric_step(
      self,
      batch_samples,
      output_batch,
      ):

    metric_list = list()
    select_fn = lambda dict_, mask : {k : v[mask] for k,v in dict_.items()}

    mask_s = batch_samples[Fls.supervised].bool()

    if any(mask_s):
      metric_list.append(
          [
            'mesh_predictor_s',
            select_fn(batch_samples, mask_s),
            select_fn(output_batch, mask_s),
            ]
          )
    if any(~mask_s):
      metric_list.append(
          [
            'mesh_predictor_u',
            select_fn(batch_samples, ~mask_s),
            select_fn(output_batch, ~mask_s),
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
          {f'{prefix}' + k:v for k, v in _metric_dict.items()}
          )

    return metric_dict


  def visualize_step(
      self,
      batch_samples,
      output_batch=None,
      display_render=True,
      display_skeleton=True,
      vis_dir='',
      ):

    if output_batch is None:
      _ = self.eval_step(
          batch_samples,
          vis_dir=vis_dir,
          display=True,
          )
      return 0

    os.makedirs(vis_dir, exist_ok=True)

    return self._visualize_step(
        batch_samples,
        output_batch,
        multiFrame=False,
        display_render=display_render,
        display_skeleton=display_skeleton,
        vis_dir=vis_dir,
        )

