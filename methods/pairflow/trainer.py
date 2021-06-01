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
from data import Fields as Fls



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
        ''' Performs a train step.

        Args:
            batch_samples (Easydict): data dictionary
        '''

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

        output_batch = self.model({
          'supervised':supervised_batch,
          'unsupervised':unsupervised_batch,
          })

        # compute loss
        loss = torch.tensor(0.).to(device); loss_dict = dict()
        # flow loss
        if bool(self.loss_cfg.flow):
          loss_flow, loss_flow_dict = self._compute_loss(
              supervised_batch,
              output_batch['flow']['supervised'],
              self.loss_cfg.flow,
              )
          loss += loss_flow
          loss_dict.update({'flow_'+k:v for k, v in loss_flow_dict.items()})
        # mesh predictor loss
        if bool(self.loss_cfg.mesh_predictor):
          loss_predictor, loss_predictor_dict = self._compute_loss(
              supervised_batch,
              output_batch['mesh_predictor']['supervised'],
              self.loss_cfg.mesh_predictor,
              )
          loss += loss_predictor
          loss_dict.update({'mesh_predictor_'+k:v for k, v in loss_predictor_dict.items()})

        loss.backward()
        if self.grad_norm_clip:
          clip_grad_norm_(self.model.parameters(), self.grad_norm_clip_value)
        self.optimizer.step()

        train_dict = loss_dict
        train_dict['loss'] = loss.item()

        if calc_metrics:
          train_dict.update(
              self.metric_step(supervised_batch, output_batch['flow']['supervised'])
              )

        if display:
          self.visualize_step(supervised_batch, output_batch['flow']['supervised'], vis_dir=vis_dir)

        return train_dict


    def eval_step(
        self,
        batch_samples,
        vis_dir='',
        display=False,
        calc_metrics=False,
        return_results=False,
        ):

        ''' Performs an evaluation step.

        Args:
            batch_samples (Easydict): data dictionary
        '''
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
            infer=True,
            )

          # compute loss
          loss = torch.tensor(0.).to(device); loss_dict = dict()
          # flow loss
          if bool(self.loss_cfg.flow):
            loss_flow, loss_flow_dict = self._compute_loss(
                unsupervised_batch,
                output_batch['flow']['infer_unsupervised'],
                self.loss_cfg.flow,
                )
            loss += loss_flow
            loss_dict.update({'flow_infer__'+k:v for k, v in loss_flow_dict.items()})
          # mesh predictor
          if bool(self.loss_cfg.mesh_predictor):
            loss_predictor, loss_predictor_dict = self._compute_loss(
              unsupervised_batch,
              output_batch['mesh_predictor']['unsupervised'],
              self.loss_cfg.mesh_predictor,
              )
            loss += loss_predictor
            loss_dict.update({'mesh_predictor_unsupervised__'+k:v for k, v in loss_predictor_dict.items()})
          eval_dict = loss_dict
          eval_dict['loss'] = loss.item()


        if calc_metrics:
          eval_dict.update(
              self.metric_step(unsupervised_batch, output_batch['flow']['infer_unsupervised'])
            )

        if display:
          self.visualize_step(unsupervised_batch, output_batch['flow']['infer_unsupervised'], vis_dir=vis_dir)

        if return_results:
            return eval_dict, output_batch

        return eval_dict


    def _compute_loss(
        self,
        batch_samples,
        output_batch,
        loss_cfg,
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
          if k.split('Fls.')[-1] in loss_cfg:
            lamda_fn = loss_cfg[k.split('Fls.')[-1]]
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
