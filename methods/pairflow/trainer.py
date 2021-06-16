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
        '''
        Performs a train step.
            try to predict supervised via warping unsupervised predictions i.e
            loss signals are defined on supervised ones!
        Args:
            batch_samples tuple of (Easydict): data dictionary
            first item refers to supervised batch, and second item refers to unsupervised ones
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
          },
          fwd=True,
          invert=bool(self.loss_cfg.flow_bkw) or bool(self.loss_cfg.flow_bkw_tfeq),
          )
        ## compute loss
        loss, loss_dict = self.compute_loss(
            supervised_batch,
            unsupervised_batch,
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
              {'metric_flow_s__'+k:v for k,v in self.metric_step(supervised_batch, output_batch['flow']['supervised']).items()}
              )
          train_dict.update(
              {'metric_mesh_predictor_s__'+k:v for k, v in self.metric_step(supervised_batch, output_batch['mesh_predictor']['supervised']).items()}
              )
          train_dict.update(
              {'metric_mesh_predictor_u__'+k:v for k, v in self.metric_step(unsupervised_batch, output_batch['mesh_predictor']['unsupervised']).items()}
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

        '''
        Performs an evaluation step.
            Try to infer Unsupervised predictions via warping GT supervised annotations!
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
            fwd=False,
            infer=True,
            )

          # compute loss
          loss, loss_dict = self.compute_loss(
            supervised_batch,
            unsupervised_batch,
            output_batch,
            step_mode='infer',
            loss_types=['flow_infer_u'],
            )

          eval_dict = loss_dict
          eval_dict['loss'] = loss.item()


        if calc_metrics:
          eval_dict.update(
              {'metric_flow_infer_u__'+k:v for k, v in self.metric_step(unsupervised_batch, output_batch['flow']['infer_unsupervised']).items()}
              )
          eval_dict.update(
              {'metric_mesh_predictor_u__'+k:v for k, v in self.metric_step(unsupervised_batch, output_batch['mesh_predictor']['unsupervised']).items()}
              )
          eval_dict.update(
              {'metric_mesh_predictor_s__'+k:v for k, v in self.metric_step(supervised_batch, output_batch['mesh_predictor']['supervised']).items()}
              )
          eval_dict["samples_number"]=eval_dict['metric_flow_infer_u__samples_number']

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
        supervised_batch,
        unsupervised_batch,
        output_batch,
        loss_types=list(),
        step_mode='train',
        ):
        '''
        Performs computing loss corresponding to step mode & config.
        Args:
            *supervised_batch (Easydict): data dictionary
            *unsupervised_batch (Easydict)
            *output_batch (EasyDict)
            *loss_types (list of str): [
              'mesh_predictor_s',
              'mesh_predictor_u', #infer_step
              'mesh_predictor_tfeq_u',
              'flow_s',
              'flow_bkw_u',
              'flow_bkw_tfeq_u',
              'flow_infer_u', #infer_step
              ]
              **'s' means loss on supervised images
              **'u' means loss on unsupervised images
        '''
        loss_list = list()

        ### on mesh predictor ###
        if 'mesh_predictor_s' in loss_types:
          assert bool(self.loss_cfg.mesh_predictor), 'Expected mesh_predictor loss config'
          loss_list.append([
            'mesh_predictor_s_',
            supervised_batch,
            output_batch['mesh_predictor']['supervised'],
            self.loss_cfg.mesh_predictor,
              ])

        if 'mesh_predictor_u' in loss_types:
          assert bool(self.loss_cfg.mesh_predictor), 'Expected mesh_predictor loss config'
          assert step_mode=='infer','Expected inference step to use unsupervised batch'
          loss_list.append([
            'mesh_predictor_u_',
            unsupervised_batch,
            output_batch['mesh_predictor']['unsupervised'],
            self.loss_cfg.mesh_predictor,
              ])

        ### transform equivariance ###
        if 'mesh_predictor_tfeq_u' in loss_types or 'flow_bkw_tfeq_u' in loss_types:
          batch_size,_,h,w=supervised_batch[Fls.images].shape

          tf_eq = LOSS.TfEq(
              batch_size=batch_size,
              input_hw=(h,w),
              device=self.device,
              )

          tf_eq.set_transform_mats()

          supervised_batch_tx = tf_eq.transform(supervised_batch)
          unsupervised_batch_tx = tf_eq.transform(unsupervised_batch)
          if False:#True:#DEBUG:
            debug_dir='debug-tfeq'
            os.makedirs(debug_dir, exist_ok=True)
            # rendering & saving plys
            self._visualize_step(
                supervised_batch_tx,
                supervised_batch_tx,
                return_render_images=False,
                display_render=True,
                display_skeleton=True,
                save_ply=False,
                suffix='debug_tfeq',
                vis_dir=debug_dir,
                )

          ftx_output = self.model({
            'supervised':supervised_batch_tx,
            'unsupervised':unsupervised_batch_tx,
            },
            fwd=False,
            invert='flow_bkw_tfeq_u' in loss_types,
            )

          if 'mesh_predictor_tfeq_u' in loss_types:
            assert bool(self.loss_cfg.mesh_predictor_tfeq),'Expected mesh_predictor_tfeq loss config'
            ftx = ftx_output['mesh_predictor']['unsupervised']
            tfx = tf_eq.transform(output_batch['mesh_predictor']['unsupervised'])

            loss_list.append([
              'mesh_predictor_tfeq_u_',
              ftx,
              tfx,
              self.loss_cfg.mesh_predictor_tfeq,
              ])

          if 'flow_bkw_tfeq_u' in loss_types:
            assert bool(self.loss_cfg.flow_bkw_tfeq),'Expected flow_tfeq loss config'
            ftx = ftx_output['flow']['unsupervised']
            tfx = tf_eq.transform(output_batch['flow']['unsupervised'])

            loss_list.append([
              'flow_bkw_tfeq_u_',
              ftx,
              tfx,
              self.loss_cfg.flow_bkw_tfeq,
              ])

        ### flow ###
        if 'flow_s' in loss_types:
          assert bool(self.loss_cfg.flow),'Expected flow loss config'
          loss_list.append([
            'flow_s_',
            supervised_batch,
            output_batch['flow']['supervised'],
            self.loss_cfg.flow,
            ])

        if 'flow_bkw_u' in loss_types:
          assert bool(self.loss_cfg.flow_bkw),'Expected flow_bkw loss config'
          loss_list.append([
            'flow_bkw_u_',
            output_batch['mesh_predictor']['unsupervised'],
            output_batch['flow']['unsupervised'],
            self.loss_cfg.flow_bkw,
            ])

        if 'flow_infer_u'in loss_types:
          assert bool(self.loss_cfg.flow),'Expected flow loss config'
          assert step_mode=='infer','Expected inference step to use unsupervised batch'
          loss_list.append([
            'flow_infer_u_',
            unsupervised_batch,
            output_batch['flow']['infer_unsupervised'],
            self.loss_cfg.flow,
            ])

        ### compute loss ###
        loss = torch.tensor(0.).to(self.device)
        loss_dict = dict()

        for prefix, gt, prd, cfg in loss_list:
          _loss, _loss_dict = self._compute_loss(gt, prd, cfg)
          loss += _loss
          loss_dict.update(
              {f'loss_{prefix}' + k:v for k, v in _loss_dict.items()}
              )
        return loss, loss_dict


    def _compute_loss(
        self,
        batch_samples,
        output_batch,
        loss_cfg,
        gamma_t=0.4,
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

            if batch_samples[k].shape == output_batch[k].shape:
              loss_val = fn(
                batch_samples[k],
                output_batch[k],
                )

            else:
              # multiTime output
              T = output_batch[k].shape[0]
              ws = torch.nn.functional.softmax(
                  output_batch[k].new_tensor(
                    [gamma_t ** (T-i) for i in range(T)]
                    ),
                  dim=-1
                  )

              loss_val = torch.stack(
                  [
                    fn(
                      batch_samples[k],
                      prd,
                      ) * w for prd,w in zip(output_batch[k], ws)
                    ],
                  dim=0,
                  ).sum(0)

            loss +=  lamda * loss_val
            loss_dict[k] = loss_val.item()

        return loss, loss_dict


    def visualize_step(
        self,
        batch_samples,
        output_batch=None,
        step_mode='infer',
        vis_dir='',
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
            save_ply=True,
            suffix='mesh_predictor_supervised_',
            vis_dir=vis_dir,
            )

      mesh_predictor_render_images_u = self._visualize_step(
          unsupervised_batch,
          output_batch['mesh_predictor']['unsupervised'],
          return_render_images=True,
          display_render=True,
          save_ply=True,
          suffix='mesh_predictor_unsupervised_',
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
            save_ply=True,
            suffix='flow_infer_unsupervised_',
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
            save_ply=True,
            suffix='flow_supervised_',
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
