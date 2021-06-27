import os
import time
import pickle
import open3d
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import commentjson as json
from easydict import EasyDict
from collections import defaultdict
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import utils
import builder


parser = argparse.ArgumentParser(
    description='Training a 4D SVR model (Single View Reconstruction).'
)

parser.add_argument('--exp_dir', default='exp_dir/config.json', help="Experiment Directory & config")
parser.add_argument('--restore_file', default=None, help="Weights to reload before training")
parser.add_argument('--allow_partial', dest='allow_partial', action='store_true')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--val_only', dest='val_only', action='store_false')
parser.add_argument('--test_only', dest='test_only', action='store_true')

parser.add_argument('--mesh_predictor_only', dest='mesh_predictor_only', action='store_true')



### evaluate Model ###
def evaluate(
    trainer,
    val_loader,
    vis_data,
    epoch,
    calc_metrics=False,
    display=False,
    save=False,
    vis_dir=None,
    save_dir=None,
    ):

  summary = []
  dataloader = iter(val_loader)

  # use tqdm for progress bar
  with tqdm(total=len(dataloader)) as t:
    for i in range(len(dataloader)):

      batch_samples = next(dataloader)

      eval_metrics = trainer.eval_step(
          batch_samples,
          vis_dir=vis_dir,
          display=display and i==0,
          calc_metrics=calc_metrics,
          )

      summary.append(eval_metrics)

      # update tqdm postfix
      t.set_postfix(Epoch=epoch+1, Iter=i+1, loss=eval_metrics['loss'])
      t.update()

  # metrics mean
  metrics_mean = trainer._metric_mean(summary)
  logging.info(
      "- Eval metrics:\n" + " :: ".join(F"{k}: {v:03.5f}" for k, v in metrics_mean.items())
      )

  if display:
    trainer.visualize_step(
        vis_data,
        vis_dir=os.path.join(vis_dir, 'vis_data'),
        )

  if save:
    with open(os.path.join(save_dir, 'summary.pkl'), 'wb') as fp:
      pickle.dump(summary, fp)

  return metrics_mean


### evaluate Mesh predictor module ###
def evaluate_mesh_predictor(
    trainer,
    val_loader,
    vis_data,
    epoch,
    method_name='pairflow',
    display=False,
    save=False,
    vis_dir=None,
    save_dir=None,
    ):


  """
    Evaluate mesh predictor module of different networks (models)!
    data loader should pass individual samples.
  """

  summary = []
  dataloader = iter(val_loader)

  trainer.model.eval()
  device = trainer.model.device

  if model_name in ("pairflow", ):
    mesh_predictor = trainer.model.module.mesh_predictor
  else:
    mesh_predictor = trainer.model

  with tqdm(total=len(dataloader)) as t:
    for i in range(len(dataloader)):

      batch_samples = next(dataloader)

      for k in batch_samples:
        batch_samples[k] = batch_samples[k].float().to(device)

      output_batch =  mesh_predictor(batch_samples)

      eval_metrics = trainer._metric_step(
          gt_dict=batch_samples,
          pred_dict=output_batch,
          )

      summary.append(eval_metrics)

  metrics_mean = trainer._metric_mean(summary)

  if display:

    for k in vis_data:
      vis_data[k] = vis_data[k].float().to(device)
    output_batch = mesh_predictor(vis_data)

    trainer._visualize_step(
        vis_data,
        output_batch,
        multiFrame=False,
        display_render=True,
        display_skeleton=True,
        vis_dir=vis_dir,
        )

  if save:
    with open(os.path.join(save_dir, 'summary.pkl'), 'wb') as fp:
      pickle.dump(summary, fp)

  return metrics_mean



if __name__ == '__main__':

  # load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.exp_dir, 'config.json')
  assert os.path.isfile(json_path),\
      f'Not a json configuration file found at "{json_path}"!'

  cfg = utils.get_config(json_path)

  # save_dir & log file
  exp_dir = os.path.join(args.exp_dir, 'run')

  suffix = utils.get_spec_name(
      cfg[f'{"testing" if args.test_only else "evaluating"}'].dataloader
      )

  if args.mesh_predictor_only:
    suffix += '_meshPredictorOnly'

  save_dir = os.path.join(
      exp_dir,
      f'{"eval_on_" if args.val_only else "test_on_"}\
          _{args.restore_file[:-3]}_epoch{epoch}',
      suffix)

  os.makedirs(save_dir, exist_ok=True)

  logpath = os.path.join(save_dir, 'logFile.log')
  utils.set_logger(logpath)

  # set device
  is_cuda = (torch.cuda.is_available() and not args.no_cuda)
  device = torch.device("cuda" if is_cuda else "cpu")

  # set seed
  seed = cfg.seed
  torch.manual_seed(seed)
  if is_cuda: torch.cuda.manual_seed(seed)
  utils.fix_rng_seed(seed)

  ### data loaders ###
  if  args.val_only:
    vis_loader = builder.get_dataloaders(
        cfg['evaluating'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=True,
        )
    vis_data = next(iter(vis_loader))
    vis_loader = None

    val_loader = builder.get_dataloaders(
        cfg['evaluating'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=False,
        )


  if args.test_only:
    vis_loader = builder.get_dataloaders(
        cfg['testing'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=True,
        )
    vis_data = next(iter(vis_loader))
    vis_loader = None

    test_loader = builder.get_dataloaders(
        cfg['testing'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=False,
        )


  ### model ###
  model = builder.get_model(
      cfg['model'],
      device=device,
      )


  # optimizer
  optimizer = None
  # scheduler
  scheduler = None

  ### metrics_fns ###
  metrics_fns = builder.get_metrics_fns(cfg['metrics'])

  # metrics operations
  metrics_ops = cfg['metrics'].get('ops', {'loss':'min'}).copy()
  for k,v in metrics_ops.items():
    metrics_ops[k] = min if v=="min" else max


  ### trainer object ###
  side_fns = lambda x: {
      'verts2joints_fn': x.dataset._verts2joints_fn,
      'mano_fn': x.dataset._mano_fn
      }


  trainer = builder.get_trainer(
      cfg['model'],
      model,
      optimizer,
      scheduler,
      metrics_fns,
      device,
      exp_dir,
      mean_std_loader=mean_std_loader,
      **side_fns(val_loader if args.val_only else test_loader),
      )

  ######## validate or test ##########
  epoch = trainer.restore_model(
      args.restore_file,
      allow_partial=args.allow_partial,
      )

  # save eval config
  with open(os.path.join(save_dir, 'eval_opt.json'), 'w') as fp:
    json.dump(cfg,
              fp,
              indent=2,
              sort_keys=True)

  # run
  if args.mesh_predictor_only:
    mean_metrics = evaluate_mesh_predictor(
        trainer,
        val_loader if args.val_only else test_loader,
        vis_data,
        epoch=epoch,
        method_name=cfg['model'],
        display=True,
        save=False,
        save_dir=save_dir,
        vis_dir=os.path.join(save_dir,'figs'),
        )
  else:
    mean_metrics = evaluate(
        trainer,
        val_loader if args.val_only else test_loader,
        vis_data,
        epoch=epoch,
        calc_metrics=True,
        display=True,
        save=False,
        save_dir=save_dir,
        vis_dir=os.path.join(save_dir,'figs'),
        )

  # logging metrics
  logging.info(F'========== {"evaluating" if args.val_only else "testing" } =========')
  logging.info(F'========== @ "{args.restore_file}" ==========')
  logging.info("- metrics: " + " ; ".join(F"{k}: {v:05.3f}" for k, v in mean_metrics.items()))

  # save metrics
  utils.save_dict_to_json(
      mean_metrics,
      os.path.join(save_dir, "mean_metrics.json"),
      )

