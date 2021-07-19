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
import evaluate_modules


parser = argparse.ArgumentParser(
    description='Training a 4D SVR model (Single View Reconstruction).'
)

parser.add_argument('--exp_dir', default='exp_dir/config.json', help="Experiment Directory & config")
parser.add_argument('--restore_file', default=None, help="Weights to reload before training")
parser.add_argument('--allow_partial', dest='allow_partial', action='store_true')

parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--val_only', dest='val_only', action='store_true')

parser.add_argument('--mesh_predictor', dest='mesh_predictor', action='store_true')
parser.add_argument('--mesh_predictor_sampler', default='unsupervised')
parser.add_argument('--flow_infer', dest='flow_infer', action='store_true')
parser.add_argument('--flow_infer_sampler', default="unsupervised_supervised")

parser.add_argument('--test_split', default='test')
parser.add_argument('--not_calc_error', dest='not_calc_error', action='store_true')
parser.add_argument('--not_calc_mesh_error', dest='not_calc_mesh_error', action='store_true')



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


### set save_dir ###
def get_save_dir(cfg, exp_dir, args):
  suffix = utils.get_spec_name(cfg.dataloader)
  if args.mesh_predictor:
    suffix += '_eval_mesh_predictor'
    suffix += f"_on_{args.mesh_predictor_sampler}_samples"


  elif args.flow_infer:
    suffix += '_eval_flow_infer'
    suffix += f"_on_{'unsupervised' if args.flow_infer_sampler == 'unsupervised_supervised' else ''}_samples"
  else:
    suffix += '_eval_default'

  save_dir = exp_dir + '_Setting_'+ suffix
  os.makedirs(save_dir, exist_ok=True)
  return save_dir



if __name__ == '__main__':

  # load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.exp_dir, 'config.json')
  assert os.path.isfile(json_path),\
      f'Not a json configuration file found at "{json_path}"!'

  cfg = utils.get_config(json_path)

  # set test_split
  if 'dataloader' in cfg['testing']:
    if 'datasets' in cfg['testing'].dataloader:
      for k in cfg['testing']['dataloader'].datasets:
        cfg['testing']['dataloader'].datasets[k]['split'] = args.test_split
        print(f"test_split is '{args.test_split}' in {k} dataset")

  # save_dir & log file (using cfg)
  exp_dir = os.path.join(args.exp_dir, 'run')
  save_dir = get_save_dir(
      cfg[f'{"evaluating" if args.val_only else "testing"}'],
      os.path.join(
        exp_dir,
        f'{"eval_on_" if args.val_only else "test_on_"}ckpt_{args.restore_file[:-3]}'),
      args)
  utils.set_logger(os.path.join(save_dir, 'logFile.log'))

  # set device
  is_cuda = (torch.cuda.is_available() and not args.no_cuda)
  device = torch.device("cuda" if is_cuda else "cpu")
  # save eval config
  with open(os.path.join(save_dir, 'opt.json'), 'w') as fp:
    json.dump(cfg,
              fp,
              indent=2,
              sort_keys=True)


  # set seed
  seed = cfg.seed
  torch.manual_seed(seed)
  if is_cuda: torch.cuda.manual_seed(seed)
  utils.fix_rng_seed(seed)

  ### data loaders ###
  if  args.val_only:
    logging.info("Reading cfg['evaluating'] ... !")
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
  else:
    logging.info("Reading cfg['testing'] ... !")
    vis_loader = builder.get_dataloaders(
        cfg['testing'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=True,
        )
    vis_loader = iter(vis_loader)
    vis_data = next(vis_loader)

    vis_loader = None

    val_loader = builder.get_dataloaders(
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
      mean_std_loader=None,
      **side_fns(val_loader),
      )

  ######## validate or test ##########
  epoch = trainer.restore_model(
      args.restore_file,
      allow_partial=args.allow_partial,
      )

  ####### evaluate modules OR Default evaluate ########
  if args.mesh_predictor or args.flow_infer:
    if args.mesh_predictor:
      module_name = "mesh_predictor"
      sampler_mode = args.mesh_predictor_sampler
      if cfg['model'].name in ('pairflow', 'pair_disentangle_flow', ):
        model = trainer.model.module.mesh_predictor
      elif cfg['model'].name in ('obman'):
        model = trainer.model
      else:
        raise NotImplementedError
    if args.flow_infer:
      module_name = "flow_infer"
      sampler_mode = args.flow_infer_sampler
      if cfg['model'].name in ("pairflow", 'pair_disentangle_flow'):
        model = trainer.model
      else:
        raise NotImplementedError

    evaluate_modules._evaluate(
        model=model,
        device=trainer.device,
        val_loader=val_loader,
        sampler_mode=sampler_mode,
        module_name=module_name,
        calc_error= not args.not_calc_error,
        calc_mesh_error= not args.not_calc_mesh_error,
        save_dir=save_dir,
        )

  else:

    mean_metrics = evaluate(
        trainer,
        val_loader,
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

