import os
import commentjson as json
import time
import open3d
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from collections import defaultdict
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from torch.optim import Adam, Adamax

import utils
import builder

parser = argparse.ArgumentParser(
    description='Training a 4D SVR model (Single View Reconstruction).'
)

parser.add_argument('--exp_dir', default='exp_dir/config.json', help="Experiment Directory & config")
parser.add_argument('--eval_only', dest='eval_only', action='store_true')
parser.add_argument('--test_only', dest='test_only', action='store_true')
parser.add_argument('--restore_file', default=None, help="Weights to reload before training")
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--augment', action='store_true', help='Training with augmentation')



def restore_metrics(exp_dir):
  logging.info(f"Restoring metrics from {exp_dir}")

  path = os.path.join(exp_dir, 'all_metrics_eval.json')
  if os.path.exists(path):
    with open(path, 'r') as f:
      all_eval_metrics = json.load(f)

  path = os.path.join(exp_dir, 'all_metrics_train.json')
  if os.path.exists(path):
    with open(path, 'r') as f:
      all_train_metrics = json.load(f)

  for metric in metrics_ops:
    path = os.path.join(exp_dir, F'metrics_eval_best_{metric}.json')
    if os.path.exists(path):
      with open(path, 'r') as f:
        post_metrics = json.load(f)
        best_metrics[metric] = post_metrics
        best_metrics_vals[metric] = post_metrics[metric]


def train(
    trainer,
    train_loader,
    epoch,
    ):
    """
      Args:
    """
    summary = []
    dataloader = iter(train_loader)

    # use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i in range(len(dataloader)):
        batch = next(dataloader)
        train_metrics = trainer.train_step(batch)

        summary.append(train_metrics)

        # update tqdm postfix
        t.set_postfix(Epoch=epoch+1, Iter=i+1, loss=train_metrics['loss'])
        t.update()

    # compute mean of all metrics
    metrics_mean = {
        metric:np.mean([x[metric] for x in summary]) for metric in summary[0]
        }
    logging.info("- Train metrics: \n" + "::".join(F"{k}: {v:05.3f}" for k, v in metrics_mean.items()))
    return metrics_mean


def evaluate(
    trainer,
    val_loader,
    vis_data,
    epoch,
    vis_dir=None):
    """
      Args:
    """
    summary = []
    dataloader = iter(val_loader)

    # use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i in range(len(dataloader)):
        batch = next(dataloader)
        eval_metrics = trainer.eval_step(batch)

        summary.append(eval_metrics)

        # update tqdm postfix
        t.set_postfix(Epoch=epoch+1, Iter=i+1, loss=eval_metrics['loss'])
        t.update()

    # compute mean of all metrics 
    metrics_num = np.sum([x['samples_number'] for x in summary])
    metrics_mean = {
        metric:np.sum([x[metric]*x['samples_number'] for x in summary]) / metrics_num for metric in summary[0]
        }
    metrics_mean.update({'samples':float(metrics_num)})
    metrics_mean.pop('samples_number')
    logging.info("- Eval metrics:\n" + " :: ".join(F"{k}: {v:05.3f}" for k, v in metrics_mean.items()))

    # visualize
    trainer.visualize(
        vis_data,
        vis_dir,
        name=f'epoch{epoch}')

    return metrics_mean


def train_evaluate(
    trainer,
    train_loader,
    val_loader,
    vis_data,
    metrics_ops,
    writer,
    num_epochs=100,
    eval_freq=2,
    backup_freq=50,
    restore_file=None,
    early_stopping=False,
    early_stopping_patience=0,
    ):
  """
  """

  exp_dir = trainer.exp_dir

  global all_train_metrics, all_eval_metrics, best_metrics_vals, best_metrics
  fn_to_val = {min: np.inf, max: -np.inf}
  all_train_metrics = defaultdict(list)
  all_eval_metrics = defaultdict(list)
  best_metrics_vals = {metric:fn_to_val[metrics_ops[metric]] for metric in metrics_ops}
  best_metrics = dict()
  epoch_start = 0

  if early_stopping:
    early_stopping = utils.EarlyStopping(
        patience=early_stopping_patience,
        delta=0)

  # restore weight
  if restore_file:
    restore_metrics(exp_dir)
    epoch_start = trainer.restore_model(restore_file)

  # run for epochs!
  for epoch in range(epoch_start, num_epochs):

    logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))

    # traininig
    train_metrics = train(
        trainer,
        train_loader,
        epoch)
    for metric in train_metrics:
      all_train_metrics[metric].append(train_metrics[metric])

    # lr schedular & writer
    if trainer.scheduler is not None:
      trainer.scheduler.step()
      writer.add_scalar('learning_rate', trainer.scheduler.get_lr()[0], epoch)
    for k,v in train_metrics.items():
      writer.add_scalar('train_'+k, v, epoch)

    # save model 
    trainer.save_model(epoch, save_file='last.pt')
    if (epoch+1) % backup_freq == 0:
      trainer.save_model(epoch, save_file=f'e{epoch+1}.pt')

    # evaluate 
    if (epoch+1) % eval_freq == 0:
      eval_metrics = evaluate(
          trainer,
          val_loader,
          vis_data,
          epoch)

      utils.save_dict_to_json(
          eval_metrics,
          os.path.join(exp_dir, "metrics_eval_last_weights.json"))

      for metric in eval_metrics:
        all_eval_metrics[metric].append(eval_metrics[metric])

      # writer
      for k,v in eval_metrics.items():
        writer.add_scalar('val_'+k, v, epoch)

      # is best ?
      for metric in metrics_ops:
        if metric not in eval_metrics:
          continue
        if metrics_ops[metric] == min:
          is_best = eval_metrics[metric] <= best_metrics_vals[metric]
        else:
          is_best = eval_metrics[metric] >= best_metrics_vals[metric]
        if is_best:
          logging.info("- Found new best {}".format(metric))
          trainer.save_model(epoch, save_file='best.pt')
          eval_metrics.update({'epoch':epoch+1})
          best_metrics_vals[metric] = eval_metrics[metric]
          best_metrics[metric] = eval_metrics
          utils.save_dict_to_json(
              best_metrics,
              os.path.join(exp_dir, "metrics_eval_best_{}.json".format(metric))
              )

    # early stop
    if early_stopping:
      early_stopping(eval_metrics['loss'], model)
      if early_stopping.early_stop:
        logging.info("EARLY STOPPING")
        break

    utils.save_dict_to_json(
        all_eval_metrics,
        os.path.join(exp_dir, "all_metrics_eval.json")
        )
    utils.save_dict_to_json(all_train_metrics,
        os.path.join(exp_dir, "all_metrics_train.json")
        )
  return best_metrics


if __name__ == '__main__':

  # load the parameters from json file
  args = parser.parse_args()
  json_path = os.path.join(args.exp_dir, 'config.json')
  assert os.path.isfile(json_path), f'Not a json configuration file found at "{json_path}"!'
  cfg = utils.get_config(json_path)#TODO add inheritance config

  # log file
  exp_dir = os.path.join(args.exp_dir,'training')

  # set device
  is_cuda = (torch.cuda.is_available() and not args.no_cuda)
  device = torch.device("cuda" if is_cuda else "cpu")

  # set seed
  seed = cfg.seed
  torch.manual_seed(seed)
  if is_cuda: torch.cuda.manual_seed(seed)
  utils.fix_rng_seed(seed)

  # data loaders
  if not (args.eval_only or args.test_only):
    train_loader = builder.get_dataloaders(
        cfg['training'].dataloader,
        training=True,
        do_augment=args.augment,
        shuffle=True)

  if not args.test_only:
    cfg['visualizing'] = cfg['evaluating']
    cfg['visualizing']['dataloader'].batch_size = 16

    vis_loader = builder.get_dataloaders(
        cfg['visualizing'].dataloader,
        training=False,
        do_augment=False,
        shuffle=True)
    vis_data = next(iter(vis_loader))
    vis_loader = None

    val_loader = builder.get_dataloaders(
        cfg['evaluating'].dataloader,
        training=False,
        do_augment=False,
        shuffle=False)

  if args.test_only:
    pass #TODO

  # model & optimizer & scheduler
  model = builder.get_model(
      cfg['model'],
      device=device)
  if model is not None:
    # optimizer
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = builder.get_optimizer(
        cfg['training'].optimizer,
        cfg['training'].optimizer_kwargs,
        cfg['training'].lr,
        trainable_parameters)
    # scheduler
    scheduler = builder.get_scheduler(
        cfg['training'].get('scheduler', None),
        cfg['training'].get('scheduler_kwargs',{}),
        optimizer)
  else:
    optimizer = None
    scheduler = None

  # metrics_fns
  metrics_fns = builder.get_metrics_fns(cfg['metrics'])
  metrics_ops = cfg['metrics'].get('ops', {'loss':min})

  # specific trainer object
  side_fns = lambda x: {
      'verts2joints_fn': x.dataset._verts2joints_fn,
      'mano_fn':x.dataset._mano_fn
      }
  trainer = builder.get_trainer(
      cfg['model'],
      model,
      optimizer,
      scheduler,
      metrics_fns,
      device,
      exp_dir,
      **side_fns(val_loader if not args.test_only else test_loader)
      )


  # train & validate & test
  if args.eval_only  or args.test_only:
    epoch = trainer.restore_model(
        args.restore_file,
        off_scheduler=True,
        )
    exp_dir = os.path.join(
        args.exp_dir,
        f'{"eval_on_" if args.eval_only else "test_on_"}{args.restore_file[:-3]}')

    # set loggers 
    logpath = os.path.join(exp_dir, 'logFile.log')
    utils.set_logger(logpath)

    # run
    post_metrics = evaluate(
        trainer,
        val_loader if args.eval_only else test_loader,#TODO create inheritance from val config maybe
        vis_data,
        epoch=epoch,
        vis_dir=exp_dir)

    # logging metrics
    logging.info(F'== {"eval" if args.eval_only else "testing" } @ "{args.restore_file}"==')
    logging.info("- metrics: " + " ; ".join(F"{k}: {v:05.3f}" for k, v in post_metrics.items()))

    # save metrics
    utils.save_dict_to_json(
        post_metrics,
        os.path.join(exp_dir, "result.json"),
        )

  else:
    # set loggers & writers
    logpath = os.path.join(exp_dir, 'logFile.log')
    utils.set_logger(logpath)
    logging.info('======== Training ========')
    writer = SummaryWriter(exp_dir, max_queue=100)
    logging.info(exp_dir)

    # run
    best_metrics = train_evaluate(
        trainer,
        train_loader,
        val_loader,
        vis_data,
        metrics_ops,
        writer,
        num_epochs=cfg['training'].get('num_epochs',100),
        eval_freq=cfg['training'].get('eval_freq', 1),
        backup_freq=cfg['training'].get('backup_freq',50),
        restore_file=args.restore_file,
        )

    # logging metrics
    logging.info("Final Mean Metrics")
    logging.info("- metrics: \n" + " :: ".join(F"{k}: {v:05.3f}" for k, v in best_metrics.items()))

    # save metrics
    best_metrics_path = os.path.join(exp_dir, 'best_metrics.json')
    utils.save_dict_to_json(best_metrics, best_metrics_path)

