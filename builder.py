import os
import copy
import commentjson as json
import shutil
import pickle
import logging
import configparser
import numpy as np
#import open3d as o3d
from easydict import EasyDict

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

import data
import metrics
import utils

from methods import obman, graph, pairflow


method_dict = {
    'obman':obman,
    'graph':graph,
    'pairflow':pairflow
    }

scheduler_dict = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    }

# ********************** DATA **********************


def get_dataloaders(
    cfg,
    data_cfg,
    augmentation_cfg,
    mano_root='misc/mano_v1_2/models',
    training=False,
    shuffle=False,
    ):
  """
    Args:
      cfg (EasyDict) : dataloader config.
      data_cfg (EasyDict): data specification config.
      augmentation_cfg (EasyDict): augmentation config.
  """
  dataloader = data.get_dataloaders(
      cfg,
      data_cfg,
      augmentation_cfg,
      mano_root=mano_root,
      training=training,
      shuffle=shuffle,
      )
  return dataloader


# ********************** TRAINER **********************


def get_trainer(
    cfg,
    model,
    optimizer,
    scheduler,
    metrics_fns,
    device,
    exp_dir,
    **kwargs
    ):
  method_name = cfg.name
  if method_name in method_dict:
    return method_dict[method_name].get_trainer(
        cfg[method_name],
        model,
        optimizer,
        scheduler,
        metrics_fns,
        device,
        exp_dir,
        **kwargs,
        )
  else:
    raise NotImplementedError(f"model name {method} is not yet defined.")


# ********************** MODEL **********************


def get_model(
    cfg,
    device,
    **kwargs
    ):
  method_name = cfg.name
  if method_name in method_dict:
    model = method_dict[method_name].get_model(
        cfg[method_name],
        device,
        **kwargs,
        )
  else:
    raise NotImplementedError(f"model name {method} is not yet defined.")

  return torch.nn.DataParallel(model).to(device)

# ********************** OPTIMIZER **********************


def get_optimizer(
    method_name,
    opt_name,
    opt_kwargs,
    lr,
    model,
    ):

  """Creates an optimizer based on the opt_config and model parameters.
  Args:
    - opt_name (str):
    - opt_kwargs: (EasyDict) optimizer config. It should have corresponding fields.
    - lr (float): learning rate
    - parameters (nn.params): trainable parameters.
  Returns:
    - optimizer object
  """
  if method_name in method_dict:
    return method_dict[method_name].get_optimizer(
        opt_name,
        opt_kwargs,
        lr,
        model,
        )
  else:
    raise NotImplementedError(f"model name {method} is not yet defined.")



def get_scheduler(
    sch_name,
    sch_kwargs,
    optimizer,
    ):
  """ Creates a learning-rate scheduler
  Args:
    - sch_name (str):   scheduler name. ("StepLR"|"MultiStepLR"|"ExponentialLR").
    - sch_kwargs (EasyDict): schedular specific parameters.
    - optimizer (torch.optim)
  Returns:
    - scheduler object
  """
  if sch_name in scheduler_dict:
    return scheduler_dict[sch_name](
        optimizer,
        **sch_kwargs,
        )
  else:
    return None


# ********************** METRICS **********************

def get_metrics_fns(cfg, **kwargs):
  return metrics.get_metrics_fns(cfg.names)


