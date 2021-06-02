import commentjson as json
import logging
import os
import math
import shutil
import numpy as np
import torch
from datetime import datetime, timedelta
from torch import nn, optim
from torch.nn import functional as F
from easydict import EasyDict
from prettytable import PrettyTable

_RNG_SEED = None


def get_config(path, default_path=None):
  # load configuration from file itself
  with open(path, 'r') as f:
    cfg_special = EasyDict(json.load(f))

  # load parent config or default config
  inherit_from = cfg_special.get('inherit_from', None)

  if inherit_from is not None:
    cfg = get_config(inherit_from, default_path)

  elif default_path is not None:
    with open(default_path, 'r') as f:
      cfg = yaml.load(f)

  else:
    cfg = EasyDict()

  # include main configuration
  update_recursive(cfg, cfg_special)
  return cfg


def update_recursive(dict1, dict2):
  ''' Update two config dictionaries recursively.
  Args:
    dict1 (dict): first dictionary to be updated
    dict2 (dict): second dictionary which entries should be used
  '''
  for k, v in dict2.items():
    if k not in dict1:
      dict1[k] = EasyDict()
    if isinstance(v, dict):
      update_recursive(dict1[k], v)
    else:
      dict1[k] = v


def get_spec_name(cfg, name=''):
  for k,v in cfg.items():
    if k in (
        "labelset",
        "do_augment",
        "add_first_last_frame",
        "use_mini_dataset",
        ):
      continue
    name += f'_{k}_'
    if isinstance(v, dict):
      name = get_spec_name(v, name)
    else:
      name += ','.join(map(str,v)) if isinstance(v, list) else str(v)
  return name


def adjust_bn_params(
    model,
    module_name,
    fix=True):
  if fix:
    for name, module in model.named_modules():
      if module_name in name:
        if isinstance(module, torch.nn.modules.BatchNorm1d):
          module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
          module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
          module.eval()


def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed.
    Args:
        seed (int):
    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.
    Example:
        Fix random seed in both tensorpack and tensorflow.
    .. code-block:: python
            import utils
            seed = 42
            utils.fix_rng_seed(seed)
            torch.manual_seed(seed)
            if config.cuda: torch.cuda.manual_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


class NumpyRNG(object):
  """
  Singleton class for numpy rng.
  It should be created after calling fix_rng_seed
  """
  _rng = None
  def __new__(cls, *args, **kwargs):
    if not cls._rng:
      cls._rng = get_rng()
    return cls._rng



def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
      for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def save_dict_to_json(
    d,
    json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    def to_float(d):
      for k,v in d.items():
        if type(v) is dict:
          d[k] = to_float(v)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
          d[k] = float(v)
        elif isinstance(v, list):
          d[k] = [float(x) for x in v]
      return d

    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = to_float(d) #{k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=2)



def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    scheduler=None,
    allow_partial=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    assert os.path.exists(filepath), f"File doesn't exist {filepath}"
    checkpoint = torch.load(filepath)

    if allow_partial:
      model_dict = model.state_dict()
      pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
      for k in pretrained_dict:
        logging.warning("Loading pretrained weight: {}".format(k))
      model_dict.update(pretrained_dict)
      model.load_state_dict(model_dict)
    else:
      model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint



def save_checkpoint(
    state,
    filepath):
    """Saves model and training parameters at checkpoint

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    """
    torch.save(state, filepath)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    logging.info(table)
    logging.info(f"Total Trainable Params: {total_params}")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, self.val_loss_min))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

