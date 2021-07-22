import os
import pickle
import numpy as np
import commentjson as json

import torch

import utils
import builder
from data import Fields as Fls
from trainer import BaseTrainer
from visualize.visualizer  import save_images


def get_images_file(
    val_loader,
    sampler_mode,
    methods_list,
    visual_fn,
    samples_ids,
    save_dir,
    ):

  # results dict
  results_dict = {}
  for method, _dir, _, in methods_list:
    results_dict[method] = pop_pred_samples(
        os.path.join(_dir, 'pred.pkl'),
        samples_ids,
        )

  # gt_dict
  gt_dict = pop_gt_samples(
      val_loader,
      sampler_mode,
      samples_ids,
      )

  # images 
  for method, _dir, _ in methods_list:
    pred_dict = results_dict[method]
    images = visualize_step(
        gt_dict=gt_dict,
        pred_dict=pred_dict,
        visual_fn=visual_fn,
        )

    print(f'saved images for {method}')
    save_images(
        images,
        title=method,
        save_dir=os.path.join(save_dir, method)
        )



#### visulaize batch pair gts & preds ####
def visualize_step(
    gt_dict,
    pred_dict,
    visual_fn,
    ):
  # with neural renderer
  images = visual_fn(
      gt_dict=gt_dict,
      pred_dict=pred_dict,
      display_render=True,
      return_render_images=True,
      )

  return images.detach().cpu().numpy()


#### pop samples based on given ids ####

def pop_gt_samples(
    val_loader,
    sampler_mode,
    samples_ids,
    ):
  assert sampler_mode=="unsupervised", NotImplementedError
  data_sampler = val_loader
  batch_samples = []
  for i in samples_ids:
    sample = data_sampler.dataset.pop_sample(i)
    sample = dict(
        map(lambda kv:(kv[0], torch.tensor(kv[1]) if isinstance(kv[1], (np.ndarray, bool)) else kv[1]), sample.items())
        )
    batch_samples.append(sample)

  batch_samples = {
      k:torch.stack([sample[k] for sample in batch_samples]).cuda().float() for k in batch_samples[0]}
  return batch_samples


def pop_pred_samples(
    pred_dir,
    samples_ids,
    ):

  with open(pred_dir, 'rb') as fp:
    pred = pickle.load(fp)
  pred_joints3d_list = pred[0]
  pred_verts3d_list = pred[1]
  return {
      Fls.joints3d:torch.from_numpy(pred_joints3d_list[samples_ids]).cuda().float(),
      Fls.verts3d:torch.from_numpy(pred_verts3d_list[samples_ids]).cuda().float()
      }



##### main ####

def main(
    methods_list,
    sampler_mode,
    split,
    exp_dir,
    save_dir,
    criteria1='Baseline + Alignment Supervision',
    criteria2='Baseline'
    ):

  json_path = os.path.join(exp_dir, 'config.json')
  assert os.path.isfile(json_path),\
      f'Not a json configuration file found at "{json_path}"!'

  cfg = utils.get_config(json_path)

  # set test_split
  if 'dataloader' in cfg['testing']:
    if 'datasets' in cfg['testing'].dataloader:
      for k in cfg['testing']['dataloader'].datasets:
        cfg['testing']['dataloader'].datasets[k]['split'] = split
        cfg['testing']['dataloader'].datasets[k]['sampler_mode'] = sampler_mode
        print(f"test_dataset is {k} on split '{split}' with {sampler_mode} samples")

  ### data loaders ###
  val_loader = builder.get_dataloaders(
        cfg['testing'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=False,
        )

  #### sample ids ####
  for method, _dir, _ in methods_list:
    if criteria1 == method:
      error_file = os.path.join(_dir, 'errors.pkl')
      criteria1 = read_pickle(error_file)['xyz_mean3d_list']
      break

  for method, _dir, _ in methods_list:
    if criteria2 == method:
      error_file = os.path.join(_dir, 'errors.pkl')
      criteria2 = read_pickle(error_file)['xyz_mean3d_list']
      break

  dist = np.array(criteria1) - np.array(criteria2)
  idx = dist.argsort()
  suc_ids = idx[:64]
  fail_ids = idx[-64:]

  #### get images ####
  for name, samples_ids in zip(['success', 'failure'], [suc_ids, fail_ids]):
    t = BaseTrainer()
    get_images_file(
      val_loader,
      sampler_mode,
      methods_list,
      visual_fn=t._visualize_step,
      samples_ids=samples_ids,
      save_dir=os.path.join(save_dir,name),
      )
