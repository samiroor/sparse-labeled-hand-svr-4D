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
from evaluate_modules import Sampler


def get_images_file(
    test_loader,
    sampler_mode,
    methods_dict,
    visual_fn,
    samples_ids=[0,1,2,3],
    save_dir='tmp'
    ):

  # results dict
  results_dict = {}
  for method, _dir in methods_dict.items():
    results_dict[method] = pop_pred_samples(
        os.path.join(_dir, 'pred.pkl'),
        samples_ids,
        )

  # gt_dict
  gt_dict = pop_gt_samples(
      test_loader,
      sampler_mode,
      samples_ids,
      )

  # images 
  for method, _dir in methods_dict.items():
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
    test_loader,
    sampler_mode,
    samples_ids,
    ):
  assert sampler_mode=="unsupervised", NotImplementedError
  data_sampler = Sampler(test_loader, sampler_mode)
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
    args,
    methods_dict,
    save_dir,
    ):

  json_path = os.path.join(args.exp_dir, 'config.json')
  assert os.path.isfile(json_path),\
      f'Not a json configuration file found at "{json_path}"!'

  cfg = utils.get_config(json_path)

  # set test_split
  if 'dataloader' in cfg['testing']:
    if 'datasets' in cfg['testing'].dataloader:
      for k in cfg['testing']['dataloader'].datasets:
        cfg['testing']['dataloader'].datasets[k]['split'] = args.on_split
        print(f"test_split is '{args.on_split}' in {k} dataset")

  ### data loaders ###
  test_loader = builder.get_dataloaders(
        cfg['testing'].dataloader,
        data_cfg=cfg['data_spec'],
        augmentation_cfg={},
        training=False,
        shuffle=False,
        )

  #### samples ids ### #TODO
  samples_ids = [4095, 4094, 4092, 4093, 4096, 4091, 4090, 12821, 12823, 12824] # np.random.choice(2000,16)

  #### get images ####
  t = BaseTrainer()
  get_images_file(
      test_loader,
      args.sampler_mode,
      methods_dict,
      visual_fn=t._visualize_step,
      samples_ids=samples_ids,
      save_dir=save_dir,
      )


