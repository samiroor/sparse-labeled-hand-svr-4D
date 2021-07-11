import os
import open3d
import pickle
import logging
import numpy as np
from tqdm import tqdm
import commentjson as json

import torch

from data import Fields as Fls
from data import ShImgDataset
from metrics.hand3d import xyz_error, mesh_error


### evaluate flow module ###
def evaluate_flow_infer(
    model,
    device,
    val_loader,
    vis_data,
    sampler_mode='unsupervised_supervised',
    save_preds=False,
    save_gts=False,
    save_errors=True,
    save_dir=None,
    ):

  gt_joints3d_list = []; gt_verts3d_list = []
  pred_joints3d_list = []; pred_verts3d_list = []

  model.eval()
  data_sampler = Sampler(val_loader, sampler_mode)
  data_sampler = iter(data_sampler)

  with tqdm(total=len(data_sampler)) as t:
    for i in range(len(data_sampler)):
      sample_s, sample_u, _ = next(data_sampler)

      sample_s = dict(
          map(lambda kv:(kv[0], kv[1].float().to(device)), sample_s.items())
          )
      sample_u = dict(
          map(lambda kv:(kv[0], kv[1].float().to(device)), sample_u.items())
          )

      with torch.no_grad():
        output_batch = model({
          'supervised':sample_s,
          'unsupervised':sample_u,
          },
          fwd=False,
          infer=True,
          )
        pred_u = output_batch['flow']['infer_unsupervised']

      # append gt_list & pred_list 
      sample_u = dict(
          map(lambda kv:(kv[0], kv[1].detach().cpu().numpy()), sample_u.items())
          )
      pred_u = dict(
          map(lambda kv:(kv[0], kv[1].detach().cpu().numpy()), pred_u.items())
          )

      gt_joints3d_list.append(sample_u[Fls.joints3d])
      gt_verts3d_list.append(sample_u[Fls.verts3d])

      pred_joints3d_list.append(pred_u[Fls.joints3d])
      pred_verts3d_list.append(pred_u[Fls.verts3d])

      t.set_postfix(sample=i+1, sampler_mode=sampler_mode)
      t.update()


  # concatenate
  dataset = None
  gt_joints3d_list = np.concatenate(gt_joints3d_list, axis=0)
  gt_verts3d_list = np.concatenate(gt_verts3d_list, axis=0)
  pred_joints3d_list = np.concatenate(pred_joints3d_list, axis=0)
  pred_verts3d_list = np.concatenate(pred_verts3d_list, axis=0)

  # calc errors
  calc_metrics(
      gt_joints3d_list=gt_joints3d_list,
      gt_verts3d_list=gt_verts3d_list,
      pred_joints3d_list=pred_joints3d_list,
      pred_verts3d_list=pred_verts3d_list,
      save_preds=save_preds,
      save_gts=save_gts,
      save_errors=save_errors,
      save_dir=save_dir,
      )


### evaluate mesh predictor module ###
def evaluate_mesh_predictor(
    model,
    device,
    val_loader,
    vis_data,
    sampler_mode='unsupervised',
    save_preds=False,
    save_gts=False,
    save_errors=True,
    save_dir=None,
    ):

  gt_joints3d_list = []; gt_verts3d_list = []
  pred_joints3d_list = []; pred_verts3d_list = []

  model.eval()
  data_sampler = Sampler(val_loader, sampler_mode)
  data_sampler = iter(data_sampler)

  with tqdm(total=len(data_sampler)) as t:
    for i in range(len(data_sampler)):
      sample = next(data_sampler)
      sample = dict(
          map(lambda kv:(kv[0], kv[1].float().to(device)), sample.items())
          )

      with torch.no_grad():
        pred = model(sample)

      # append gt_list & pred_list 
      sample = dict(
          map(lambda kv:(kv[0], kv[1].detach().cpu().numpy()), sample.items())
          )
      pred = dict(
          map(lambda kv:(kv[0], kv[1].detach().cpu().numpy()), pred.items())
          )

      gt_joints3d_list.append(sample[Fls.joints3d])
      gt_verts3d_list.append(sample[Fls.verts3d])

      pred_joints3d_list.append(pred[Fls.joints3d])
      pred_verts3d_list.append(pred[Fls.verts3d])

      t.set_postfix(sample=i+1, sampler_mode=sampler_mode)
      t.update()

  # concatenate
  dataset = None
  gt_joints3d_list = np.concatenate(gt_joints3d_list, axis=0)
  gt_verts3d_list = np.concatenate(gt_verts3d_list, axis=0)
  pred_joints3d_list = np.concatenate(pred_joints3d_list, axis=0)
  pred_verts3d_list = np.concatenate(pred_verts3d_list, axis=0)

  # calc errors
  calc_metrics(
      gt_joints3d_list=gt_joints3d_list,
      gt_verts3d_list=gt_verts3d_list,
      pred_joints3d_list=pred_joints3d_list,
      pred_verts3d_list=pred_verts3d_list,
      save_preds=save_preds,
      save_gts=save_gts,
      save_errors=save_errors,
      save_dir=save_dir)



### calc metrics ###
def calc_metrics(
    gt_joints3d_list,
    gt_verts3d_list,
    pred_joints3d_list,
    pred_verts3d_list,
    save_preds,
    save_gts,
    save_errors,
    save_dir):

  # dump preds & gts
  if save_preds:
    save_file = os.path.join(save_dir, 'pred.pkl')
    logging.info(f'predictions saved @ {save_file}')
    with open(save_file, 'wb') as fp:
      pickle.dump([pred_joints3d_list, pred_verts3d_list], fp)

  if save_gts:
    save_file = os.path.join(save_dir, 'gt.pkl')
    logging.info(f'gts saved @ {save_file}')
    with open(save_file, 'wb') as fp:
      pickle.dump([gt_joints3d_list, gt_verts3d_list], fp)

  # calc metrics
  logging.info(f'Calculating joints error ...')
  joints_error_dict, _= xyz_error(
      xyz_list=gt_joints3d_list,
      vis_list=None,
      xyz_pred_list=pred_joints3d_list,
      report_error_list=True, #to cherry pick
      )

  logging.info(f'Calculating mesh error ...')
  mesh_error_dict, _ = mesh_error(
      verts_list=gt_verts3d_list,
      verts_pred_list=pred_verts3d_list,
      )

  if save_errors:
    save_file = os.path.join(save_dir, 'errors.pkl')

    logging.info(f'error saved @ {save_file}')
    with open(save_file, 'wb') as fp:
      pickle.dump({'joints': joints_error_dict, 'mesh': mesh_error_dict},
          fp,
          )

  logging.info(
      "- Eval joints metrics:\n" + " \n ".join(F"{k}: {v:03.3f}" for k, v in joints_error_dict.items()
        if not isinstance(v, list))
      )

  logging.info(
      "- Eval mesh metrics:\n" + " \n ".join(F"{k}: {v:03.3f}" for k, v in mesh_error_dict.items()
        if not isinstance(v, list))
      )


### data sampler ###

class EvalDataset(ShImgDataset):
  def __init__(
      self,
      dataset,
      labelset,
      sampler_mode,
      ):

    super().__init__(
        dataset,
        labelset,
        do_augment=False,
        augmentations={})

    self.sampler_mode = sampler_mode

    # logging
    if self.sampler_mode == "all":
      logging.info(
      f"{len(self.dataset.list_videos_frames)}\
      samples in mode {self.sampler_mode}.")

    elif self.sampler_mode in ("supervised", ):
      logging.info(
      f"{len(self.dataset.list_supervised_videos_frames)} \
      samples in mode {self.sampler_mode}.")

    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
      logging.info(
      f"{len(self.dataset.list_unsupervised_videos_frames)} \
          samples in mode {self.sampler_mode}.")

  def __len__(self):
    if self.sampler_mode == "all":
      return self.dataset.num_videos_frames
    elif self.sampler_mode in ("supervised", ):
      return self.dataset.num_supervised_videos_frames
    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
      return self.dataset.num_unsupervised_videos_frames
    else:
      raise NotImplementedError


  def __getitem__(self, item):
    # all frames
    if self.sampler_mode == "all":
      video_frame = self.dataset.list_videos_frames[item]
      video_id, frame = map(int, video_frame.split(':'))
      video = self.dataset.videos[video_id]
      frame_status = frame in video._supervised_frames
      sample, meta = self.get_sample(video, frame, frame_status, video_id)
      sample.pop('aug')
      return sample

    # supervied frames
    elif self.sampler_mode == "supervised":
      video_frame = self.dataset.list_supervised_videos_frames[item]
      video_id, frame = map(int, video_frame.split(':'))
      video = self.dataset.videos[video_id]
      frame_status = frame in video._supervised_frames
      assert frame_status == 1
      sample, meta = self.get_sample(video, frame, frame_status, video_id)
      sample.pop('aug')
      return sample

    # unsupervised frames
    elif self.sampler_mode == "unsupervised":
      video_frame = self.dataset.list_unsupervised_videos_frames[item]
      video_id, frame = map(int, video_frame.split(':'))
      video = self.dataset.videos[video_id]
      frame_status = frame in video._supervised_frames
      assert frame_status == 0
      sample, meta = self.get_sample(video, frame, frame_status, video_id)
      sample.pop('aug')
      return sample

    # keep  order of (s, u) to feed network!
    elif self.sampler_mode == "unsupervised_supervised":
      video_frame = self.dataset.list_unsupervised_videos_frames[item]
      video_id, frame = map(int, video_frame.split(':'))
      video = self.dataset.videos[video_id]
      frame_status = frame in video._supervised_frames
      assert frame_status == 0
      sample_u, meta = self.get_sample(video, frame, frame_status, video_id)

      frame_s = video.get_nearest_supervised_frames(frame, n=1)[0]
      frame_status = frame_s in video._supervised_frames
      assert frame_status == 1
      sample_s,meta = self.get_sample(video, frame_s, frame_status, video_id)
      sample_u.pop('aug')
      sample_s.pop('aug')
      return sample_s, sample_u, (frame_s, frame)

    else:
      raise NotImplementedError


def Sampler(val_loader, sampler_mode='supervised'):
  eval_dataset = EvalDataset(
      dataset=val_loader.dataset.dataset,
      labelset=val_loader.dataset.labelset,
      sampler_mode=sampler_mode,
      )

  return torch.utils.data.DataLoader(
      dataset=eval_dataset,
      batch_size=32,
      num_workers=8,
      shuffle=False,
      pin_memory=True,
      )


#class Sampler(object):
#  def __init__(self, dataset, sampler_mode='supervised'):
#    self.dataset = dataset
#    self.sampler_mode = sampler_mode
#    self.i = 0
#
#    # logging
#    if self.sampler_mode == "all":
#      logging.info(
#      f"{len(self.dataset.dataset.list_videos_frames)}\
#      samples in mode {self.sampler_mode}.")
#
#    elif self.sampler_mode in ("supervised", ):
#      logging.info(
#      f"{len(self.dataset.dataset.list_supervised_videos_frames)} \
#      samples in mode {self.sampler_mode}.")
#
#    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
#      logging.info(
#      f"{len(self.dataset.dataset.list_unsupervised_videos_frames)} \
#          samples in mode {self.sampler_mode}.")
#
#  def __len__(self):
#    if self.sampler_mode == "all":
#      return self.dataset.dataset.num_videos_frames
#    elif self.sampler_mode in ("supervised", ):
#      return self.dataset.dataset.num_supervised_videos_frames
#    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
#      return self.dataset.dataset.num_unsupervised_videos_frames
#    else:
#      raise NotImplementedError
#
#  def __iter__(self):
#    self.i = 0
#    while self.i < self.__len__():
#      # all frames
#      if self.sampler_mode == "all":
#        video_frame = self.dataset.dataset.list_videos_frames[self.i]
#        video_id, frame = map(int, video_frame.split(':'))
#        video = self.dataset.dataset.videos[video_id]
#        frame_status = frame in video._supervised_frames
#        sample, meta = self.dataset.get_sample(video, frame, frame_status, video_id)
#        sample.pop('aug')
#        yield sample
#
#      # supervied frames
#      elif self.sampler_mode == "supervised":
#        video_frame = self.dataset.dataset.list_supervised_videos_frames[self.i]
#        video_id, frame = map(int, video_frame.split(':'))
#        video = self.dataset.dataset.videos[video_id]
#        frame_status = frame in video._supervised_frames
#        assert frame_status == 1
#        sample, meta = self.dataset.get_sample(video, frame, frame_status, video_id)
#        sample.pop('aug')
#        yield sample
#
#      # unsupervised frames
#      elif self.sampler_mode == "unsupervised":
#        video_frame = self.dataset.dataset.list_unsupervised_videos_frames[self.i]
#        video_id, frame = map(int, video_frame.split(':'))
#        video = self.dataset.dataset.videos[video_id]
#        frame_status = frame in video._supervised_frames
#        assert frame_status == 0
#        sample, meta = self.dataset.get_sample(video, frame, frame_status, video_id)
#        sample.pop('aug')
#        yield sample
#
#      # keep  order of (s, u) to feed network!
#      elif self.sampler_mode == "unsupervised_supervised":
#        video_frame = self.dataset.dataset.list_unsupervised_videos_frames[self.i]
#        video_id, frame = map(int, video_frame.split(':'))
#        video = self.dataset.dataset.videos[video_id]
#        frame_status = frame in video._supervised_frames
#        assert frame_status == 0
#        sample_u, meta = self.dataset.get_sample(video, frame, frame_status, video_id)
#
#        frame_s = video.get_nearest_supervised_frames(frame, n=1)[0]
#        frame_status = frame_s in video._supervised_frames
#        assert frame_status == 1
#        sample_s,meta = self.dataset.get_sample(video, frame_s, frame_status, video_id)
#        sample_u.pop('aug')
#        sample_s.pop('aug')
#        yield sample_s, sample_u, (frame_s, frame)
#
#      else:
#        raise NotImplementedError
#
#      self.i += 1
