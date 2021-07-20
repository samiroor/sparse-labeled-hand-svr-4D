import os
import open3d
import pickle
import logging
import numpy as np
from tqdm import tqdm
import commentjson as json

import torch

from data import Fields as Fls
from metrics.hand3d import xyz_error, mesh_error



### evaluate mesh predictor module ###
def _evaluate_mesh_predictor(
    model,
    device,
    val_loader,
    sampler_mode='unsupervised',
    ):

  gt_joints3d_list = []; gt_verts3d_list = []
  pred_joints3d_list = []; pred_verts3d_list = []

  model.eval()
  data_sampler = iter(val_loader)

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
  gt_joints3d_list = np.concatenate(gt_joints3d_list, axis=0)
  gt_verts3d_list = np.concatenate(gt_verts3d_list, axis=0)
  pred_joints3d_list = np.concatenate(pred_joints3d_list, axis=0)
  pred_verts3d_list = np.concatenate(pred_verts3d_list, axis=0)

  return gt_joints3d_list, gt_verts3d_list, pred_joints3d_list, pred_verts3d_list



### evaluate flow module ###
def _evaluate_flow_infer(
    model,
    device,
    val_loader,
    sampler_mode='unsupervised_supervised',
    ):

  gt_joints3d_list = []; gt_verts3d_list = []
  pred_joints3d_list = []; pred_verts3d_list = []

  model.eval()
  data_sampler = iter(val_loader)

  with tqdm(total=len(data_sampler)) as t:
    for i in range(len(data_sampler)):
      sample_s, sample_u = next(data_sampler)

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
  gt_joints3d_list = np.concatenate(gt_joints3d_list, axis=0)
  gt_verts3d_list = np.concatenate(gt_verts3d_list, axis=0)
  pred_joints3d_list = np.concatenate(pred_joints3d_list, axis=0)
  pred_verts3d_list = np.concatenate(pred_verts3d_list, axis=0)

  return gt_joints3d_list, gt_verts3d_list, pred_joints3d_list, pred_verts3d_list



### evaluate  module ###
def _evaluate(
    model,
    device,
    val_loader,
    sampler_mode,
    module_name='mesh_predictor',
    save_preds_gts=True,
    calc_error=True,
    calc_mesh_error=True,
    save_dir=None,
    ):

  logging.info(f"Evaluating module {module_name} ...")
  gt_file = os.path.join(save_dir, 'gt.pkl')
  pred_file = os.path.join(save_dir, 'pred.pkl')
  error_file = os.path.join(save_dir, 'errors.pkl')

  # obtain gt & pred
  if os.path.exists(pred_file) and os.path.exists(gt_file):
    logging.info(f"LOADING GTs from {gt_file}")
    logging.info(f"LOADING PRDs from {pred_file}")
    gt_joints3d_list, gt_verts3d_list = read_pickle(gt_file)
    pred_joints3d_list, pred_verts3d_list = read_pickle(pred_file)

  else:
    logging.info(f"RUNNING module {module_name} in sampler mode {sampler_mode}")
    if module_name == "flow_infer":
      assert sampler_mode == "unsupervised_supervised"
      gt_joints3d_list, gt_verts3d_list, pred_joints3d_list, pred_verts3d_list = _evaluate_flow_infer(
          model,
          device,
          val_loader,
          sampler_mode)
    elif module_name == "mesh_predictor":
      assert sampler_mode in ("unsupervised", "supervised", "all")
      gt_joints3d_list, gt_verts3d_list, pred_joints3d_list, pred_verts3d_list = _evaluate_mesh_predictor(
          model,
          device,
          val_loader,
          sampler_mode)
      pass
    else:
      raise NotImplementedError

    # save preds & gts
    if save_preds_gts:
      logging.info(f"SAVING GTs in {gt_file}")
      logging.info(f"SAVING PRDs on {pred_file}")
      save_pickle([gt_joints3d_list, gt_verts3d_list], gt_file)
      save_pickle([pred_joints3d_list, pred_verts3d_list], pred_file)

  # calc errors & metrics
  if calc_error:
    assert not os.path.exists(error_file), f"REMOVE {error_file} to calculate error!"
    logging.info(f"Calculating Errors")
    # calc errors
    calc_metrics(
        gt_joints3d_list=gt_joints3d_list,
        gt_verts3d_list=gt_verts3d_list,
        pred_joints3d_list=pred_joints3d_list,
        pred_verts3d_list=pred_verts3d_list,
        save_file=error_file,
        calc_mesh_error=calc_mesh_error,
        )



### calc metrics ###
def calc_metrics(
    gt_joints3d_list,
    gt_verts3d_list,
    pred_joints3d_list,
    pred_verts3d_list,
    save_file,
    calc_mesh_error=True):


  # calc metrics
  logging.info(f'Calculating joints error ...')
  joints_error_dict, _= xyz_error(
      xyz_list=gt_joints3d_list,
      vis_list=None,
      xyz_pred_list=pred_joints3d_list,
      report_error_list=True, #to cherry pick
      )

  logging.info(f'Calculating mesh error ...')
  if calc_mesh_error:
    mesh_error_dict, _ = mesh_error(
      verts_list=gt_verts3d_list,
      verts_pred_list=pred_verts3d_list,
      )
  else:
    mesh_error_dict = {}


  logging.info(f'error saved @ {save_file}')
  save_pickle (
      {'joints': joints_error_dict, 'mesh': mesh_error_dict},
      save_file)

  logging.info(
      "- Eval joints metrics:\n" + " \n ".join(F"{k}: {v:03.3f}" for k, v in joints_error_dict.items()
        if not isinstance(v, list))
      )

  logging.info(
      "- Eval mesh metrics:\n" + " \n ".join(F"{k}: {v:03.3f}" for k, v in mesh_error_dict.items()
        if not isinstance(v, list))
      )



### data sampler ###

#class EvalDataset(ShImgDataset):
#  def __init__(
#      self,
#      dataset,
#      labelset,
#      sampler_mode,
#      ):
#
#    super().__init__(
#        dataset,
#        labelset,
#        do_augment=False,
#        augmentations={})
#
#    self.sampler_mode = sampler_mode
#
#    # logging
#    if self.sampler_mode == "all":
#      logging.info(
#      f"{len(self.dataset.list_videos_frames)}\
#      samples in mode {self.sampler_mode}.")
#
#    elif self.sampler_mode in ("supervised", ):
#      logging.info(
#      f"{len(self.dataset.list_supervised_videos_frames)} \
#      samples in mode {self.sampler_mode}.")
#
#    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
#      logging.info(
#      f"{len(self.dataset.list_unsupervised_videos_frames)} \
#          samples in mode {self.sampler_mode}.")
#
#  def __len__(self):
#    if self.sampler_mode == "all":
#      return self.dataset.num_videos_frames
#    elif self.sampler_mode in ("supervised", ):
#      return self.dataset.num_supervised_videos_frames
#    elif self.sampler_mode in ("unsupervised_supervised", "unsupervised"):
#      return self.dataset.num_unsupervised_videos_frames
#    else:
#      raise NotImplementedError
#
#  def __getitem__(self, item):
#    return self.pop_sample(item)
#
#  def pop_sample(self, item):
#    # all frames
#    if self.sampler_mode == "all":
#      video_frame = self.dataset.list_videos_frames[item]
#      video_id, frame = map(int, video_frame.split(':'))
#      video = self.dataset.videos[video_id]
#      frame_status = frame in video._supervised_frames
#      sample, meta = self.get_sample(video, frame, frame_status, video_id)
#      sample.pop('aug')
#      return sample
#
#    # supervied frames
#    elif self.sampler_mode == "supervised":
#      video_frame = self.dataset.list_supervised_videos_frames[item]
#      video_id, frame = map(int, video_frame.split(':'))
#      video = self.dataset.videos[video_id]
#      frame_status = frame in video._supervised_frames
#      assert frame_status == 1
#      sample, meta = self.get_sample(video, frame, frame_status, video_id)
#      sample.pop('aug')
#      return sample
#
#    # unsupervised frames
#    elif self.sampler_mode == "unsupervised":
#      video_frame = self.dataset.list_unsupervised_videos_frames[item]
#      video_id, frame = map(int, video_frame.split(':'))
#      video = self.dataset.videos[video_id]
#      frame_status = frame in video._supervised_frames
#      assert frame_status == 0
#      sample, meta = self.get_sample(video, frame, frame_status, video_id)
#      sample.pop('aug')
#      return sample
#
#    # keep  order of (s, u) to feed network!
#    elif self.sampler_mode == "unsupervised_supervised":
#      video_frame = self.dataset.list_unsupervised_videos_frames[item]
#      video_id, frame = map(int, video_frame.split(':'))
#      video = self.dataset.videos[video_id]
#      frame_status = frame in video._supervised_frames
#      assert frame_status == 0
#      sample_u, meta = self.get_sample(video, frame, frame_status, video_id)
#
#      frame_s = video.get_nearest_supervised_frames(frame, n=1)[0]
#      frame_status = frame_s in video._supervised_frames
#      assert frame_status == 1
#      sample_s,meta = self.get_sample(video, frame_s, frame_status, video_id)
#      sample_u.pop('aug')
#      sample_s.pop('aug')
#      return sample_s, sample_u, (frame_s, frame)
#
#    else:
#      raise NotImplementedError
#
#
#def Sampler(val_loader, sampler_mode='supervised'):
#  eval_dataset = EvalDataset(
#      dataset=val_loader.dataset.dataset,
#      labelset=val_loader.dataset.labelset,
#      sampler_mode=sampler_mode,
#      )
#
#  return torch.utils.data.DataLoader(
#      dataset=eval_dataset,
#      batch_size=32,
#      num_workers=8,
#      shuffle=False,
#      pin_memory=True,
#      )


def read_pickle(filename):
  with open(filename, 'rb') as fp:
    filedata = pickle.load(fp)
  return filedata

def save_pickle(filedata, filename):
  with open(filename, 'wb') as fp:
    pickle.dump(filedata, fp)




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
