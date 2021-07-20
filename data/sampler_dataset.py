import logging
import torch
from .singlehand_img_dataset import ShImgDataset



class SamplerDataset(ShImgDataset):
  def __init__(
      self,
      dataset,
      labelset,
      do_augment=False,
      augmentation={},
      training=False,
      sampler_mode='all',
      **kwargs,
      ):

    super().__init__(
        dataset,
        labelset,
        do_augment=do_augment,
        augmentations=augmentation,
        training=training,
        )

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
    return self.pop_sample(item)

  def pop_sample(self, item):
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
      return sample_s, sample_u

    else:
      raise NotImplementedError


