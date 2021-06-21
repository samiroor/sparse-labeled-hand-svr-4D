import os
import random
import numpy as np
from collections import defaultdict

import torch
from .supervised_singlehand_img_dataset import SupervisedShImgDataset


"""
Pair Supervised frame Any frame
Picking a Supervised frame along with an  frame with at least gap distance
Single Hand Img dataset!

"""


class PairSupAnyShImgDataset(SupervisedShImgDataset):
  def __init__(self,
               dataset,
               labelset,
               gap=1,
               do_augment=False,
               augmentation={},
               training=False,
               **kwargs):
    """
    Args:
      gap (int)|tuple: distance between selected supervised frame and selected unsupervised frame.

      return a supervised sample and unsupervised sample
      (with flag of supervisory status to optional check )
      return metas: return sample filename along sample

      *len of loader is number of supervised frames!

    """
    super().__init__(dataset,
                     labelset,
                     do_augment=do_augment,
                     augmentation=augmentation,
                     training=training,
                     **kwargs)

    self.gap = gap

  def __getitem__(self, item):

    if self.training:
      video, supervised_frame, one_frame, one_frame_status, video_id = self.next_trainitem(item)
    else:
      video, supervised_frame, one_frame, one_frame_status, video_id = self.next_valitem(item)

    supervised_sample, supervised_meta = self.get_sample(video, supervised_frame, 1, video_id)
    supervised_sample.pop('aug')

    one_sample, one_meta = self.get_sample(video, one_frame, one_frame_status, video_id)
    one_sample.pop('aug')

    if self.return_metas:
      return supervised_sample, one_sample, (supervised_meta, one_meta)

    return supervised_sample, one_sample


  def next_trainitem(self, item):
    video, supervised_frame, frame_status, video_id = self.get_supervised_frame(item)
    assert frame_status == 1,'Expected to be picked from supervised collection.'

    gap = self.gap if isinstance(self.gap, int) else random.randint(self.gap[0], self.gap[1])

    one_frame, one_frame_status = self.pass_one_frame(
        video,
        supervised_frame,
        gap=gap,
        first_next=random.random()>0.5,
        )

    return video, supervised_frame, one_frame, one_frame_status, video_id


  def next_valitem(self, item):
    video, supervised_frame, frame_status, video_id = self.get_supervised_frame(item)
    assert frame_status == 1,'Expected to be picked from supervised collection.'

    # set seed to pick same frames for every epoch evaluation
    random_state = np.random.RandomState(item)
    gap = self.gap if isinstance(self.gap, int) else random_state.randint(self.gap[0], self.gap[1])

    one_frame, one_frame_status = self.pass_one_frame(
        video,
        supervised_frame,
        gap=gap,
        first_next=random_state.choice([0,1]),
        )

    return video, supervised_frame, one_frame, one_frame_status, video_id


  def pass_one_frame(
      self,
      video,
      frame,
      gap,
      first_next=True,
      s_u_ratio = 0.5,
      ):

    """
      To have correct setting, note that
      supervised frames are set as Start + Every 1/supervisory_fraction frames + End
      so to have an unsupervised frame picked logically, given spacing should satisfy
      "spacing < 1/supervisory_fraction"
    """
    if random.random() < s_u_ratio:
      frames = video.get_nearest_supervised_frames(
          frame + gap if first_next else frame - gap,
          n=1,
          )
      return frames[0], 1
    else:
      return self.pass_unsupervised_frame(
          video,
          frame,
          gap,
          first_next,
          )

  def pass_unsupervised_frame(self, video, frame, gap, first_next=True):
    """
      To have correct setting, note that
      supervised frames are set as Start + Every 1/supervisory_fraction frames + End
      so to have an unsupervised frame picked logically, given spacing should satisfy
      "spacing < 1/supervisory_fraction"
    """
    if first_next:
      frames, _ = video.get_next_unsupervised_frames(frame + gap, n=1, spacing=1)
      if len(frames) < 1:
        frames,_ = video.get_prior_unsupervised_frames(frame - gap, n=1, spacing=1)

    else:
      frames,_ = video.get_prior_unsupervised_frames(frame - gap, n=1, spacing=1)
      if len(frames) < 1:
        frames, _ = video.get_next_unsupervised_frames(frame + gap, n=1, spacing=1)

    assert len(frames) !=0, "Expected at least one unsupervised frame!"
    return frames[0], frames[0] in video._supervised_frames
