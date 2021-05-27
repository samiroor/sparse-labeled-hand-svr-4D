import os
import random
from collections import defaultdict

import torch
from .supervised_singlehand_img_dataset import SupervisedShImgDataset


"""
Pair Supervised frame Unsupervised frame Single Hand Img dataset!
"""


class PairSupUnsupShImgDataset(SupervisedShImgDataset):
  def __init__(self,
               dataset,
               res_size,
               labelset,
               spacing,
               augmentation,
               do_augment=False,
               training=False,
               **kwargs):
    """
    Args:
      spacing (int)|tuple: space between supervised frame and selected unsupervised frame.

      return a supervised sample and unsupervised sample
      (with flag of supervisory status to optional check )
      return metas: return sample filename along sample

      *len of loader is number of supervised frames!

    """
    super().__init__(dataset,
                     res_size,
                     labelset,
                     augmentation,
                     do_augment=do_augment,
                     training=training,
                     **kwargs)

    self.spacing = spacing


  def __getitem__(self):
    video, supervised_frame, unsupervised_frame, video_id = self.next_trainitem(item) if self.training  else self.next_valitem(item)

    supervised_sample, supervised_meta = self.get_sample(video, supervised_frame, 1, video_id)
    supervised_sample.pop('aug')

    unsupervised_sample, unsupervised_meta = self.get_sample(video, unsupervised_frame, 0, video_id)
    unsupervised_sample.pop('aug')

    if self.return_metas:
      return supervised_sample, unsupervised_sample, (supervised_meta, unsupervised_meta)

    return supervised_sample, unsupervised_sample


  def next_trainitem(self, item):
    video, supervised_frame, frame_status, video_id = self.get_supervised_frame(item)
    assert frame_status == 1,'Expected to be picked from supervised collection.'
    unsupervised_frame, frame_status = self.select_unsupervised_frame(video, supervised_frame)
    assert frame_status == 0,'Expected to be picked from unsupervised collection.'
    return video, supervised_frame, unsupervised_frame, video_id


  def next_valitem(self, item):
    return self.next_trainitem(item)


  def select_unsupervised_frame(self, video, frame, len_seq=1):
    spacing = self.spacing if isinstance(self.spacing, int) else random.randint(self.spacing[0], self.spacing[1])

    frames, _ = video.get_next_unsupervised_frames(frame, len_seq, spacing)

    if len(frames) < len_seq:
      pad,_ = video.get_prior_frames(frame-1, len_seq - len(frames), spacing)
      frames.extend(pad)

    return frames[-1], frames[-1] in video._supervised_frames
