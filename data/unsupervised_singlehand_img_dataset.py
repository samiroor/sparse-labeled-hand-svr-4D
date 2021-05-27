import os
import random
from collections import defaultdict

import torch
from .singlehand_img_dataset import ShImgDataset


"""
  Unsupervised Single Hand Img dataset!
"""


class UnsupervisedShImgDataset(ShImgDataset):
  def __init__(self,
               dataset,
               res_size,
               labelset,
               augmentation,
               do_augment=False,
               training=False,
               **kwargs):
    """
    Args:
      return an unsupervised sample
      (with flag of supervisory status to optional check , it should be always zero)
      return metas: return sample filename along sample

      *len of loader is number of unsupervised frames!
    """

    super().__init__(dataset,
                     res_size,
                     labelset,
                     augmentation,
                     do_augment=do_augment,
                     training=training,
                     **kwargs)


  def __len__(self):
    return self.dataset.num_unsupervised_videos_frames


  def next_trainitem(self, item):
    return self.get_unsupervised_frame(item)


  def next_valitem(self, item):
    return self.next_trainitem(item)


  def get_unsupervised_frame(self, item):
    video_frame = self.dataset.list_unsupervised_videos_frames[item]
    video_id, frame = int(video_frame.split(':')[0]),int(video_frame.split(':')[-1])
    # get video object
    video = self.dataset.videos[video_id]
    return video, frame, frame in video._supervised_frames, video_id
