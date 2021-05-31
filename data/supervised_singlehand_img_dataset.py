import os
import random
from collections import defaultdict

import torch
from .singlehand_img_dataset import ShImgDataset


"""
  Supervised Single Hand Img dataset!
"""


class SupervisedShImgDataset(ShImgDataset):
  def __init__(self,
               dataset,
               labelset,
               do_augment=False,
               augmentation={},
               training=False,
               **kwargs):
    """
    Args:
      return a supervised sample
      (with flag of supervisory status to optional check , it should be always one)
      return metas: return sample filename along sample

      *len of loader is number of supervised frames!
    """

    super().__init__(dataset,
                     labelset,
                     do_augment=do_augment,
                     augmentation=augmentation,
                     training=training,
                     **kwargs)


  def __len__(self):
    return self.dataset.num_supervised_videos_frames


  def next_trainitem(self, item):
    return self.get_supervised_frame(item)


  def next_valitem(self, item):
    return self.next_trainitem(item)


  def get_supervised_frame(self, item):
    video_frame = self.dataset.list_supervised_videos_frames[item]

    video_id, frame = int(video_frame.split(':')[0]),int(video_frame.split(':')[-1])
    # get video object
    video = self.dataset.videos[video_id]
    return video, frame, frame in video._supervised_frames, video_id
