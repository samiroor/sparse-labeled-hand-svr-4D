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
               res_size,
               labelset,
               augmentation,
               do_augment=False,
               training=False,
               **kwargs):
    """
    Args:
      dataset (Data class): specific to each dataset.
      spacing (int): space between frames of seq.
      ref_bone (int, int): desired ref bone ids.
      root_joint (int): desired root joint id.
      res_size (int, int): desired image resolution.
      labelset (list): list of some names refer to the desired labels.
      augmentation (EasyDict): necessary augmentations.
      do_augment (bool):  do augmentation or not ...
      training (bool): do training or not ...
      ref_handside (str) left|right should be matched with annotation side!
    """
    super().__init__(dataset,
                     res_size,
                     labelset,
                     augmentation,
                     do_augment=do_augment,
                     training=training,
                     **kwargs)


  def __len__(self):
    return self.dataset.num_supervised_videos_frames


  def next_trainitem(self, item):
    video_frame = self.dataset.list_supervised_videos_frames[item]
    video_id, frame = int(video_frame.split(':')[0]),int(video_frame.split(':')[-1])
    # get video object
    video = self.dataset.videos[video_id]
    return video, frame, frame in video._supervised_frames, video_id


  def next_valitem(self, item):
    return self.next_trainitem(item)

