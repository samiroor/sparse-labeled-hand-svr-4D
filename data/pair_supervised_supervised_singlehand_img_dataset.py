import os
import random
import numpy as np
from collections import defaultdict

import torch
from .supervised_any_singlehand_img_dataset import PairSupAnyShImgDataset


"""
Pair Supervised frame Supervised frame
Picking a Supervised frame along with an Supervised frame with at least gap distance
Single Hand Img dataset!

"""


class PairSupSupShImgDataset(PairSupAnyShImgDataset):
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
                     gap=gap,
                     do_augment=do_augment,
                     augmentation=augmentation,
                     training=training,
                     **kwargs)

    self.s_u_pick_ratio = 1.0
