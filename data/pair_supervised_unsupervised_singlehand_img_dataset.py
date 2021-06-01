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

    self.gap = gap # for eval it is not a tuple!

  def __getitem__(self, item):

    if self.training:
      video, supervised_frame, unsupervised_frame, video_id = self.next_trainitem(item)
    else:
      video, supervised_frame, unsupervised_frame, video_id = self.next_valitem(item)

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

    gap = self.gap if isinstance(self.gap, int) else random.randint(self.gap[0], self.gap[1])

    unsupervised_frame, frame_status = self.pass_unsupervised_frame(
        video,
        supervised_frame,
        gap=gap,
        first_next=random.random() > 0.5,
        )

    assert frame_status == 0,'Expected to be picked from unsupervised collection.'
    return video, supervised_frame, unsupervised_frame, video_id


  def next_valitem(self, item):
    video, supervised_frame, frame_status, video_id = self.get_supervised_frame(item)
    assert frame_status == 1,'Expected to be picked from supervised collection.'

    # set seed
    random.seed(item)
    gap = self.gap if isinstance(self.gap, int) else random.randint(self.gap[0], self.gap[1])

    random.seed(item)
    unsupervised_frame, frame_status = self.pass_unsupervised_frame(
        video,
        supervised_frame,
        gap=gap,
        first_next=random.random() > 0.5,
        )

    assert frame_status == 0,'Expected to be picked from unsupervised collection.'
    return video, supervised_frame, unsupervised_frame, video_id


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
