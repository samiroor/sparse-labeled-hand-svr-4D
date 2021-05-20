import os
import random
from collections import defaultdict

import torch
from .singlehand_img_dataset import ShImgDataset


"""
  Single Hand Seq dataset!
"""

class ShSeqDataset(ShImgDataset):
  def __init__(self,
               dataset,
               res_size,
               labelset,
               nseq,
               spacing,
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
      nseq (int): number of frames in given seq.
      spacing (int)|tuple: space between selected frames.

      return a sequenced samples (traverse on ALL frames with equal spacing) from a video.
      (supervisory flag would be optional to use)
      (all frames in seq might randomly be supervised or not or mixed  according to supervisory fraction & nseq(seq length)!))

    """
    super().__init__(dataset,
                     res_size,
                     labelset,
                     augmentation,
                     do_augment=do_augment,
                     training=training,
                     **kwargs)

    self.nseq = nseq
    self.spacing = spacing


  def __len__(self):
    return (self.dataset.num_videos_frames-1) // self.nseq + 1

  def __getitem__(self, item_):
    item = item_* self.nseq

    video, frames, frames_status, video_id = self.next_trainitem(item) if self.training else self.next_valitem(item)

    samples, metas = self.get_samples(video, frames, frames_status, video_id)
    if self.return_metas:
      return samples,'#'.join(metas)
    return samples

  def next_trainitem(self, item):
    video_frame = self.dataset.list_videos_frames[item]
    video_id, frame = int(video_frame.split(':')[0]),int(video_frame.split(':')[-1])
    # get video object
    video = self.dataset.videos[video_id]
    frames, frames_status = self.select_frames(video, frame)
    return video, frames, frames_status, video_id

  def next_valitem(self, item):
    return self.next_trainitem(item)

  def select_frames(self, video, frame):
    spacing = self.spacing if isinstance(self.spacing, int) else random.randint(self.spacing[0],self.spacing[1])

    frames, _ = video.get_next_frames(frame, self.nseq, spacing)
    if len(frames) < self.nseq:
      pad,_ = video.get_prior_frames(frame-1, self.nseq - len(frames), spacing)
      frames.extend(pad)

    frames = list(set(frames))
    frames.sort()
    status = lambda frm: int(frm in video._supervised_frames)
    return frames, [status(frm) for frm in frames]

  def get_samples(self, video, frames, frames_status, video_id):
    samples = list()
    metas = list()
    aug = dict()
    for frame, frame_status in zip(frames, frames_status):
      sample, meta = self.get_sample(video, frame, frame_status, video_id, aug)
      aug = sample.pop('aug', None)
      samples.append(sample)
      metas.append(meta)
    samples = self.merge_samples(samples)
    return samples, metas

  def merge_samples(self, samples):
    mrg_smpl = defaultdict(list)
    for key in samples[0].keys():
      for smpl in samples:
        mrg_smpl[key].append(torch.Tensor(smpl[key]))
      # merge
      mrg_smpl[key] = torch.stack(mrg_smpl[key])
    return mrg_smpl
