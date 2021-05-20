import os
import random
import numpy as np
from collections import defaultdict
import torch

from .standard_fields import Fields as Fls
from data import ShSeqDataset


"""
  FLOW Single Hand Seq dataset!
"""

class FlowShSeqDataset(ShSeqDataset):
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

      - kwargs['input_type']: input type to temporal & spatial encoders ,
      (could be image_seq or verts3d_seq)
      input to the velocity_net is always verts3d at a special time (t)
    """
    super().__init__(dataset,
                     res_size,
                     labelset,
                     nseq,
                     spacing,
                     augmentation,
                     do_augment=do_augment,
                     training=training,
                     **kwargs)
    from IPython import embed;embed()

  def __len__(self):
    return (self.dataset.num_videos_frames-1) // self.nseq + 1

  def __getitem__(self, item_):
    item = item_ * self.nseq

    video, frames, frames_status, video_id = self.next_trainitem(item) if self.training else self.next_valitem(item)
    samples, metas = self.get_samples(video, frames, frames_status, video_id)

    # time of seq [0 >> 1].
    idx = torch.randint(self.nseq, ())
    seq_time = np.arange(self.nseq).astype(np.float32) / (self.nseq-1)

    samples.update({
      Fls.seq_time : seq_time,
      Fls.t0_time : seq_time[0],
      Fls.t1_time : seq_time[idx],
      Fls.verts3d_t0 : samples[Fls.verts3d][0],
      Fls.verts3d_t1 : samples[Fls.verts3d][idx],
      })

    if self.return_metas:
      return samples,'#'.join(metas)
    return samples
