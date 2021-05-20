import os
import numpy as np
import glob

class Videos:
  def __init__(self):
    # list of video objects 
    self.videos = []

    # list of ['video_id:frame_id',...]
    self.list_videos_frames = []
    self.list_supervised_videos_frames = []
    self.list_unsupervised_videos_frames = []

    self.num_videos = 0
    self.num_videos_frames = 0
    self.num_supervised_videos_frames = 0
    self.num_unsupervised_videos_frames = 0

    pass

  def video_dir2video_id(self, video_dir):
    video_ids = []
    for video_id, video in enumerate(self.videos):
      if video_dir in video.video_dir:
        video_ids.append(video_id)
    if len(video_ids) == 0:
      raise ValueError("Not such a video directory!")
    return np.array(video_ids)

  def video_dir2video(self,video_dir):
    return [self.videos[_] for _ in self.video_dir2video_id(video_dir)]

