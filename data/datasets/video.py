import os
import glob
import random
import numpy as np
from tqdm import tqdm

from collections import defaultdict

class Video:
  def __init__(self,
               video_dir,
               num_frames=0,
               image_size=None,
               annot_metric="m",
               crop_bbox=False,
               crop_factor=1.5,
               supervise_fraction=1,
               add_first_last_frame=True):
    """
    Args:
      video_dir (str or tuple): video specification.
      num_frames (int): number of frames in a video.
      hand_model (MANO model):  of ho3d.
      obj_meshes (dict): dictionary of object meshes.
      annot_metric (str):  "mm"|"m" desired metric of annotations.
      crop_bbox (bool): cropping frames around the hand.
      crop_factor (float): desired tightness of cropped frame around the hand annotations.
      supervise_fraction (float): desired fraction of frames which have GT annotations available.
      add_first_last_frame (bool): to include first-last frames in supervised set frames.

      # properties:
        _image_size (int, int)
        _joints (dict): "joint_name": joint_id
        _bones (list):  of pair connected joints
        _annot_metric (str)
        _crop_bbox (bool)
        _crop_factor (float)
        _num_frames (int) : number of frames in video
        _all_frames (list): list of frames ids
        _supervised_frames (list): list of supervised frames ids
        _unsupervised_frames (list): list of unsupervised frames ids
    """

    self.video_dir = video_dir

    # Supervised - Unsupervised frames
    self.supervise_fraction = supervise_fraction
    self.add_first_last_frame = add_first_last_frame

    # Specific to each dataset
    self._image_size = image_size
    self._annot_metric = annot_metric
    self._crop_bbox = crop_bbox
    self._crop_factor = crop_factor

    self._num_frames = num_frames
    self._all_frames = list(range(self._num_frames))
    self._supervised_frames = []
    self._unsupervised_frames = []

    if self.supervise_fraction < 1.0:
      spacing = int(1./self.supervise_fraction)
      for frame in self._all_frames:
        if self.is_strong(frame, spacing):
          self._supervised_frames.append(frame)
        else:
          self._unsupervised_frames.append(frame)
    else:
      self._supervised_frames = self._all_frames
    print("#Video: {} #Supervised frms: {} #UnSupervised frms: {}".format(self.video_dir,
                                                                        len(self._supervised_frames),
                                                                        len(self._unsupervised_frames)))

  def is_strong(self, frame, spacing):
    return (self.add_first_last_frame and (frame == 0 or frame == self._num_frames - 1)) or \
        (frame % spacing) == 0

  def has_more_frames(self, frame):
    return frame < self._num_frames - 1

  def get_next_frames(self, frame, n, spacing=1):
    """
      return "<=n  sequenced frames" after a specified "frame" (including "frame").
    """
    frames = []
    all_frames = self._all_frames
    while len(frames) < n:
      if frame > max(all_frames):
        break
      if frame in all_frames:
        frames.append(frame)
      frame += spacing
    return frames, frame

  def get_prior_frames(self, frame, n, spacing=1):
    """
      return "<=n  sequenced frames" ending with a specified "frame" (including "frame").
    """
    frames = []
    all_frames = self._all_frames
    while len(frames) < n:
      if frame < min(all_frames):
        break
      if frame in all_frames:
        frames.append(frame)
      frame -= spacing
    return frames, frame

  def get_next_unsupervised_frames(self, frame, n):
    """
      return "<=n seqquenced  Unsupervised frames" starting from a specific "frame" 
      (including frame if applicapable)!
    """
    frames = []
    unsupervised = self._unsupervised_frames
    while len(frames) < n:
      if frame > max(unsupervised):
        break
      if frame in unsupervised:
        frames.append(frame)
      frame += 1
    return frames, frame

  def get_prior_unsupervised_frames(self, frame, n):
    """
      return "<=n seqquenced  Unsupervised frames" ending with a specific "frame" 
      (including frame if applicapable)!
    """
    frames = []
    unsupervised = self._unsupervised_frames
    while len(frames) < n:
      if frame < min(unsupervised):
        break
      if frame in unsupervised:
        frames.append(frame)
      frame -= 1
    return frames, frame

  def get_intermediate_unsupervised_frames(self, frame1, frame2, n, spacing=None):
    """
      return "<=n seqquenced  Unsupervised equidistant frames between two specific "frames"
      (including specific frames if applicapable)!
    """
    unsupervised = self._unsupervised_frames
    flags = [(lambda id: frame1 <= id <= frame2)(id) for id in unsupervised]
    valid_unsupervised = np.array(unsupervised)[flags]

    if spacing is None:
      spacing = int((valid_unsupervised.max() - valid_unsupervised.min()) // n)

    frames = []
    frame = frame1
    while len(frames) < n :
      if frame > valid_unsupervised.max():
        break
      if frame in valid_unsupervised:
        frames.append(frame)
      frame += spacing
    return frames, frame

  def get_next_supervised_frames(self, frame, n):
    """
      return "<=n seqquenced  supervised frames" starting from a specific "frame"
      (including frame if applicapable)!
    """
    frames = []
    supervised = self._supervised_frames
    while len(frames) < n:
      if frame > max(supervised):
        break
      if frame in supervised:
        frames.append(frame)
      frame += 1
    return frames, frame

  def get_prior_supervised_frames(self, frame, n):
    """
      return "<=n seqquenced  supervised frames" ending with a specific "frame"
      (including frame if applicapable)!
    """
    frames = []
    supervised = self._supervised_frames
    while len(frames) < n:
      if frame < min(supervised):
        break
      if frame in supervised:
        frames.append(frame)
      frame -= 1
    return frames, frame

  def get_close_supervised_frames(self, frame, n):
    """
      return "<=n Close arbitrary Supervised frames" to a specified "frame"!
    """
    supervised = np.array(self._supervised_frames)
    d = np.abs(supervised - frame)
    frames = supervised[np.argsort(d)][:n]
    return list(frames)

  def get_data_dict(self, handmodel=None, ref_handside=None):
    summary = list()
    for frame in self._all_frames:
      _, verts3d, faces= self.get_verts3d(frame, handmodel, ref_handside)
      pose, shape = self.get_mano(frame, ref_handside)
      summary.append({
        'image_path': self.get_image_path(frame),
        'camintr' : self.get_camintr(frame),
        'joints3d': self.get_joints3d(frame, ref_handside),
        'joints2d': self.get_joints2d(frame, ref_handside),
        'handside':self.get_side(frame),
        'verts3d': verts3d,
        'faces': faces,
        'pose': pose,
        'shape':shape,
        'ref_handside':ref_handside,
        'video_dir':self.video_dir,
        'frame':frame})

    return summary

  def get_side(self, frame, **kwargs):
    """
    return hand side (str)
    """
    pass

  def get_visibility(self, frame, **kwargs):
    """
     return visiblity status of each joint [0,1,1,...,1] #shape:(num_joints, )
    """
    pass

  def get_image_path(self, frame):
    pass

  def get_image(self, frame):
    """
      return RGB image
    """
    pass

  def get_mask(self, frame):
    """
      return hand mask
    """
    pass

  def get_bb(self, frame):
    """
      return hand bbox(x,y,w,h)
    """
    pass

  def get_center_scale(self, frame):
    """
      return center of hand bbox ,cropping_scale max(bbox_w,bbox_h)*crop_factor
    """
    pass

  def get_camintr(self, frame):
    """
      return camera intrinsic of that frame
    """
    pass

  def get_joints3d(self, frame,**kwargs):
    pass

  def get_joints2d(self, frame,**kwargs):
    pass

  def get_mano(self, frame,**kwargs):
    """
      return pose, shape of that frame
    """
    pass

  def get_verts3d(self, frame, handmodel, **kwargs):
    """
      Args:
        pose; shape; roottrans of that frame
        return xyz, V, faces
    """
    pass

  # OBJECT Properties
  def get_obj_textures(self, frame):
    pass

  def get_obj_faces(self, frame):
    pass

  def get_obj_verts3d_can(self, frame, no_center=False):
    pass

  def get_obj_verts3d(self, frame):
    pass

  def get_obj_verts2d(self, frame):
    pass

  def get_objvis(self, frame):
    pass

  def get_obj_corners3d_can(self, frame):
    pass

  def get_obj_corners3d(self, frame):
    pass

  def get_obj_corners2d(self, frame):
    pass
