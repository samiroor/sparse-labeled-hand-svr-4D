import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as t3d

import torch

from ..video import Video
from ..meshutils import meshnorm
from . import interhand_utils



class InterHandVideo(Video):
  def __init__(self,
               video_dir,
               anns,
               campos,camrot,
               focal,princpt,
               video_joints,
               reorder_idx,
               video_mano_params,
               image_size=None,
               annot_metric="m",
               crop_bbox=False,
               crop_factor=1.5,
               supervise_fraction=1.,
               add_first_last_frame=True):
    """
    Args:
      video_dir (tuple str): (capture,seq_name, camera)
      anns (list of dict) :[ann0, ann1, ann2]}; ann is passed annotation of orderd frames in video.
      campos,camrot : camera extrinsic params specific for each video
      focal,princpt : camera intrinsic params specific for each video
      video_joints : annotations for world joints specific for each video
      joints_order (dict) : dataset skeleton order {"joint_name":joint_id}
      video_mano_params : mano params specific for each video
      annot_metric (str) :  "mm"|"m" desired metric of annotations.
      crop_bbox (bool) : cropping frames around the hand.
      crop_factor (float) : desired tightness of cropped frame around the hand annotations.
      supervise_fraction (float) : desired fraction of frames which have GT annotations available.
      add_first_last_frame (bool) : to include first-last frames in supervised set frames.

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
    super(InterHandVideo, self).__init__(video_dir,
                                         num_frames=len(anns),
                                         image_size=image_size,
                                         annot_metric=annot_metric,
                                         crop_bbox=crop_bbox,
                                         crop_factor=crop_factor,
                                         supervise_fraction=supervise_fraction,
                                         add_first_last_frame=add_first_last_frame)

    self.anns = anns
    self.campos = campos
    self.camrot = camrot
    self.focal = focal
    self.princpt = princpt
    self.video_joints = video_joints
    self.reorder_idx = reorder_idx
    self.video_mano_params = video_mano_params

    for ann in self.anns:
      assert ann['file_name'].split('/image')[0] in self.video_dir,"Wrong assigned annotations!"

  def split_arr(self, arr, handside):
    if handside == "right":
      return arr[:21]
    elif handside == "left":
      return arr[-21:]
    else:
      return arr

  def get_anno(self, frame):
    return self.anns[frame]

  def get_side(self, frame):
    ann = self.get_anno(frame)
    return ann['hand_type']

  def get_joints3dworld(self, frame):
    ann = self.get_anno(frame)
    joints_world = np.array(self.video_joints[str(ann['frame_idx'])]['world_coord'], dtype=np.float32)
    return joints_world

  def get_visibility(self, frame, handside=None):
    ann = self.get_anno(frame)
    visibility = np.array([vb[0] for vb in ann['joint_valid']], dtype=np.bool)[self.reorder_idx]
    return self.split_arr(visibility, handside)

  def get_image_path(self, frame):
    ann = self.get_anno(frame)
    image_path = os.path.join(self.video_dir,f'image{int(ann["frame_idx"]):04d}.jpg')
    return image_path

  def get_image(self, frame):
    image_path = self.get_image_path(frame)
    img = Image.open(image_path).convert("RGB")
    return img

  def get_mask(self, frame):
    pass

  def get_bb(self, frame): # minx,miny,w,h
    ann = self.get_anno(frame)
    return  np.array(ann['bbox']).astype(np.int32)

  def get_center_scale(self, frame):
    if self._crop_bbox:
      x, y, w, h = self.get_bb(frame)
      max_delta = max(max(w, h), 50.0)
      scale = min(max_delta * self._crop_factor, 500.0)
      center = np.array([x + w / 2., y + h / 2.])
    else:
      center = np.array(self._image_size) / 2.0
      scale = max(self._image_size)
    return center, scale

  def get_camintr(self, _):
    f = self.focal; c = self.princpt
    cam_mat = np.array([[f[0],0.,c[0]],[0.,f[1],c[1]],[0.,0.,1.]],dtype=np.float32)
    return cam_mat

  def get_joints3d(self, frame, handside=None):
    joints_world = self.get_joints3dworld(frame)
    joints3d = interhand_utils.world2cam(joints_world.transpose(1,0),
                                         self.camrot,
                                         self.campos.reshape(3,1)).transpose(1,0)[self.reorder_idx]
    if self._annot_metric == 'm':
      joints3d /= 1000.
    return self.split_arr(joints3d, handside).astype(np.float32)

  def get_joints2d(self, frame, handside=None):
    handKps = interhand_utils.cam2pixel(self.get_joints3d(frame,handside),
                                        self.focal,
                                        self.princpt)[:,:2]
    return np.array(handKps, dtype=np.float32)

  def get_mano(self, frame, handside=None):
    assert handside in ('left','right'), "Expected a handside"
    ann = self.get_anno(frame)
    mano_param = self.video_mano_params[str(ann['frame_idx'])][handside]
    pose = np.array(mano_param['pose'], dtype=np.float32)
    shape =np.array(mano_param['shape'], dtype=np.float32)
    pose[:3] = t3d.from_matrix(self.camrot.dot(t3d.from_rotvec(pose[:3].copy()).as_matrix())).as_rotvec()
    return pose,shape

  def get_verts3d(self, frame, handmodel, handside=None):
    pose, shape = self.get_mano(frame, handside)
    if isinstance(handmodel, dict):
      handmodel = handmodel[handside]
    xyz_root = self.get_joints3d(frame,handside)[handmodel.center_idx,:]
    xyz_rel, v_rel, F = handmodel(torch.from_numpy(pose[None]), torch.from_numpy(shape[None]))
    xyz = xyz_rel[0].numpy() + xyz_root
    v = v_rel[0].numpy() + xyz_root
    return xyz.astype(np.float32), v.astype(np.float32), F.numpy().astype(np.int32)
