import os
import cv2
import glob
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as t3d

import torch

from . import fphab_utils
from ..video import Video
from ..meshutils import meshnorm

class FPHABVideo(Video):
  def __init__(self,
               root_dir,
               video_dir,
               skel_info=None,
               mano_info=None,
               object_info=None,
               fhb_objects=None,
               reorder_idx=None,
               resolution_factor=0.25,
               image_prefix="Video_files_480",
               image_size=None,
               annot_metric="m",
               crop_bbox=False,
               crop_factor=1.5,
               supervise_fraction=1.,
               add_first_last_frame=True):
    """
    Args:
      root_dir (str)
      video_dir (tuple): (root_dir, subject, action_name, action_seq_idx, nframe).
      skel_info (dict)
      mano_info (dict)
      object_info (dict)
      fhb_objects (dict)
      resolution_factor (float): fphab images are so large, so we work on small version of images.
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
    super(FPHABVideo, self).__init__(video_dir,
                                     num_frames=len(skel_info),
                                     image_size=image_size,
                                     annot_metric=annot_metric,
                                     crop_bbox=crop_bbox,
                                     crop_factor=crop_factor,
                                     supervise_fraction=supervise_fraction,
                                     add_first_last_frame=add_first_last_frame)

    self.root_dir = root_dir
    self.video_dir = video_dir
    self.skel_info = skel_info
    self.mano_info = mano_info
    self.object_info = object_info
    self.fhb_objects = fhb_objects
    self.reorder_idx= reorder_idx
    self.image_prefix = image_prefix


    # camera
    self.camextr = np.array([
      [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
      [0, 0, 0, 1]])
    self.camintr = np.array([
      [1395.749023, 0, 935.732544],
      [0, 1395.749268, 540.681030],
      [0, 0, 1]])
    self.camintr[:2] = self.camintr[:2] * resolution_factor


  def get_side(self, _):
    return 'right'

  def get_visibility(self, _, handside=None):
    if handside:
      assert handside == "right"
    return np.ones((21,), bool)

  def get_image_path(self, frame):
    image_path = os.path.join(self.root_dir,
                              self.image_prefix,
                              self.video_dir,
                              "color",
                              "color_{:04d}.jpeg".format(frame))
    return image_path

  def get_image(self, frame):
    image_path = self.get_image_path(frame)
    img = Image.open(image_path).convert("RGB")
    return img

  def get_mask(self, frame):
    pass

  def get_bb(self, frame):
    annots = self.get_joints2d(frame)
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    return np.array([min_x, min_y, max_x - min_x, max_y - min_y]).astype(np.int32)

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
    return self.camintr.astype(np.float32)

  def get_skel_annots(self, frame):
    skel = self.skel_info[frame]
    return skel

  def get_joints3d(self, frame, handside=None):
    if handside:
      assert handside == "right"
    skel = self.get_skel_annots(frame)
    skel = skel[self.reorder_idx]
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    joints3d  = self.camextr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
    if self._annot_metric == 'm':
      joints3d /= 1000.
    return joints3d.astype(np.float32)

  def get_joints2d(self, frame, handside=None):
    if handside:
      assert handside == "right"
    joints3d = self.get_joints3d(frame)
    camintr = self.get_camintr(frame)
    hom_2d = camintr.dot(joints3d.transpose()).transpose()
    joints2d = (hom_2d / hom_2d[:, 2:])[:, :2]
    return joints2d.astype(np.float32)

  def get_mano_annots(self, frame):
    mano_params = self.mano_info["{:06d}.pkl".format(frame)]
    pose = mano_params["fullpose"]
    shape = mano_params["shape"]
    trans = mano_params["trans"]*1000. #mm
    return pose.astype(np.float32), shape.astype(np.float32), trans.astype(np.float32)

  def get_mano(self, frame, handside=None):
    if handside:
      assert handside == "right"
    pose, shape, _ = self.get_mano_annots(frame)
    return pose, shape

  def get_verts3d(self, frame, handmodel, handside=None):
    if handside:
      assert handside == "right"
    pose, shape = self.get_mano(frame)
    if isinstance(handmodel,dict):
      handmodel = handmodel['right']
    xyz_root = self.get_joints3d(frame)[handmodel.center_idx,:]
    xyz_rel, v_rel, F = handmodel(torch.from_numpy(pose[None]), torch.from_numpy(shape[None]))
    xyz = xyz_rel[0].numpy() + xyz_root
    v = v_rel[0].numpy() + xyz_root
    return xyz.astype(np.float32), v.astype(np.float32), F.numpy().astype(np.int32)

  # OBJECT Properties
  def get_obj_annots(self, frame):
    obj, trans = self.object_info[frame]
    return obj.astype(np.float32), trans.astyep(np.float32)

  def get_obj_textures(self, frame):
    obj,_ = self.get_obj_annots(frame)
    objtextures = self.fhb_objects[obj]["textures"]
    return np.array(objtextures).astype(np.float32)

  def get_obj_faces(self, frame):
    obj,_ = self.get_obj_annots(frame)
    objfaces = self.fhb_objects[obj]["faces"]
    return np.array(objfaces).astype(np.int32)

  def get_obj_verts3d_can(self, frame, no_center=False):
    obj,_ = self.get_obj_annots(frame)
    verts = self.fhb_objects[obj]["verts"] * 1000
    if self._annot_metric == "m":
      verts /= 1000.0
    if no_center:
      return verts, None, None
    else:
      return meshnorm.center_vert_bbox(verts, scale=False)

  def get_obj_verts3d(self, frame):
    obj, trans =  self.get_obj_annots(frame)
    verts = self.fhb_objects[obj]["verts"]* 1000.0
    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    verts_trans = trans.dot(hom_verts.T).T
    verts_camcoords = self.camextr.dot(verts_trans.transpose()).transpose()[:, :3]
    if self._annot_metric == "m":
      verts_camcoords /= 1000.0
    return np.array(verts_camcoords).astype(np.float32)

  def get_obj_verts2d(self, frame):
    verts = self.get_obj_verts3d(frame)
    verts_hom2d = self.camintr.dot(verts.transpose()).transpose()
    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
    return verts_proj.astype(np.float32)

  def get_objvis(self, frame):
    objvis = np.ones((self.get_obj_verts2d(frame).shape[0] ,) ,bool)
    return objvis
