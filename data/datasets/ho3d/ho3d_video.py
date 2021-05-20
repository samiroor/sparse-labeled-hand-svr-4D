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
from . import ho3d_utils

class HO3DVideo(Video):
  def __init__(self,
               video_dir,
               anns,
               obj_meshes=None,
               image_size=None,
               annot_metric="m",
               crop_bbox=False,
               crop_factor=1.5,
               supervise_fraction=1.,
               add_first_last_frame=True):
    """
    Args:
      video_dir (str): video directory.
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
    super(HO3DVideo, self).__init__(video_dir,
                                    num_frames=len(anns),
                                    image_size=image_size,
                                    annot_metric=annot_metric,
                                    crop_bbox=crop_bbox,
                                    crop_factor=crop_factor,
                                    supervise_fraction=supervise_fraction,
                                    add_first_last_frame=add_first_last_frame)


    self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    self.change_OpenGL_coords = True
    self.anns = anns

  def get_anno(self, frame):
    return self.anns[frame]

  def get_side(self, _):
    return 'right'

  def get_visibility(self, _, handside=None):
    if handside:
      assert handside == "right"
    return np.ones((21,), bool)

  def get_image_path(self, frame):
    image_path = os.path.join(self.video_dir, 'rgb', '{:04d}.png'.format(frame))
    return image_path

  def get_image(self, frame):
    image_path = self.get_image_path(frame)
    img = Image.open(image_path).convert("RGB")
    return img

  def get_mask(self, frame):
    pass

  def get_bb(self, frame):
    anno = self.get_anno(frame)
    if 'handBoundingBox' in anno.keys():
      bb = anno['handBoundingBox']
      return  np.array([bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]).astype(np.int32)
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

  def get_camintr(self, frame):
    anno = self.get_anno(frame)
    cam_mat = anno['camMat']
    return cam_mat.astype(np.float32)

  def get_joints3d(self, frame, handside=None):
    if handside:
      assert handside == "right"

    anno = self.get_anno(frame)

    if len(anno['handJoints3D'].shape) > 1:
      joints3d = anno['handJoints3D'][ho3d_utils.jointsMapManoToSimple]
    else:
      joints3d = np.expand_dims(anno['handJoints3D'],0)

    if self._annot_metric == 'mm':
      joints3d *= 1000.
    if self.change_OpenGL_coords:
      joints3d = joints3d.dot(self.coord_change_mat.T)

    return joints3d.astype(np.float32)

  def get_joints2d(self, frame, handside=None):
    if handside:
      assert handside == "right"

    handKps = ho3d_utils.project_3D_points(self.get_camintr(frame),
                                           self.get_joints3d(frame),
                                           is_OpenGL_coords=not self.change_OpenGL_coords)
    return handKps.astype(np.float32)

  def get_mano(self, frame, handside=None):
    if handside:
      assert handside == "right"

    anno = self.get_anno(frame)
    pose = anno['handPose']
    shape = anno['handBeta']
    if self.change_OpenGL_coords:
      pose[:3] = t3d.from_matrix(self.coord_change_mat.dot(t3d.from_rotvec(pose[:3].copy()).as_matrix())).as_rotvec()
    return pose, shape

  def get_verts3d(self, frame, handmodel, handside=None):
    if handside:
      assert handside == "right"

    pose, shape = self.get_mano(frame)
    if isinstance(handmodel,dict):
      handmodel = handmodel["right"]
    xyz_root = self.get_joints3d(frame)[handmodel.center_idx,:]
    xyz_rel, v_rel, F = handmodel(torch.from_numpy(pose[None]), torch.from_numpy(shape[None]))
    xyz = xyz_rel[0].numpy() + xyz_root
    v = v_rel[0].numpy() + xyz_root
    return xyz.astype(np.float32), v.astype(np.float32), F.numpy().astype(np.int32)

  # OBJECT Properties
  def get_obj_textures(self, frame):
    anno = self.get_anno(frame)
    obj_id = anno["objName"]
    textures = self.obj_meshes[obj_id]["textures"]
    return textures

  def get_obj_faces(self, frame):
    anno = self.get_anno(frame)
    obj_id = anno["objName"]
    objfaces = self.obj_meshes[obj_id]["faces"]
    return np.array(objfaces).astype(np.int32)

  def get_obj_verts3d_can(self, frame, no_center=False):
    anno = self.get_anno(frame)
    obj_id = anno["objName"]
    verts = self.obj_meshes[obj_id]["verts"]
    if self._annot_metric == 'mm':
      verts *= 1000.
    if self.change_OpenGL_coords:
      verts = verts.dot(self.coord_change_mat.T)
    if no_center:
      return verts, None, None
    else:
      return meshnorm.center_vert_bbox(verts, scale=False)

  def get_obj_verts3d(self, frame):
    anno = self.get_anno(frame)
    obj_id = anno["objName"]
    trans = anno["objTrans"]
    rot = cv2.Rodrigues(anno["objRot"])[0]
    verts = self.obj_meshes[obj_id]["verts"]
    verts_trans = rot.dot(verts.transpose()).transpose() + trans
    if self._annot_metric == 'mm':
      verts_trans *= 1000.
    if self.change_OpenGL_coords:
      verts_trans = verts_trans.dot(self.coord_change_mat.T)
    return np.array(verts_trans).astype(np.float32)

  def get_obj_verts2d(self, frame):
    verts = self.get_obj_verts3d(frame)
    camintr = self.get_camintr(frame)
    verts_hom2d = camintr.dot(verts.transpose()).transpose()
    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
    return verts_proj.astype(np.float32)

  def get_objvis(self, frame):
    objvis = np.ones((self.get_obj_verts2d(frame).shape[0] ,) ,bool)
    return objvis

  def get_obj_corners3d_can(self, frame):
    _, obj_cantrans, obj_canscale = self.get_obj_verts3d_can(frame)
    anno = self.get_anno(frame)
    corners = anno["objCorners3DRest"]
    if self._annot_metric == 'mm':
      corners *= 1000.
    if self.change_OpenGL_coords:
      corners = corners.dot(self.coord_change_mat.T)
    obj_cancorners = (corners - obj_cantrans) / obj_canscale
    return obj_cancorners

  def get_obj_corners3d(self, frame):
    anno = self.get_anno(frame)
    corners = anno["objCorners3DRest"]
    trans = anno["objTrans"]
    rot = cv2.Rodrigues(anno["objRot"])[0]
    corners_trans = rot.dot(corners.transpose()).transpose() + trans
    if self._annot_metric == "mm":
      corners_trans *= 1000
    if self.change_OpenGL_coords:
      corners_trans = corners_trans.dot(self.coord_change_mat.T)
    return  np.array(corners_trans).astype(np.float32)

  def get_obj_corners2d(self, frame):
    corners3d = self.get_obj_corners3d(frame)
    camintr = self.get_camintr(frame)
    corners_hom2d = camintr.dot(corners3d.transpose()).transpose()
    corners_proj = (corners_hom2d / corners_hom2d[:, 2:])[:, :2]
    return corners_proj.astype(np.float32)
