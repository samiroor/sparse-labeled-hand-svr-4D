import random

import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

import kornia
import kornia.geometry.conversions as kgc

from data import Fields as Fls


class TfEq(nn.Module):
  """
    Transformation Equivariance loss:
      TE loss penalizes the difference between output
      of a model on the transformed image and
      transformed models output of the original image.
  """

  def __init__(
      self,
      batch_size=8,
      input_hw=(256,256),
      max_angle=1.57,
      scale_jitter=0.01,
      center_jitter=0.1,
      transform_type='rotation',
      device='cpu',
      ):

    super().__init__()
    self.batch_size = batch_size
    self.input_hw = input_hw

    self.max_angle = max_angle
    self.scale_jitter = scale_jitter
    self.center_jitter = center_jitter
    self.transform_type = transform_type
    self.device = device

  def set_transform_mats(
      self,
      ):

    """
     set transformation matrix:
       self.affinetrans would be used for transforming image & 2D annotations!
       self.rot_mat would be used for transforming 3D annotations!
       self.post_rot_trans is scale-trans transformation applied on rotated center of image used for transforming camera intrinsic.
    """
    batch_size = self.batch_size
    input_hw = self.input_hw

    # scale
    scale = torch.ones(batch_size, ) * max(input_hw)
    if 'scale' in self.transform_type:
      scale_jitter = (Normal(0, 1).sample((batch_size,)) + 1.) * self.scale_jitter
      sacle_jitter = sacle_jitter.clamp(1 - self.scale_jitter, 1 + self.scale_jitter)
      scale *= scale_jitter

    # rotation center
    center = torch.ones(batch_size, 2)
    center[..., 0] = input_hw[1] / 2  # x
    center[..., 1] = input_hw[0] / 2  # y
    if 'trans' in self.transform_type:
      center_jitter = Uniform(-1,1).sample((batch_size, 2)) * scale * self.center_jitter
      center += center_jitter

    # rotation
    rot = torch.zeros(batch_size, )
    if 'rotation' in self.transform_type:
      rot = Uniform(-self.max_angle, self.max_angle).sample((batch_size,))

    # get transformation
    affinetrans, post_rot_trans, rot_mat = get_affine_transform(
        center,
        scale,
        rot,
        input_hw,
        )

    self.affinetrans = affinetrans.to(self.device)
    self.post_rot_trans = post_rot_trans.to(self.device)
    self.rot_mat = rot_mat.to(self.device)



  def transform(
      self,
      input_batch):
    """
      would augment all items availbale in input_batch via applyinig transformation matrices!
    """
    output_batch = input_batch.copy()

    if Fls.images in input_batch:
      output_batch[Fls.images] = transform_img(
          input_batch[Fls.images],
          self.affinetrans,
          self.input_hw,
          )

    if Fls.joints2d in input_batch:
      output_batch[Fls.joints2d] = transform_coords(
          input_batch[Fls.joints2d],
          self.affinetrans,
          )

    for _ in [
        Fls.joints3d,
        Fls.joints3d_rel,
        Fls.joints3d_nrm,
        Fls.verts3d,
        Fls.verts3d_rel,
        Fls.verts3d_nrm,
        ]:
      if _ in input_batch:
        output_batch[_] = (self.rot_mat @ input_batch[_].transpose(2, 1)).transpose(2, 1)

    if Fls.camsintr in input_batch:
      camintr = self.rot_mat @ input_batch[Fls.camsintr] @ self.rot_mat.transpose(2, 1)
      output_batch[Fls.camsintr] = self.post_rot_trans @ camintr

    for _ in [
        Fls.mano_poses,
        Fls.mano_poses_quat,
        Fls.mano_params,
        ]:
      if _ in input_batch:
        if _ == Fls.mano_poses_quat:
          vp = input_batch[_][:, :4]
          vp_tf = kgc.rotation_matrix_to_quaternion(
              self.rot_mat @ kgc.quaternion_to_rotation_matrix(vp)
              )
          output_batch[_][:, :4] = vp_tf

        else:
          vp = input_batch[_][:, :3]
          vp_tf = kgc.rotation_matrix_to_angle_axis(
              self.rot_mat @ kgc.angle_axis_to_rotation_matrix(vp)
              )
          output_batch[_][:, :3] = vp_tf

    return output_batch


def get_affine_transform(
    center,
    scale,
    rot,
    input_hw,
    ):

  """
  wriiten similar to datautils!
   Args:
    *center (float torch.tensor) : random center of rotation in image of shape (batch_size, 2)
    *scale (float torch.tensor): random scale of image in shape (batch_size, )
    *rot (float torch.tensor): random rotation angle of shape (batch_size, )
    *input_hw (float, float): resolution of images

   Return:

     *total_trans (folat torch.tensor) : final affine transformation of shape (batch_size, 3, 3);
      applied on image and 2D annotations.
     *post_rot_trans (float torch.tensor): scale-trans transformation on rotated center of shape (batch_size, 3, 3)
      applied on camera intrinsic (transformation of scale-trans (without rotation))
     *rot_mat (float torch.tensor): rotation matrix of shape (batch_size, 3, 3)
      applied on 3D annotations

 """
  batch_size = rot.shape[0]
  rot_mat = torch.zeros((batch_size, 3, 3))
  sn, cs = torch.sin(rot), torch.cos(rot)
  rot_mat[:, 0, 0] = cs
  rot_mat[:, 0, 1] = -sn
  rot_mat[:, 1, 0] = sn
  rot_mat[:, 1, 1] = cs
  rot_mat[:, 2, 2] = 1.

  # Rotate center to obtain coordinate of center in rotated image
  c_ = torch.cat([center, torch.ones(batch_size, 1)], dim=-1)
  origin_rot_center = (rot_mat @ c_.unsqueeze(-1))[:, :2].squeeze()

  t_mat = torch.eye(3)[None].repeat(batch_size, 1, 1)
  t_mat[:, 0, 2] = -input_hw[1] / 2
  t_mat[:, 1, 2] = -input_hw[0] / 2
  t_inv = t_mat.clone()
  t_inv[:, :2, 2] *= -1

  #transformed_center = t_inv @ rot_mat @ t_mat @ c_.unsqueeze(-1)[:, :2].squeeze()

  post_rot_trans = get_affine_trans_no_rot(
      origin_rot_center,
      scale,
      input_hw,
      )

  total_trans = post_rot_trans @ rot_mat

  #affinetrans_post_rot = get_affine_trans_no_rot(
  #    transformed_center,
  #    scale,
  #    input_hw,
  #    )

  return total_trans, post_rot_trans, rot_mat



def get_affine_trans_no_rot(
    center,
    scale,
    res,
    ):

  """
  Args:
    center (float torch.tensor): (batch_size, 2)
    scale  (float torch.tensor): (batch_size,)
    res (float, float): resolution

  Return :
    affinet : affine transformation of shape (batch_size, 3, 3)
  """

  affinet = torch.zeros((center.shape[0], 3, 3))

  scale_ratio = float(res[0]) / float(res[1])
  affinet[:, 0, 0] = float(res[0]) / scale
  affinet[:, 1, 1] = float(res[1]) / scale * scale_ratio
  affinet[:, 0, 2] = res[0] * (-center[:, 0] / scale + 0.5)
  affinet[:, 1, 2] = res[1] * (-center[:, 1] / scale * scale_ratio + 0.5)
  affinet[:, 2, 2] = 1
  return affinet



def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts (float torch.tensor): (batch_size, point_nb, 2)
    """
    batch_size = pts.shape[0]
    np = pts.shape[1]

    if invert:
        affine_trans = affine_trans.pinverse()
    hom2d = torch.cat([pts, torch.ones(batch_size, np, 1, device=pts.device)], dim=-1)
    transformed_rows = (affine_trans @ hom2d.transpose(2,1)).transpose(2,1) [:, :, :2]
    return transformed_rows



def transform_img(img, affine_trans, res):
  """
    img [b, c, h, w]
  """
  with torch.no_grad():
    img_tf = kornia.warp_affine(
      img,
      affine_trans[:, :2, :],
      dsize=res,
      )
  return img_tf
