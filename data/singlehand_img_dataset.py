import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter
from collections import defaultdict
from scipy.spatial.transform import Rotation as t3d

import torch
import torchvision
from torch.utils.data import Dataset
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torchvision.transforms import functional as func_transforms

from . import datautil
from .standard_fields import Fields as Fls

"""
  Single Hand image dataset!
"""

class ShImgDataset(Dataset):
  def __init__(self,
               dataset,
               res_size,
               labelset,
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

      return a sample image (with flag of supervisory status to optional use)
      return metas: return sample filename along sample
    """

    # dataset
    self.dataset = dataset
    self.labelset = labelset
    self.aug = augmentation
    self.do_augment = do_augment
    self.training = training

    # property
    self._ref_handside = self.dataset._ref_handside
    self._root_joint = self.dataset._root_joint
    self._ref_bone = self.dataset._ref_bone

    self._bones = self.dataset._bones
    self._joints = self.dataset._joints
    self._annot_metric = self.dataset._annot_metric
    self._torch_hand_model = self.dataset._torch_hand_model[self._ref_handside]

    # desired Input Resolution
    if not self.dataset._crop_bbox:
      self.res_size = self.dataset._image_size
    else:
      self.res_size = res_size

    self.return_metas = False
    if "return_metas" in kwargs:
      self.return_metas = kwargs["return_metas"]


  def __len__(self):
    return self.dataset.num_videos_frames


  def __getitem__(self, item):
    video, frame, frame_status, video_id = self.next_trainitem(item) if self.training  else self.next_valitem(item)
    sample, meta = self.get_sample(video, frame, frame_status, video_id)
    sample.pop('aug')
    if self.return_metas:
      return sample,meta
    return sample


  def next_trainitem(self, item):
    video_frame = self.dataset.list_videos_frames[item]
    video_id, frame = int(video_frame.split(':')[0]),int(video_frame.split(':')[-1])
    # get video object
    video = self.dataset.videos[video_id]
    return video, frame, frame in video._supervised_frames, video_id


  def next_valitem(self, item):
    return self.next_trainitem(item)


  def get_sample(self, video, frame, frame_status, video_id, aug={}):
    sample = dict()
    sample['aug'] = dict()
    sample[Fls.supervised] = frame_status

    # annotations
    handside = video.get_side(frame)
    assert handside == self._ref_handside,"implementation works when annotation handside matches ref_handside!"

    img_path = video.get_image_path(frame)
    img = video.get_image(frame) if 'image' in self.labelset else None
    mask = video.get_mask(frame)  if 'mask' in self.labelset else None

    _,verts3d,faces = video.get_verts3d(frame, self._torch_hand_model, self._ref_handside) if 'verts3d' in self.labelset else None
    pose, shape = video.get_mano(frame, self._ref_handside) if 'mano_params' in self.labelset else (None, None)
    joints3d = video.get_joints3d(frame, self._ref_handside)
    joints2d = video.get_joints2d(frame, self._ref_handside)
    jointsvis = video.get_visibility(frame, self._ref_handside)

    camintr = video.get_camintr(frame)
    center, scale = video.get_center_scale(frame)

    # object annotations
    objfaces=None; objvis=None
    objverts3d=None; objverts3dcan=None; objverts2d=None
    objcorners3d=None; objcorners3dcan =None; objcorners2d=None
    if "obj" in self.labelset:
      objfaces = video.get_obj_faces(frame)
      objvis = video.get_objvis(frame)
      objverts3dcan, objcantrans, objcanscale = video.get_obj_verts3d_can(frame)
      objverts3d = video.get_obj_verts3d(frame)
      objverts2d = video.get_obj_verts2d(frame)
      sample[Fls.objfaces_prim] = objfaces
      sample[Fls.objverts3dcan_prim] = objverts3dcan
      if "obj_corners" in self.labelset:
        objcorners3dcan = video.get_obj_corners3d_can(frame)
        objcorners3d = video.get_obj_corners3d(frame)
        objcorners2d = video.get_obj_corners2d(frame)
        sample[Fls.objcorners3dcan_prim] = objcorners3dcan

    # flip or not
    flip = False; rot=0.0

    # augmentation
    if self.do_augment:
      img, aug = self.run_augment(img, scale, center, prev_aug=aug)
      center = aug['center']; scale = aug['scale']; rot = aug['rot']
      sample['aug'] = aug

    # affine Transformation
    affinetrans, post_rot_trans, rot_mat = datautil.get_affine_transform(center,
                                                                         scale,
                                                                         self.res_size,
                                                                         rot=rot)
    # transforming Image; Mask
    if img is not None:
      img = datautil.transform_img(img,
                                 affinetrans,
                                 self.res_size).crop((0,
                                                      0,
                                                      self.res_size[0],
                                                      self.res_size[1]))
    if mask is not None:
      mask = datautil.transform_img(mask,
                                    affinetrans,
                                    self.res_size).crop((0,
                                                         0,
                                                         self.res_size[0],
                                                         self.res_size[1]))
    # passing annotations
    sample[Fls.images_paths] = img_path
    sample[Fls.affinetrans] = affinetrans
    sample[Fls.joints2d_prim] = joints2d

    joints2d = datautil.transform_coords(joints2d, affinetrans)
    joints3d = rot_mat.dot(joints3d.transpose(1, 0)).transpose()
    if verts3d is not None:
      verts3d = rot_mat.dot(verts3d.transpose(1, 0)).transpose()
    camintr = rot_mat.dot(camintr.dot(rot_mat.transpose(1, 0)))
    camintr = post_rot_trans.dot(camintr)
    if pose is not None:
      pose[:3] = t3d.from_matrix(rot_mat.dot(t3d.from_rotvec(pose[:3].copy()).as_matrix())).as_rotvec()
    # visibility
    onplane = np.logical_and(joints2d >= 0, joints2d < self.res_size)
    self_visibility = np.logical_and(onplane[:, 0], onplane[:, 1]).astype(int)
    jointsvis = (jointsvis.astype(int) * self_visibility).astype(bool)

    # mask
    if mask is not None:
      mask = np.array(mask)

    # normalize Image
    if img is not None:
      img = datautil.normalize(np.array(img))
      img = func_transforms.to_tensor(img).float()
      sample[Fls.images] = img
    sample[Fls.handsides] = np.array((0,1)) if handside=="right" else np.array((1,0))
    sample[Fls.camsintr] = camintr
    sample[Fls.jointsvis] = jointsvis
    sample[Fls.joints2d] = joints2d
    sample[Fls.joints3d] = joints3d

    # to check camera
    #hom_2d = camintr.dot(joints3d.transpose()).transpose()
    #joints2d = (hom_2d/ hom_2d[:,2:])[:,:2]
    #sample[Fls.joints2d]= joints2d

    if mask is not None:
      sample[Fls.masks] = mask
    if verts3d is not None:
      sample[Fls.verts3d] = verts3d
      sample[Fls.faces] = faces
    if pose is not None:
      sample[Fls.mano_poses] = pose
      sample[Fls.mano_shapes] = shape
      sample[Fls.mano_poses_quat]= datautil.rotvec2quat(pose)
      sample[Fls.mano_params] = np.concatenate((pose, shape, joints3d[self._root_joint]), axis=-1)

    if objverts3d is not None:
      sample[Fls.objverts2d_prim] = objverts2d
      objverts2d = datautil.transform_coords(objverts2d, affinetrans)
      objverts3d = rot_mat.dot(objverts3d.transpose(1, 0)).transpose()
      sample[Fls.objverts2d] = objverts2d
      sample[Fls.objverts3d] = objverts3d

      onplane = np.logical_and(objverts2d >= 0, objverts2d < self.res_size)
      self_visibility = np.logical_and(onplane[:, 0], onplane[:, 1]).astype(int)
      objvis = (objvis.astype(int) * self_visibility).astype(bool)
      sample[Fls.objvis] = objvis

      if objcorners3d is not None:
        sample[Fls.objcorners2d_prim] = objcorners2d
        objcorners2d = datautil.transform_coords(objcorners2d, affinetrans)
        objcorners3d = rot_mat.dot(objcorners3d.transpose(1, 0)).transpose()
        sample[Fls.objcorners2d] = objcorners2d
        sample[Fls.objcorners3d] = objcorners3d

    if 'joints3d_rel' in self.labelset:
      root_joint = joints3d[self._root_joint]
      sample[Fls.joints3d_root] = root_joint[None]
      sample[Fls.joints3d_rel] = joints3d - root_joint
      if verts3d is not None:
        sample[Fls.verts3d_rel] = verts3d - root_joint

    if 'joints3d_nrm' in self.labelset:
      root_joint = joints3d[self._root_joint]
      ref_scale = np.linalg.norm(joints3d[self._ref_bone[0]] - joints3d[self._ref_bone[1]])
      sample[Fls.joints3d_root] = root_joint[None]
      sample[Fls.joints3d_scale] = ref_scale
      sample[Fls.joints3d_nrm] = (joints3d - root_joint) / ref_scale
      if verts3d is not None:
        sample[Fls.verts3d_nrm] = (verts3d - root_joint) / ref_scale

    meta = f'{video_id}:{frame}:{sample[Fls.images_paths]}'
    sample.pop(Fls.images_paths, None)

    if "image_features" in self.labelset:
      assert self.dataset.images_features is not None, "Not have been loaded images features!"
      assert not self.do_augment,"Saved images features do not have augmentation!"
      sample[Fls.images_features] = self.dataset.images_features[meta]

    return sample,meta


  def run_augment(self, img, scale, center, rot=0.0, prev_aug={}):
    # scale
    if self.aug['do_scale']:
      if self.aug['same_scale'] and 'scale' in prev_aug.keys():
        scale = prev_aug['scale']
      else:
        scale_jit = Normal(0, 1).sample().item() + 1
        scale_jittering = self.aug['scale_jitter'] * scale_jit
        scale_jittering = np.clip(scale_jittering, 1 - self.aug['scale_jitter'], 1 + self.aug['scale_jitter'])
        scale = scale * scale_jittering

    # rotation
    if self.aug['do_rot']:
      if self.aug['same_rot'] and 'rot' in prev_aug.keys():
        rot = prev_aug['rot']
      else:
        rot = Uniform(low=-self.aug['max_rot'], high=self.aug['max_rot']).sample().item()

    # center Jitter
    if self.aug['do_center']:
      if self.aug['same_center'] and 'center' in prev_aug.keys():
        center = prev_aug['center']
      else:
        center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
        center_offsets = self.aug['center_jitter'] * scale * center_jit
        center = center + center_offsets.astype(int)

    # color 
    if self.aug['do_color']:
      assert img is not None
      if self.aug['same_color'] and 'color' in prev_aug.keys():
        sat = prev_aug['color']['sat']
        contrast = prev_aug['color']['contrast']
        hue = prev_aug['color']['hue']
        bright = prev_aug['color']['bright']
        blur_radius = prev_aug['color']['blur_radius']
      else:
        blur_radius = Uniform(low=0, high=1).sample().item() * self.aug['color']['blur_radius']
        bright, contrast, sat, hue = datautil.get_color_params(brightness=self.aug['color']['brightness'],
                                                               saturation=self.aug['color']['saturation'],
                                                               hue=self.aug['color']['hue'],
                                                               contrast=self.aug['color']['contrast'])
      img = img.filter(ImageFilter.GaussianBlur(blur_radius))
      img = datautil.apply_jitter(img,
                                  brightness=bright,
                                  saturation=sat,
                                  hue=hue,
                                  contrast=contrast)
    # Save Aug
    curr_aug = {'scale':scale, 'rot':rot, 'center':center}
    if self.aug['do_color']:
      curr_aug.update({'color':{'blur_radius':blur_radius,
                                'sat':sat,
                                'bright':bright,
                                'contrast':contrast,
                                'hue':hue}
                                })
    return img, curr_aug


  def _mano_fn(self, poses, shapes=None, xyz_root=None, return_np=False):
    # tensor input
    if not isinstance(poses, torch.Tensor):
      poses = torch.tensor(poses)
      shapes = torch.tensor(shapes) if shapes is not None else None
      xyz_root = torch.tensor(xyz_root) if xyz_root is not None else None

    # works in batch & cpu & (#TODO Faces are not in batch)
    is_batch = True
    if len(poses.shape) == 1:
      is_batch = False
      poses = poses[None]
      shapes = shapes[None] if shapes is not None else None
      xyz_root = xyz_root[None] if xyz_root is not None else None

    device = poses.device
    dtype = poses.dtype
    xyz, v, F = self._torch_hand_model.to(device)(poses, shapes)
    if xyz_root is not None:
      xyz +=  xyz_root
      v += xyz_root

    if not is_batch:
      xyz = xyz.squeeze(0)
      v = v.squeeze(0)

    if return_np:
      _ = lambda x: x.detach().cpu().numpy()
    else:
      _ = lambda x: x.to(device=device, dtype=dtype)
    return _(xyz), _(v), _(F)


  def _verts2joints_fn(self, v, return_np=False):
    # tensor input
    if not isinstance(v, torch.Tensor):
      v = torch.tensor(v)

    # works in batch & cpu
    is_batch = True
    if len(v.shape) == 2:
      is_batch = False
      v = v[None]

    device = v.device
    dtype = v.dtype
    xyz, _ = self._torch_hand_model.to(device).verts2joints(v)

    if not is_batch: xyz = xyz.squeeze(0)

    if return_np:
      _ = lambda x: x.detach().cpu().numpy()
    else:
      _ = lambda x: x.to(device=device, dtype=dtype)
    return _(xyz)
