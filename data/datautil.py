import random
import numpy as np
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from PIL import Image
from scipy.spatial.transform import Rotation as t3d

import torch
import torchvision


# ######################################### #
#  CENTER & SCALE                           #
# ######################################### #
def get_center_scale(bb,
                     scale_factor=2.2):
  x, y, w, h = bb
  max_delta = max(max(w, h), 50.0)
  scale = min(max_delta * scale_factor, 500.0)

  c_x = x + w / 2.
  c_y = y + h / 2.

  center = np.asarray([c_x, c_y])
  return center, scale


def get_annot_scale(annots, visibility=None, scale_factor=2.2):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]

    if len(annots) != 0:
        min_x, min_y = annots.min(0)
        max_x, max_y = annots.max(0)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        max_delta = max(max(delta_x, delta_y), 50.0)
    else:
        max_delta = 50.0

    s = min(max_delta * scale_factor, 500.0)

    return s


def get_annot_center(annots, visibility=None):
    # Get scale
    if visibility is not None:
        annots = annots[visibility]

    if len(annots) != 0:
        min_x, min_y = annots.min(0)
        max_x, max_y = annots.max(0)
        c_x = int((max_x + min_x) / 2)
        c_y = int((max_y + min_y) / 2)
        center = np.asarray([c_x, c_y])
    else:
        center = np.asarray([0.0, 0.0])

    return center.astype(float)

# ####################################### #
#  AFFINE TRANSFORMATION                  #
# ####################################### #

def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.pinv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows


def transform_img(img, affine_trans, res):
    """
    Args:
    center (tuple): crop center coordinates
    scale (int): size in pixels of the final crop
    res (tuple): final image size
    """
    trans = np.linalg.pinv(affine_trans)

    img = img.transform(
        tuple(res),
        Image.AFFINE,
        (
            trans[0, 0],
            trans[0, 1],
            trans[0, 2],
            trans[1, 0],
            trans[1, 1],
            trans[1, 2],
        ),
    )
    return img


def sample_center_scale(
    scale, center, scale_jittering, center_jittering, rot=0
):
    """
    Randomly jitter center, scale and rotation
    Center is located in square of size 2*center_jitter_factor
    in center of cropped image
    """
    center_offsets = (
        center_jittering * scale * Uniform(low=-1, high=1).sample((2,)).numpy()
    )

    center = center + center_offsets.astype(int)

    # Scale jittering
    scale_jittering = scale_jittering * Normal(0, 1).sample().item() + 1
    scale_jittering = np.clip(
        scale_jittering, 1 - scale_jittering, 1 + scale_jittering
    )
    scale = scale * scale_jittering
    rot = Uniform(low=-rot, high=rot).sample().item()
    return scale, center, rot


def get_affine_transform(center, scale, res, rot=0.0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = (
        t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    )
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(
        transformed_center[:2], scale, res
    )
    return total_trans.astype(np.float32), post_rot_trans.astype(np.float32), rot_mat #(
        #total_trans.astype(np.float32),
        #affinetrans_post_rot.astype(np.float32),
    #)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    scale_ratio = float(res[0]) / float(res[1])
    affinet[0, 0] = float(res[0]) / scale
    affinet[1, 1] = float(res[1]) / scale * scale_ratio
    affinet[0, 2] = res[0] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[1] * (-float(center[1]) / scale * scale_ratio + 0.5)
    affinet[2, 2] = 1
    return affinet


def get_affine_transform_bak(center, scale, res, rot):
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / scale
    t[1, 1] = float(res[0]) / scale
    t[0, 2] = res[1] * (-float(center[0]) / scale + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / scale + 0.5)
    t[2, 2] = 1
    if rot != 0:
        rot_mat = np.zeros((3, 3))
        sn, cs = np.sin(rot), np.cos(rot)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t))).astype(np.float32)
    return t, t

# ###################################### #
# COLOR & JITTER                         #
# ###################################### #
def get_color_params(brightness=0,
                     contrast=0,
                     saturation=0,
                     hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def apply_jitter(img,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0):

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    jittered_img = apply_jitter(
        img,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    return jittered_img

# ################################### #
# NORMALIZE & DENORMALIZE             #
# ################################### #


def normalize(img):
    """
    :param img: RGB image in float
    :return: normalized image
    """
    img = img.astype(float)
    if img.max() > 1.:
        img /= 255.

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_nm = np.zeros_like(img, img.dtype)

    for c, (m, s) in enumerate(zip(mean, std)):
        img_nm[:, :, c] = (img[:, :, c] - m) / s

    return img_nm


def denormalize(img_nm):
    """
    :param img_nm: normalized image in float
    :return: unnormalized image in float in [0, 1]
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = np.zeros_like(img_nm, img_nm.dtype)

    for c, (m, s) in enumerate(zip(mean, std)):
        img[:, :, c] = img_nm[:, :, c] * s + m

    return img


def torch_denormalize(img_nm):
    """
    :param img_nm: normalized image in float
    :return: unnormalized image in float in [0, 1]
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = torch.zeros_like(img_nm) #b,c,H,W

    for c, (m, s) in enumerate(zip(mean, std)):
      img[:, c, :, :] = img_nm[:, c, :, :] * s + m

    return img.clamp(0,1)


# ###################################### #
# AXIS ANGLE 2 QUAT                      #
# ###################################### #


def rotvec2quat(poses):

  is_Tensor = False
  batch = True

  if isinstance(poses, torch.Tensor):
    is_Tensor =True
    dtype = poses.dtype
    device = poses.device
    poses = poses.numpy()

  if len(poses.shape) == 1:
    batch = False
    poses = poses[None]

  quats = np.concatenate([t3d.from_rotvec(poses[:, 3*i:3*i+3].copy()).as_quat() for i in range(poses.shape[1]//3)], axis=-1)

  if not batch:
    quats = quats[0]

  if is_Tensor:
    quats = torch.tensor(quats, dtype=dtype, device=device)

  return quats


def quat2rotvec(quats):

  is_Tensor = False
  batch = True

  if isinstance(quats, torch.Tensor):
    is_Tensor =True
    dtype = quats.dtype
    device = quats.device
    quats = quats.numpy()

  if len(quats.shape) == 1:
    batch = False
    quats = quats[None]

  poses = np.concatenate([t3d.from_quat(quats[:, 4*i:4*i+4].copy()).as_rotvec() for i in range(quats.shape[1]//4)], axis=-1)

  if not batch:
    poses = poses[0]

  if is_Tensor:
    poses = torch.tensor(poses, dtype=dtype, device=device)

  return poses

