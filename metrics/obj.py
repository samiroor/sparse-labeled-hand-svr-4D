import numpy as np


def get_obj3d_error(
    verts_list=None,
    verts_pred_list=None,
    corners_list=None,
    corners_pred_list=None,
    annot_metric='m',
    ):

  # to show errors in "mm"
  if annot_metric == "mm":
    metric_factor = 1.0
  elif annot_metric == "m":
    metric_factor = 1000.0

  errors = {}
  if verts_list is not None:
    errors['objverts3d_mepe_trans (mm)'] = np.linalg.norm(verts_list - verts_pred_list, axis=-1).mean() * metric_factor
    errors['obj_samples_number'] = len(verts_list)

  if corners_list is not None:
    errors['objcorners3d_mepe_trans (mm)'] = np.linalg.norm(corners_list - corners_pred_list, axis=-1).mean() * metric_factor
    errors['obj_samples_number'] = len(corners_list)

  return errors


# @ original image resolution
def get_obj2d_error(
    verts_orig_list=None,
    verts_orig_pred_list=None,
    corners_orig_list=None,
    corners_orig_pred_list=None,
    ):

  errors = {}
  if verts_orig_list is not None:
    errors['objverts2d_mepe (px)'] = np.linalg.norm(verts_orig_list - verts_orig_pred_list, axis=-1).mean()

  if corners_list is not None:
    errors['objcorners2d_mepe (px)'] = np.linalg.norm(corners_orig_list - corners_orig_pred_list, axis=-1).mean()

  return errors


