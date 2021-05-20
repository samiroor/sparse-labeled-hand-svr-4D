import numpy as np

from .funcs import  *
from .eval_util import EvalUtil


# orig means @ original resolution of image
def get_hand2d_error(
    xy_orig_list,
    xy_orig_pred_list,
    vis_list=None,
    num_kp=21,
    ):

  eval_xy = EvalUtil(num_kp)

  # vis list is none
  if vis_list is None:
    vis_list = [ [1]* num_kp for xy in xy_orig_list ]

  for xy, xy_pred, vis in zip(xy_orig_list, xy_orig_pred_list, vis_list):
    eval_xy.feed(xy, vis, xy_pred)

  start, end, steps = 0., 100.0, 100
  epe_mean, epe_median, auc, pck, thresh, _ = eval_xy.get_measures(start, end, steps)

  return {
      "joints2d_epe_mean_px": epe_mean,
      "joints2d_epe_median_px": epe_median,
      "joints2d_auc": auc,
      "joints2d_pck": pck if self.report_pck else 0.0,
      "joints2d_thresholds": thresh if self.report_pck else 0.0,
      "samples_number":xy_orig_list.shape[0],
      }
