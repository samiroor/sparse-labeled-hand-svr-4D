import numpy as np

from .funcs import  *
from .eval_util import EvalUtil
from scipy.spatial.transform import Rotation as t3d



def get_hand3d_error(
    gt_file=None,
    pred_file=None,
    xyz_list=None,
    verts_list=None,
    xyz_pred_list=None,
    verts_pred_list=None,
    vis_list=None,
    output_dir="",
    annot_metric="m",
    ref_bone=[8,9],
    report_mesh_error=False):

  if gt_file is not None and pred_file is not None:
    # read "gt" from file
    with open(gt_file, 'r') as fi:
      gt = json.load(fi)
      xyz_list, verts_list = np.array(gt[0]), np.array(gt[1])
      vis_list = None
      if len(gt)==3:
          vis_list = np.array(gt[2])

    # read "prd" from file
    with open(pred_file, 'r') as fi:
      pred = json.load(fi)
      xyz_pred_list, verts_pred_list = np.array(pred[0]), np.array(pred[1])

  else:
    assert any(_ is not None for _ in [xyz_list, xyz_pred_list])

  errors = hand3d_error(
      xyz_list=xyz_list,
      verts_list=verts_list,
      vis_list=vis_list,
      xyz_pred_list=xyz_pred_list,
      verts_pred_list=verts_pred_list,
      output_dir=output_dir,
      annot_metric=annot_metric,
      ref_bone=ref_bone,
      report_mesh_error=report_mesh_error,
      )

  return errors



def hand3d_error(
    xyz_list,
    verts_list,
    vis_list,
    xyz_pred_list,
    verts_pred_list,
    output_dir='',
    annot_metric='m',
    ref_bone=[8,9],
    report_mesh_error=False,
    shape_is_mano=True,
    plot_pck=False,
    report_pck=False,
    num_kp=21,
    num_verts=778,
    ):

  errors = {}; curves = []

  # mesh error
  if report_mesh_error:
    errors_mesh, curves_mesh = mesh_error(
        verts_list=verts_list,
        verts_pred_list=verts_pred_list,
        annot_metric=annot_metric,
        shape_is_mano=shape_is_mano,
        num_kp=num_verts,
        report_pck=report_pck,
        )
    errors = errors_mesh; curves = curves_mesh

  # xyz error
  errors_xyz, curves_xyz = xyz_error(
      xyz_list=xyz_list,
      vis_list=vis_list,
      xyz_pred_list=xyz_pred_list,
      annot_metric=annot_metric,
      ref_bone=ref_bone,
      num_kp=num_kp,
      report_pck=report_pck,
      )
  errors.update(errors_xyz); curves += curves_xyz

  if plot_pck and report_pck:
    createHTML(
        output_dir,
        curves
        )

  return errors



def xyz_error(
    xyz_list,
    vis_list,
    xyz_pred_list,
    annot_metric='m',
    ref_bone=[8,9],
    num_kp=21,
    report_pck=False,
    report_error_list=False,
    ):

  # to show error in "mm"
  if annot_metric == "mm":
    metric_factor = 1.0
  elif annot_metric == "m":
    metric_factor = 1000.0

  # vis list is none
  if vis_list is None:
    vis_list = [ [1]* num_kp for xyz in xyz_list ]

  # init eval utils
  eval_xyz = EvalUtil(num_kp)
  eval_xyz_trans_al = EvalUtil(num_kp)
  eval_xyz_scale_trans_al = EvalUtil(num_kp)
  eval_xyz_scale_depth_al = EvalUtil(num_kp)
  eval_xyz_procrustes_al = EvalUtil(num_kp)
  eval_xyz_procrustes_woR_al = EvalUtil(num_kp)

  eval_list = ["xyz",
               "xyz_trans_al",
               #"xyz_scale_trans_al",
               "xyz_scale_depth_al",
               "xyz_procrustes_al",
               "xyz_procrustes_woR_al"]

  errors = dict()
  curves = []
  errors['samples_number'] = len(xyz_list)

  for item in eval_list:
    eval_item = eval('eval_{}'.format(item))

    for idx in range(len(xyz_list)):
      xyz, vis = xyz_list[idx], vis_list[idx]
      xyz, vis = [np.array(x).copy() for x in [xyz, vis]]
      xyz_pred = np.array(xyz_pred_list[idx]).copy()

      # Convert Metrics to "mm"
      xyz *= metric_factor
      xyz_pred *= metric_factor

      if item == "xyz":
        # Not aligned errors
        eval_item.feed(xyz, vis, xyz_pred)
      elif item == "xyz_trans_al":
        # trans align
        xyz_pred_tr_aligned = align_tr(xyz, xyz_pred)
        eval_item.feed(xyz, vis, xyz_pred_tr_aligned )
      elif item == "xyz_scale_trans_al":
        # scale & trans align
        xyz_pred_sc_tr_aligned = align_sc_tr(xyz, xyz_pred, ref_bone)
        eval_item.feed(xyz, vis, xyz_pred_sc_tr_aligned)
      elif item == "xyz_scale_depth_al":
        # scale & depth align
        xyz_pred_sc_depth_aligned = align_sc_depth(xyz, xyz_pred)
        eval_item.feed(xyz, vis, xyz_pred_sc_depth_aligned)
      elif item == "xyz_procrustes_woR_al":
        # procrustes align without R
        xyz_pred_aligned_woR = align_w_scale_woR(xyz, xyz_pred)
        eval_item.feed(xyz, vis, xyz_pred_aligned_woR)
      elif item == "xyz_procrustes_al":
        # procrustes align
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        eval_item.feed(xyz, vis, xyz_pred_aligned)

    # Get Statistics
    mean3d_list, mean3d, _, auc3d, pck, thresh, auc3d_sub = eval_item.get_measures(0.0, 50., 100)
    #errors["metric"] = "mm"
    if report_error_list:
      errors['{}_mean3d_list'.format(item)] = mean3d_list
    errors['{}_mean3d'.format(item)] = mean3d
    errors['{}_auc3d_sub'.format(item)] = auc3d_sub
    errors['{}_auc3d'.format(item)] = auc3d
    if report_pck:
      errors['pck_{}'.format(item)] = pck
      errors['thresh_{}(mm)'.format(item)] = thresh
      if plot_pck:
        curves.append(curve(thresh,
                            pck,
                            'Distance in mm',
                            'Percentage of correct keypoints',
                            'PCK curve for {} keypoint error'.format(item)))

  return errors, curves



def mesh_error(
    verts_list,
    verts_pred_list,
    annot_metric='m',
    shape_is_mano=True,
    num_kp=778,
    report_pck=False,
    ):

  # to show error in "mm"
  if annot_metric == "mm":
    metric_factor = 1.0
  elif annot_metric == "m":
    metric_factor = 1000.0

  errors = dict()
  curves = []
  errors['samples_number'] = len(verts_list)


  eval_mesh_err = EvalUtil(num_kp)
  eval_mesh_err_aligned = EvalUtil(num_kp)
  eval_list = ['', '_aligned']
  f_score, f_score_aligned = list(), list()
  f_threshs = [5., 15.]

  for item in eval_list:
    eval_item = eval('eval_mesh_err{}'.format(item))
    for idx in range(len(verts_list)):
      print(f'eval_mesh {item} - sample {idx}')
      verts = np.array(verts_list[idx]).copy()
      verts_pred = np.array(verts_pred_list[idx]).copy()
      # Convert Metrics to "mm"
      verts *= metric_factor
      verts_pred *= metric_factor
      if item == '':
        if shape_is_mano:
          eval_item.feed(
              verts,
              np.ones_like(verts[:, 0]),
              verts_pred)

        l=list()
        for t in f_threshs:
          # for each threshold calculate the f score and the f score of the aligned vertices
          f, _, _ = calculate_fscore(verts, verts_pred, t)
          l.append(f)
        f_score.append(l)
      elif item == "_aligned":
        if shape_is_mano:
          verts_pred_aligned = align_w_scale(verts, verts_pred)
        else:
          xyz = np.array(xyz_list[idx]).copy() * metric_factor
          xyz_pred = np.array(xyz_pred_list[idx]).copy() * metric_factor
          # use trafo estimated from keypoints
          trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
          verts_pred_aligned = align_by_trafo(verts_pred, trafo)

        if shape_is_mano:
          eval_item.feed(
              verts,
              np.ones_like(verts[:, 0]),
              verts_pred_aligned)

        # F-scores
        la = list()
        for t in f_threshs:
          # for each threshold calculate the f score and the f score of the aligned vertices
          f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
          la.append(f)
        f_score_aligned.append(la)


    mean3d_list, mean3d, _, auc3d, pck, thresh, _ = eval_item.get_measures(0.0, 50., 100)
    errors['mesh{}_mean3d'.format(item)] = mean3d
    errors['mesh{}_auc3d'.format(item)] = auc3d
    if report_pck:
      errors['pck_mesh{}'.format(item)] = pck
      errors['thresh_mesh{} (mm)'.format(item)] = thresh
      if plot_pck:
        curves.append(curve(thresh,
                            pck,
                           'Distance in mm',
                           'Percentage of correct vertices',
                           'PCV curve for mesh{} error'.format(item)))

  # F SCORE
  f_out = {}
  f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
  for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
    f_out.update({f'f_score_{round(t)}': f.mean()})
    f_out.update({f'f_al_score_{round(t)}': fa.mean()})
  errors.update(f_out)
  #f_out = list()
  #f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
  #for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
  #  f_out.append('f_score_%d: %f' % (round(t), f.mean()))
  #  f_out.append('f_al_score_%d: %f' % (round(t), fa.mean()))
  #errors['F scores'] = ['%s\n' % t for t in f_out]

  return errors, curves



def calc_error_vp_all(vp_file):

    with open(vp_file, 'r') as fi:
        vp = json.load(fi)

    assert len(vp) == 2, 'Expected format mismatch.'
    gt_list, pred_list = vp[0], vp[1]

    dd = []
    for vp_gt, vp_pred in  zip(gt_list, pred_list):
        R_gt =t3d.from_rotvec(vp_gt).as_matrix()
        R_pred =t3d.from_rotvec(vp_pred).as_matrix()
        dd.append(angle_distance(R_gt, R_pred))

    dd = np.array(dd)
    return {'mean_angle_distance': dd.mean(),
            'all_angle_distance':dd[:200]}

