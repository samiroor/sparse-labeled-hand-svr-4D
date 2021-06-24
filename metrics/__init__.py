from .hand3d import get_hand3d_error
from .hand2d import get_hand2d_error
from .obj import get_obj3d_error, get_obj2d_error


from data import Fields as Fls



# kwargs


def hand3d_kwargs(gt_dict, pred_dict):
  kwargs = {}
  if Fls.joints3d in pred_dict:
    kwargs.update({
      'xyz_list': gt_dict[Fls.joints3d].detach().cpu().numpy() ,
      'xyz_pred_list': pred_dict[Fls.joints3d].detach().cpu().numpy(),
      })
  if Fls.verts3d in pred_dict and Fls.verts3d in gt_dict:
    kwargs.update({
      'verts_list': gt_dict[Fls.verts3d].detach().cpu().numpy(),
      'verts_pred_list': pred_dict[Fls.verts3d].detach().cpu().numpy(),
      'report_mesh_error':False,#TODO so slooooooow should be in cuda
      })
  return kwargs


def hand2d_kwargs(gt_dict, pred_dict):
  kwargs = {}
  if Fls.joints2d in pred_dict:
    assert any(_ in gt_dict for _ in (Fls.affinetrans, Fls.joints2d_prim))
    kwargs.update({
      'xy_orig_list': gt_dict[Fls.joints2d_prim].detach().cpu().numpy(),
      'xy_orig_pred_list': recover_back(pred_dict[Fls.joints2d], gt_dict[Fls.affinetrans]).detach().cpu().numpy(),
        })
  return kwargs


def obj3d_kwargs(gt_dict, pred_dict):
  kwargs = {}
  if Fls.objverts3d in pred_dict:
    kwargs.update({
      'verts_list': gt_dict[Fls.objverts3d].detach().cpu().numpy(),
      'verts_pred_list': pred_dict[Fls.objverts3d].detach().cpu().numpy(),
      })

  if Fls.objcorners3d in pred_dict:
    kwargs.update({
        'corners_list': gt_dict[Fls.objcorners3d].detach().cpu().numpy(),
        'corners_pred_list': pred_dict[Fls.objcorners3d].detach().cpu().numpy(),
        })
  return kwargs


def obj2d_kwargs(gt_dict, pred_dict):
  kwargs = {}
  if Fls.objverts2d in pred_dict:
    assert any(_ in gt_dict for _ in (Fls.affinetrans, Fls.objverts2d_prim))
    kwargs.update({
      'verts_orig_list': gt_dict[Fls.objverts2d_prim].detach().cpu().numpy(),
      'verts_orig_pred_list': recover_back(pred_dict[Fls.objverts2d], gt_dict[Fls.affinetrans]).detach().cpu().numpy(),
        })

  if Fls.objcorners2d in pred_dict:
    assert any(_ in gt_dict for _ in (Fls.affinetrans, Fls.objcorners2d_prim))
    kwargs.update({
      'corners_orig_list': gt_dict[Fls.objcorners2d_prim].detach().cpu().numpy(),
      'corners_orig_pred_list': recover_back(pred_dict[Fls.objcorners2d], gt_dict[Fls.affinetrans]).detach().cpu().numpy(),
      })
  return kwargs



def recover_back(joints_trans, affinetrans):
    """
    Given 2d point coordinates and an affine transform, recovers original pixel points
    (locations before translation, rotation, crop, scaling... are applied during data
    augmentation)
    """
    batch_size = joints_trans.shape[0]
    point_nb = joints_trans.shape[1]
    hom2d = torch.cat([joints_trans, joints_trans.new_ones(batch_size, point_nb, 1)], -1)
    rec2d = torch.inverse(affinetrans).bmm(hom2d.transpose(1, 2).float()).transpose(1, 2)[:, :, :2]
    return rec2d




# get metrics fns


# dicts
metrics_dict = {
    'hand3d':get_hand3d_error,
    'hand2d':get_hand2d_error,
    'obj3d':get_obj3d_error,
    'obj2d': get_obj2d_error,
    }

kwargs_dict = {
    'hand3d':hand3d_kwargs,
    'hand2d':hand2d_kwargs,
    'obj3d':obj3d_kwargs,
    'obj2d':obj2d_kwargs,
    }


def get_kwargs(
    gt_dict,
    pred_dict,
    name='hand3d',
    ):
  return kwargs_dict[name](gt_dict=gt_dict, pred_dict=pred_dict)



def get_metric_fn(name, **kwargs):
  return lambda gt_dict, pred_dict: metrics_dict[name](
      **get_kwargs(
        gt_dict,
        pred_dict,
        name,
        ) if not bool(kwargs) else kwargs)


def get_metrics_fns(names):#TODO, **kwargs):
  fns = []
  for name in names:
    fns.append(get_metric_fn(name))
  return fns
