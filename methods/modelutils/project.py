import torch
from manopth import rodrigues_layer



def recover_3d_proj(
    objpoints3d,
    camintr,
    est_scale,
    est_trans,
    annot_metric='m',
    off_z=0.4,
    input_res=(128, 128),
    ):
    """
    Given estimated centered points, camera intrinsics and predicted scale and translation
    in pixel world, compute the point coordinates in camera coordinate system
    """
    if annot_metric == 'mm' and off_z < 1:
      off_z *= 1000.0

    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d


def batch_proj2d(
    verts,
    camintr,
    camextr=None,
    ):
    # Project 3d vertices on image plane
    if camextr is not None:
        verts = camextr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d



def recover_object(canverts,
                   scaletrans,
                   camintr,
                   scale_factor,
                   trans_factor,
                   annot_metric=None,
                   input_res=None,
                   cancorners=None):

    """
    Args:
    Compute object vertex and corner positions in camera coordinates by predicting object translation
    and scaling, and recovering 3D positions given known object model
      canverts: (torch.Tensor) given known object model.
      scaletrans: (torch.Tensor) of shape [batch_size, channels] with channels == 6
        with in first position the predicted scale values and in 2,3 the 
        predicted translation values, and global rotation encoded as axis-angles
        in channel positions 4,5,6
      input_res (width,height): (tuple) width & height of input image
      scale_factor: (float)
      trans_factor: (float)
    """
    scale = scaletrans[:,:1]
    trans = scaletrans[:,1:3]
    rotaxisang = scaletrans[:,3:]
    batch_size = scaletrans.shape[0]
    # Get rotation matrixes from axis-angles
    rotmat = rodrigues_layer.batch_rodrigues(rotaxisang).view(rotaxisang.shape[0], 3, 3)
    rotverts = rotmat.bmm(canverts.float().transpose(1, 2)).transpose(1, 2)
    final_trans = trans.unsqueeze(1) * trans_factor
    final_scale = scale.view(batch_size, 1, 1) * scale_factor
    verts3d, center3d = recover_3d_proj(rotverts,
                                        camintr,
                                        final_scale,
                                        final_trans,
                                        annot_metric=annot_metric,
                                        input_res=input_res)
    # Recover 2D positions given camera intrinsic parameters and object vertex
    # coordinates in camera coordinate reference
    pred_verts2d = batch_proj2d(verts3d, camintr)
    obj_results = {
        'objverts3d': verts3d,
        'objverts2d': pred_verts2d,
        'objscale': final_scale,
        'objtrans': final_trans}

    # Corners
    if cancorners is not None:
      rotcorners = rotmat.bmm(cancorners.float().transpose(1, 2)).transpose(1, 2)
      corners3d = rotcorners + center3d
      pred_corners2d = batch_proj2d(corners3d, camintr)
      obj_results.update({
        'objcorners3d':corners3d,
        'objcorners2d': pred_corners2d,
        })
    return obj_results
