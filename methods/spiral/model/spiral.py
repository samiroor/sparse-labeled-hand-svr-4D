import torch
import torch.nn as nn

from manopth import rodrigues_layer

from methods.modelutils import project
from methods.modelutils.adaptor import ManoAdaptor, rec_freeze
from data import Fields as Fls


class Spiral(nn.Module):

  def __init__(
      self,
      encoder,
      spiral_base,
      spiral_decoder,
      R_head,
      st_head,
      obj_st_head,
      scale_factor=0.0001,trans_factor=100.0,
      return_spiral_features=False,
      return_encoder_features=False,
      use_adaptor=False,
      adaptor_path=None,
      annot_metric='m',
      center_idx=0,
      **kwargs):

    super().__init__()

    self.encoder = encoder
    self.spiral_base = spiral_base
    self.spiral_decoder = spiral_decoder

    self.R_head = R_head

    self.st_head = st_head
    self.obj_st_head = obj_st_head

    self.scale_factor = scale_factor
    self.trans_factor = trans_factor

    self.return_spiral_features = return_spiral_features
    self.return_encoder_features = return_encoder_features
    self.annot_metric = annot_metric
    self.center_idx = center_idx

    self.adaptor = None
    if use_adaptor:
      self.adaptor = ManoAdaptor(
          )
      rec_freeze(self.adaptor)


  def forward(self, input_batch):
    output_batch = None
    assert all(_ in input_batch for _ in (Fls.images, Fls.camsintr, ))

    # feature
    features = self.encoder(input_batch[Fls.images])
    H, W = tuple(input_batch[Fls.images].shape[2:])
    camsintr = input_batch[Fls.camsintr]

    # spiral
    spiral_features = self.spiral_base(features)
    verts3dcan = self.spiral_decoder(spiral_features)

    assert self.adaptor is not None
    # mano adaptor
    if self.adaptor:
      adapt_joints, _ = self.adaptor(verts3dcan)
      adapt_joints = adapt_joints.transpose(1, 2)
      final_joints = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
      final_verts  = verts3dcan - adapt_joints[:, self.center_idx].unsqueeze(1)
      verts3dcan=final_verts; joints3dcan=final_joints

    # rotation 
    rotaxisang = self.R_head(features)
    rotmat = rodrigues_layer.batch_rodrigues(rotaxisang).view(rotaxisang.shape[0], 3, 3)
    verts3d_rel = rotmat.bmm(verts3dcan.float().transpose(1, 2)).transpose(1, 2)
    joints3d_rel = rotmat.bmm(joints3dcan.float().transpose(1, 2)).transpose(1, 2)

    # output
    output_batch = {
        Fls.joints3d_rel: joints3d_rel,
        Fls.verts3d_rel: verts3d_rel,
        Fls.verts3dcan: verts3dcan,
        Fls.joints3dcan:joints3dcan,
        }

    # hand scale trans
    if self.st_head is not None:
      scaletrans = self.st_head(features)
      scale = scaletrans[:,:1] * self.scale_factor
      trans = scaletrans[:,1:] * self.trans_factor

      joints3d, center3d = project.recover_3d_proj(
          joints3d_rel,
          camsintr,
          scale,
          trans,
          annot_metric=self.annot_metric,
          input_res=(W, H),
          )

      verts3d = verts3d_rel + center3d
      verts2d = project.batch_proj2d(
          verts3d,
          camsintr,
          )
      joints2d = project.batch_proj2d(
          joints3d,
          camsintr,
          )

      # output
      output_batch.update({
        Fls.verts3d: verts3d,
        Fls.joints3d: joints3d,
        Fls.joints2d: joints2d,
        Fls.verts2d: verts2d,
        })

    # obj scale trans
    if self.obj_st_head is not None:
      assert all(_ in input_batch for _ in (Fls.objverts3dcan_prim,  ))
      objverts3dcan = input_batch[Fls.objverts3dcan_prim]
      obj_scaletrans = self.obj_st_head(features)
      obj_out_dict = project.recover_object(
          objverts3dcan,
          obj_scaletrans,
          camsintr,
          self.scale_factor,
          self.trans_factor,
          annot_metric=self.annot_metric,
          input_res=(W, H),
          cancorners=input_batch[Fls.objcorners3dcan_prim] if Fls.objcorners3dcan_prim in input_batch else None,
          )

      # output
      output_batch.update({
        Fls.objverts3d: obj_out_dict['objverts3d'],
        Fls.objverts2d: obj_out_dict['objverts2d'],
        Fls.objscale: obj_out_dict['objscale'],
        Fls.objtrans: obj_out_dict['objtrans'],
        })
      if 'objcorners3d' in obj_out_dict:
        output_batch.update({
          Fls.objcorners3d: obj_out_dict['objcorners3d'],
          Fls.objcorners2d: obj_out_dict['objcorners2d'],
          })

    # retur features
    if self.return_spiral_features:
      output_batch['spiral_features'] = spiral_features
    if self.return_encoder_features:
      output_batch['encoder_features'] = features

    return output_batch
