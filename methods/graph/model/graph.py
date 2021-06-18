import torch
import torch.nn as nn

from methods.modelutils import project
from methods.modelutils.adaptor import ManoAdaptor, rec_freeze
from data import Fields as Fls


class Graph(nn.Module):

  def __init__(
      self,
      encoder,
      graph_base,
      graph_head,
      st_head,
      obj_st_head,
      adaptor_path=None,
      return_tail_features=False,
      return_encoder_features=False,
      return_encoder_spatial_features=False,
      return_encoder_tail_features=False,
      center_idx=0,
      annot_metric='m',
      scale_factor=0.0001,
      trans_factor=100.0,
      off_z=0.4,
      **kwargs):

    super().__init__()

    self.encoder = encoder

    self.graph_base = graph_base
    self.graph_head = graph_head

    self.st_head = st_head
    self.obj_st_head = obj_st_head

    self.return_tail_features = return_tail_features
    self.return_encoder_features = return_encoder_features
    self.return_encoder_spatial_features = return_encoder_spatial_features
    self.return_encoder_tail_features = return_encoder_tail_features

    self.center_idx = center_idx
    self.annot_metric = annot_metric #TODO reference mesh file should be in that
    self.scale_factor = scale_factor
    self.trans_factor = trans_factor
    self.off_z = off_z

    self.adaptor_path = adaptor_path
    self.adaptor = ManoAdaptor(load_path=self.adaptor_path)
    rec_freeze(self.adaptor)


  def forward(self, input_batch):
    output_batch = None
    assert all(_ in input_batch for _ in (Fls.images, Fls.camsintr, ))

    # feature
    if self.encoder.return_spatial_features:
      features, spatial_features = self.encoder(input_batch[Fls.images])
    else:
      features = self.encoder(input_batch[Fls.images])
      spatial_features = None

    H, W = tuple(input_batch[Fls.images].shape[2:])
    camsintr = input_batch[Fls.camsintr]

    # graph
    graph_features = self.graph_base(features)
    verts3d_rel = self.graph_head(graph_features)

    assert self.adaptor is not None, 'Must have adaptor to extract joints'
    # mano adaptor
    if self.adaptor:
      adapt_joints, _ = self.adaptor(verts3d_rel)
      adapt_joints = adapt_joints.transpose(1, 2)
      final_joints = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
      final_verts  = verts3d_rel - adapt_joints[:, self.center_idx].unsqueeze(1)
      verts3d_rel=final_verts; joints3d_rel=final_joints

    # output
    output_batch = {
        Fls.joints3d_rel: joints3d_rel,
        Fls.verts3d_rel: verts3d_rel,
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
          off_z=self.off_z,
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
          off_z=self.off_z,
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
    if self.return_tail_features:
      output_batch[Fls.tail_features] = graph_features

    if self.return_encoder_features:
      output_batch[Fls.encoder_features] = features

    if self.return_encoder_spatial_features:
      assert spatial_features is not None
      output_batch[Fls.encoder_spatial_features] = spatial_features

    if self.return_encoder_tail_features:
      output_batch[Fls.encoder_tail_features] = torch.cat([features, graph_features], dim=-1)

    return output_batch
