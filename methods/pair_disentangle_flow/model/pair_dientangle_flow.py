import torch
import torch.nn as nn


from data import Fields as Fls

def get_key(key):
  if key == "verts3d":
    return Fls.verts3d
  elif key =="verts3d_rel":
    return Fls.verts3d_rel
  elif key == "mano_params":
    return Fls.mano_params
  elif key == "encoder_features":
    return Fls.encoder_features
  elif key == "tail_features":
    return Fls.tail_features
  elif key == "encoder_spatial_features":
    return Fls.encoder_spatial_features
  elif  key == "encoder_tail_features":
    return Fls.encoder_tail_features
  else:
    raise NotImplementedError()


class PairDistFlow(nn.Module):

  def __init__(
      self,
      mesh_predictor,
      context_head_root,
      context_head_rel,
      flow_root,
      flow_rel,
      morph_key="mano_params",
      feature_key_root='encoder_features',
      feature_key_rel='tail_features',
      morph_shape=False,
      use_gt_mano_shape=False,
      invert_from_gts=False,
      infer_from_gts=True,
      **kwargs,
      ):

    super().__init__()

    self.mesh_predictor = mesh_predictor

    self.context_head_root = context_head_root
    self.context_head_rel = context_head_rel

    self.flow_root = flow_root
    self.flow_rel = flow_rel

    self.morph_key = get_key(morph_key)
    self.feature_key_root = get_key(feature_key_root)
    self.feature_key_rel = get_key(feature_key_rel)

    self.morph_shape = morph_shape
    self.use_gt_mano_shape = use_gt_mano_shape
    assert not all([morph_shape, use_gt_mano_shape]),\
        "Could not be applicable together!"

    self.invert_from_gts = invert_from_gts
    self.infer_from_gts = infer_from_gts

    self.center_idx = self.mesh_predictor.center_idx
    self.annot_metric = self.mesh_predictor.annot_metric
    self.adaptor = self.mesh_predictor.adaptor


  def forward(
      self,
      input_batch,
      fwd=True,
      invert=False,
      infer=False,
      ):

    output_batch = None
    assert 'supervised' in input_batch
    assert 'unsupervised' in input_batch

    supervised_output = self.mesh_predictor(input_batch['supervised'])
    unsupervised_output = self.mesh_predictor(input_batch['unsupervised'])
    morphed_output = {}

    output_batch = {
        'mesh_predictor':{
          'supervised': supervised_output,
          'unsupervised': unsupervised_output,
          },
        'flow': {},
        }

    # Fwd: Unsupervised ==> Supervised
    if fwd:

      context_root = self.context_head_root(
          supervised_output[self.feature_key_root],
          unsupervised_output[self.feature_key_root],
          )

      context_rel = self.context_head_rel(
          supervised_output[self.feature_key_rel],
          unsupervised_output[self.feature_key_rel],
          )

      output_batch['flow']['supervised'] = self.morph(
          unsupervised_output,
          context_root,
          context_rel,
          gt_mano_shape=input_batch['supervised'][Fls.mano_shapes] \
              if Fls.mano_shapes in input_batch['supervised'] and self.use_gt_mano_shape else None,
          )

    # Invert (bkw) & Infer: supervised ==> Unsupervised
    if invert or infer:
      context_root = self.context_head_root(
          unsupervised_output[self.feature_key_root],
          supervised_output[self.feature_key_root],
          )

      context_rel = self.context_head_rel(
          unsupervised_output[self.feature_key_rel],
          supervised_output[self.feature_key_rel],
          )


    if invert:
      output_batch['flow']['unsupervised'] = self.morph(
          input_batch['supervised'] if self.invert_from_gts else supervised_output,
          context_root,
          context_rel,
          gt_mano_shape=input_batch['supervised'][Fls.mano_shapes] \
              if Fls.mano_shapes in input_batch['supervised'] and self.use_gt_mano_shape else None,
          )

    if infer:
      if invert and (self.infer_from_gts == self.invert_from_gts):
        output_batch['flow']['infer_unsupervised'] = output_batch['flow']['unsupervised']
      else:
        output_batch['flow']['infer_unsupervised'] = self.morph(
            input_batch['supervised'] if self.infer_from_gts else supervised_output,
            context_root,
            context_rel,
            gt_mano_shape=input_batch['supervised'][Fls.mano_shapes] \
                if Fls.mano_shapes in input_batch['supervised'] and self.use_gt_mano_shape else None,
            )

    return output_batch


  def morph(
      self,
      input_batch,
      context_root,
      context_rel,
      **kwargs,
      ):

    assert all(_ in input_batch for _ in (self.morph_key, ))
    if self.morph_key in (Fls.mano_params, ):
      output_batch = self.morph_mano_params(
          input_batch,
          context_root,
          context_rel,
          **kwargs,
          )

    else:
      raise NotImplementedError()

    return output_batch


  def morph_mano_params(
      self,
      input_batch,
      context_root,
      context_rel,
      gt_mano_shape=None,
      ):

    if self.morph_key == Fls.mano_params:

      mano_params = input_batch[Fls.mano_params]
      pose, shape, root = mano_params[:, :-13], mano_params[:, -13:-3], mano_params[:, -3:]

      # RELATIVE MOTION
      field_dim = pose.shape[-1] + 10 if self.morph_shape else pose.shape[-1]

      if field_dim < self.flow_rel.velocity_field.in_dim :
        # pose is pca; flow is on full pose
        pose = input_batch[Fls.mano_full_poses]

      if field_dim > self.flow_rel.velocity_field.in_dim :
        # pose is full; flow is on pca
        pass

      field_dim = pose.shape[-1] + 10 if self.morph_shape else pose.shape[-1]
      assert field_dim == self.flow_rel.velocity_field.in_dim, "Expected same dimension for morphing."
      points_rel = torch.cat(
          [pose, shape] if self.morph_shape else [pose],
          dim=-1,
          ).unsqueeze(1)

      morphed_points_rel = self.flow_rel(
          points_rel,
          context_rel,
          )

      # ROOT MOTION
      points_root = root.unsqueeze(1)
      morphed_points_root = self.flow_root(
          points_root,
          context_root,
          )

      morphed_points = torch.cat([morphed_points_rel, morphed_points_root], dim=-1)

      if False:# self.flow.return_multiTime:
        T, B = morphed_points.shape[:2]
        morphed_points = morphed_points.contiguous().view(T*B, *morphed_points.shape[2:])
      else:
        T = 1

      if not self.morph_shape:
        if gt_mano_shape is not None:
          shape = gt_mano_shape

      morphed_pose = (morphed_points[..., :-13] if self.morph_shape else morphed_points[..., :-3]).squeeze()
      morphed_shape = morphed_points[..., -13:-3].squeeze() if self.morph_shape else shape[None].repeat(T, 1, 1).view(morphed_points.shape[0], 10)
      morphed_root = morphed_points[..., -3:].squeeze()

      if morphed_pose.shape[-1] == self.mesh_predictor.manolayer.ncomps+3:
        verts3d_rel, joints3d_rel = self.mesh_predictor.manolayer(
            th_pose_coeffs=morphed_pose,
            th_betas=morphed_shape,
            )
      else:
        verts3d_rel, joints3d_rel = self.mesh_predictor.manolayer_full_poses(
            th_pose_coeffs=morphed_pose,
            th_betas=morphed_shape,
            )

      # annotation metric
      if self.annot_metric == "m":
        verts3d_rel /= 1000.0
        joints3d_rel /= 1000.0

      joints3d = joints3d_rel + morphed_root.unsqueeze(1)
      verts3d = verts3d_rel + morphed_root.unsqueeze(1)
      mano_params = torch.cat(
          [morphed_pose, morphed_shape, morphed_root],
          dim=-1,
          )
      mano_poses = morphed_pose
      mano_shapes = morphed_shape

      # output

      if False:#self.flow.return_multiTime:
        output = {
            Fls.verts3d: verts3d.contiguous().view(T, B, *verts3d.shape[1:]),
            Fls.verts3d_rel: verts3d_rel.contiguous().view(T, B, *verts3d_rel.shape[1:]),
            Fls.joints3d: joints3d.contiguous().view(T, B, *joints3d.shape[1:]),
            Fls.joints3d_rel: joints3d_rel.contiguous().view(T, B, *joints3d_rel.shape[1:]),
            Fls.mano_params: mano_params.contiguous().view(T, B, *mano_params.shape[1:]),
            Fls.mano_poses: mano_poses.contiguous().view(T, B, *mano_poses.shape[1:]),
            Fls.mano_shapes: mano_shapes.contiguous().view(T, B, *mano_shapes.shape[1:]),
            }
      else:
         output = {
            Fls.verts3d: verts3d,
            Fls.verts3d_rel: verts3d_rel,
            Fls.joints3d: joints3d,
            Fls.joints3d_rel: joints3d_rel,
            Fls.mano_params: mano_params,
            Fls.mano_poses: mano_poses,
            Fls.mano_shapes: mano_shapes,
            }

    else:
      raise NotImplementedError()

    return output
