import torch
import torch.nn as nn


from data import Fields as Fls


class PairFlow(nn.Module):

  def __init__(
      self,
      mesh_predictor,
      context_head,
      flow,
      morph_key="verts3d", #doesnot work on relative
      feature_key='encoder_features',
      morph_shape=False,
      invert_from_gts=False,
      infer_from_gts=True,
      **kwargs,
      ):

    """
    A model that
     During training:
      1) encodes supervised and unsupervised images to get their
         embeddings (f_s and f_u respectively) using mesh_predictor.

      2) decodes the unsupervised features f_u to get 3d and 2d
         vertices and joints using mesh_predictor.

      3) inputs f_s, f_u to context_head to create context features (ctx)

      4) uses predicted verts3d_u from f_u and ctx as input to flow to
         deform verts3d_u to verts3d_s

    Args:

      mesh_predictor: (nn.Module) that takes images and outputs
                      predicted mesh as well as encoded features.

      context: (nn.Module) that takes featurs of supervised and
               unsupervised images and outputs the corresponding
               context feature.

      flow: (nn.Module) that takes context features and the
            unsupervised predicted mesh and estimates a mesh
            for the supervised one.

    Returns:
      predicted supervised outputs

    """

    super().__init__()

    self.mesh_predictor = mesh_predictor
    self.context_head = context_head
    self.flow = flow

    if morph_key == "verts3d":
      self.morph_key = Fls.verts3d
    elif morph_key =="verts3d_rel":
      self.morph_key = Fls.verts3d_rel
    elif morph_key == "mano_params":
      self.morph_key = Fls.mano_params
      self.morph_shape = morph_shape
    else:
      raise NotImplementedError()

    if feature_key == "tail_features":
      self.feature_key = Fls.tail_features
    elif feature_key == "encoder_tail_features":
      self.feature_key = Fls.encoder_tail_features
    else:
      self.feature_key = Fls.encoder_features

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
      context = self.context_head(
          supervised_output[self.feature_key],
          unsupervised_output[self.feature_key],
          )

      output_batch['flow']['supervised'] = self.morph(unsupervised_output, context)

    # Invert (bkw): supervised ==> Unsupervised
    if invert or infer:
      context = self.context_head(
          unsupervised_output[self.feature_key],
          supervised_output[self.feature_key],
          )

    if invert:
      output_batch['flow']['unsupervised'] = self.morph(
          input_batch['supervised'] if self.invert_from_gts else supervised_output,
          context,
          )

    if infer:
      if invert and (self.infer_from_gts == self.invert_from_gts):
        output_batch['flow']['infer_unsupervised'] = output_batch['flow']['unsupervised']
      else:
        output_batch['flow']['infer_unsupervised'] = self.morph(
            input_batch['supervised'] if self.infer_from_gts else supervised_output,
            context,
            )

    return output_batch


  def morph(
      self,
      input_batch,
      context,
      ):

    assert all(_ in input_batch for _ in (self.morph_key, ))

    if self.morph_key in (Fls.verts3d, ):
      output_batch = self.morph_3Dpoints(
          input_batch,
          context,
          )

    elif self.morph_key in (Fls.mano_params, ):
      output_batch = self.morph_mano_params(
          input_batch,
          context,
          )

    else:
      raise NotImplementedError()

    return output_batch


  def morph_3Dpoints(
      self,
      input_batch,
      context,
      ):

    points = input_batch[self.morph_key]
    morphed_points = self.flow(
        points,
        context,
        )

    if self.flow.return_multiTime:
      T, B = morphed_points.shape[:2]
      morphed_points = morphed_points.contiguous().view(T*B, *morphed_points.shape[2:])


    if self.morph_key == Fls.verts3d:
      verts3d = morphed_points

      assert self.adaptor is not None
      adapt_joints, _ = self.adaptor(verts3d)
      adapt_joints = adapt_joints.transpose(1, 2)

      joints3d = adapt_joints
      joints3d_rel = joints3d - joints3d[:, self.center_idx].unsqueeze(1)
      verts3d_rel = verts3d - joints3d[:, self.center_idx].unsqueeze(1)

    else:
      raise NotImplementedError()

    if self.flow.return_multiTime:
      return {
          Fls.verts3d: verts3d.contiguous().view(T, B, *verts3d.shape[1:]),
          Fls.verts3d_rel: verts3d_rel.contiguous().view(T, B, *verts3d_rel.shape[1:]),
          Fls.joints3d: joints3d.contiguous().view(T, B, *joints3d.shape[1:]),
          Fls.joints3d_rel: joints3d_rel.contiguous().view(T, B, *joints3d_rel.shape[1:]),
          }
    else:
      return {
          Fls.verts3d: verts3d,
          Fls.verts3d_rel: verts3d_rel,
          Fls.joints3d: joints3d,
          Fls.joints3d_rel: joints3d_rel,
          }


  def morph_mano_params(
      self,
      input_batch,
      context,
      morph_shape=False,
      ):

    if self.morph_key == Fls.mano_params:

      mano_params = input_batch[Fls.mano_params]
      pose, shape, root = mano_params[:, :-13], mano_params[:, -13:-3], mano_params[:, -3:]

      field_dim = pose.shape[-1] + 10 + 3 if self.morph_shape else pose.shape[-1] + 3

      if field_dim < self.flow.velocity_field.in_dim :
        # pose is pca; flow is on full pose
        pose = input_batch[Fls.mano_full_poses]

      if field_dim > self.flow.velocity_field.in_dim :
        # pose is full; flow is on pca
        pass

      field_dim = pose.shape[-1] + 10 + 3 if self.morph_shape else pose.shape[-1] + 3
      assert field_dim == self.flow.velocity_field.in_dim, "Expected same dimension for morphing."

      points = torch.cat(
          [pose, shape, root] if self.morph_shape else [pose, root],
          dim=-1,
          ).unsqueeze(1)

      morphed_points = self.flow(
          points,
          context,
          )

      if self.flow.return_multiTime:
        T, B = morphed_points.shape[:2]
        morphed_points = morphed_points.contiguous().view(T*B, *morphed_points.shape[2:])

      morphed_pose = morphed_points[..., :-13].squeeze()
      morphed_root = morphed_points[..., -3:].squeeze()
      morphed_shape = morphed_points[..., -13:-3].squeeze() if self.morphed_shape else shape.repeat(morphed_points.shape[0], 1)

      verts3d_rel, joints3d_rel = self.mesh_predictor.manolayer(
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

      # output
      if self.flow.return_multiTime:
        output = {
            Fls.verts3d: verts3d.contiguous().view(T, B, *verts3d.shape[1:]),
            Fls.verts3d_rel: verts3d_rel.contiguous().view(T, B, *verts3d_rel.shape[1:]),
            Fls.joints3d: joints3d.contiguous().view(T, B, *joints3d.shape[1:]),
            Fls.joints3d_rel: joints3d_rel.contiguous().view(T, B, *joints3d_rel.shape[1:]),
            Fls.mano_params: mano_params.contiguous().view(T, B, *mano_params.shape[1:])
            }
      else:
         outputi = {
            Fls.verts3d: verts3d,
            Fls.verts3d_rel: verts3d_rel,
            Fls.joints3d: joints3d,
            Fls.joints3d_rel: joints3d_rel,
            Fls.mano_params: mano_params,
            }

    else:
      raise NotImplementedError()

    return output
