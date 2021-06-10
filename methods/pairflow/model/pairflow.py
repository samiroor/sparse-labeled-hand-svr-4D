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
    else:
      raise NotImplementedError()

    if feature_key == "tail_features":
      self.feature_key = Fls.tail_features
    elif feature_key == "encoder_tail_features":
      self.feature_key = Fls.encoder_tail_features
    else:
      self.feature_key = Fls.encoder_features

    self.center_idx = self.mesh_predictor.center_idx
    self.adaptor = self.mesh_predictor.adaptor


  def forward(self, input_batch, fwd=True, invert=False, infer=False):
    output_batch = None
    assert 'supervised' in input_batch
    assert 'unsupervised' in input_batch

    supervised_output = self.mesh_predictor(input_batch['supervised'])
    unsupervised_output = self.mesh_predictor(input_batch['unsupervised'])
    morphed_output = {}

    output_batch = {
        'mesh_predictor':{
          'supervised':supervised_output,
          'unsupervised':unsupervised_output,
          },
        'flow':{},
        }

    # inference GT supervised ==> Unsupervised
    if infer:
      context = self.context_head(
          unsupervised_output[self.feature_key],
          supervised_output[self.feature_key],
          )

      morphed_output = self.morph(input_batch['supervised'], context)

      output_batch['flow'].update({
        'infer_unsupervised':morphed_output,
        })

    # inverse supervised ==> Unsupervised
    if invert:
      context = self.context_head(
          unsupervised_output[self.feature_key],
          supervised_output[self.feature_key],
          )

      morphed_output = self.morph(supervised_output, context)

      output_batch['flow'].update({
        'unsupervised':morphed_output,
        })

    # Unsupervised ==> supervised
    if fwd:
      context = self.context_head(
          supervised_output[self.feature_key],
          unsupervised_output[self.feature_key],
          )

      morphed_output = self.morph(unsupervised_output, context)

      output_batch['flow'].update({
        'supervised':morphed_output,
        })

    return output_batch


  def morph(self, input_batch, context):
    assert all(_ in input_batch for _ in (self.morph_key, ))

    if self.morph_key in (Fls.verts3d,):
      output_batch = self.morph_verts(input_batch, context)

    elif self.morph_key in (Fls.mano_params, ):
      output_batch = self.morph_mano_params(input_batch, context)

    else:
      raise NotImplementedError()

    return output_batch


  def morph_verts(self, input_batch, context):
    points = input_batch[self.morph_key]
    morphed_points = self.flow(points, context)

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

    return {
        Fls.verts3d: verts3d,
        Fls.verts3d_rel: verts3d_rel,
        Fls.joints3d: joints3d,
        Fls.joints3d_rel: joints3d_rel,
        }


  def morph_mano_params(self, input_batch, context):
    if self.morph_key == Fls.mano_params:

      mano_params = input_batch[self.morph_key]
      pose, shape, root = mano_params[:, :-13], mano_params[:, -13:-3], mano_params[:, -3:]

      points = torch.cat([pose, root], dim=-1).unsqueeze(1)
      morphed_points = self.flow(points, context)

      morphed_root = morphed_points[:,:, -3:].squeeze()
      morphed_pose = morphed_points[:, :, :-3].squeeze()
      morphed_shape = shape

      verts3d_rel, joints3d_rel = self.mesh_predictor.manolayer(
          th_pose_coeffs=morphed_pose,
          th_betas=morphed_shape,
          )

      # annotation metric
      if self.mesh_predictor.annot_metric == "m":
        verts3d_rel /= 1000.0
        joints3d_rel /= 1000.0

      joints3d = joints3d_rel + morphed_root.unsqueeze(1)
      verts3d = verts3d_rel + morphed_root.unsqueeze(1)

    else:
      raise NotImplementedError()

    return {
        Fls.verts3d: verts3d,
        Fls.verts3d_rel: verts3d_rel,
        Fls.joints3d: joints3d,
        Fls.joints3d_rel: joints3d_rel,
        Fls.mano_poses: morphed_pose,
        Fls.mano_shapes: morphed_shape,
        Fls.mano_params: torch.cat([morphed_pose, morphed_shape, morphed_root], dim=-1)
        }
