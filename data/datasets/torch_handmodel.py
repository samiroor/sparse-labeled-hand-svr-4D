import pickle

import torch
import torch.nn as nn

from manopth.manolayer import ManoLayer


class Torch_HandModel(nn.Module):
  def __init__(self,
               mano_root,
               use_pca=False,
               use_mean_pose=True,
               ncomps=45,
               ref_side='right',
               center_idx=0,
               annot_metric="m",
               **kwargs):

    super(Torch_HandModel, self).__init__()
    self.manolayer = ManoLayer(mano_root=mano_root,
                               use_pca=use_pca,
                               flat_hand_mean=not use_mean_pose,
                               ncomps=ncomps,
                               side=ref_side,
                               center_idx=center_idx)

    self.center_idx = center_idx
    self.annot_metric = annot_metric
    self.faces = self.manolayer.th_faces.data.cpu().numpy()

    self.use_adaptor = kwargs['use_adaptor'] if 'use_adaptor'in kwargs else False
    adaptor_path = kwargs['adaptor_path'] if self.use_adaptor else None
    self.adaptor = ManoAdaptor(self.manolayer, adaptor_path)
    rec_freeze(self.adaptor)

  def forward(self, poses, shapes=None):
    verts3d_rel, joints3d_rel = self.manolayer(th_pose_coeffs=poses,
                                               th_betas=shapes)
    if self.use_adaptor:
      adapt_joints, _ = self.adaptor(verts3d_rel)
      adapt_joints = adapt_joints.transpose(1, 2)
      joints3d_rel = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
      verts3d_rel = verts3d_rel - adapt_joints[:, self.center_idx].unsqueeze(1)

    if self.annot_metric == "m":
      verts3d_rel /= 1000.0
      joints3d_rel /= 1000.0

    return joints3d_rel, verts3d_rel, self.manolayer.th_faces

  def verts2joints(self, verts3d):
    if self.annot_metric == "m":
      verts3d *= 1000.0

    adapt_joints, _ = self.adaptor(verts3d)
    adapt_joints = adapt_joints.transpose(1, 2)

    joints3d = adapt_joints
    #joints3d_rel = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
    #verts3d_rel = verts3d - adapt_joints[:, self.center_idx].unsqueeze(1)

    if self.annot_metric == "m":
      #verts3d_rel /= 1000.0
      #joints3d_rel /= 1000.0
      verts3d /=1000.0
      joints3d /= 1000.0

    #return joints3d_rel, verts3d_rel
    return joints3d, verts3d


class ManoAdaptor(torch.nn.Module):
  def __init__(self, mano_layer, load_path=None):
    super().__init__()
    self.adaptor = torch.nn.Linear(778, 21, bias=False)
    if load_path is not None:
      with open(load_path, "rb") as p_f:
        exp_data = pickle.load(p_f)
        weights = exp_data["adaptor"]
      regressor = torch.Tensor(weights)
      self.register_buffer("J_regressor", regressor)
    else:
      regressor = mano_layer._buffers["th_J_regressor"]
      tip_reg = regressor.new_zeros(5, regressor.shape[1])
      tip_reg[0, 745] = 1
      tip_reg[1, 317] = 1
      tip_reg[2, 444] = 1
      tip_reg[3, 556] = 1
      tip_reg[4, 673] = 1
      reordered_reg = torch.cat([regressor, tip_reg])[
          [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
      ]
      self.register_buffer("J_regressor", reordered_reg)
    self.adaptor.weight.data = self.J_regressor

  def forward(self, inp):
    fix_idxs = [0, 4, 8, 12, 16, 20]
    for idx in fix_idxs:
      self.adaptor.weight.data[idx] = self.J_regressor[idx]
    return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


def rec_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)


