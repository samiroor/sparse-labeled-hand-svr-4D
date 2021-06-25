import torch
import torch.nn as nn

from methods.modelutils.layers import ResnetBlockFC, get_activation
from methods.modelutils.rotation_conversions import *



class AngularLinearVelocityField(nn.Module):

  def __init__(
      self,
      angular_kwargs,
      linear_kwargs,
      rot_mode="axisang",
      same_cond_proj=True,
      **kwargs,
      ):

    super().__init__()
    self.angular_field = None; self.angular_dim = 0
    self.linear_field = None; self.linear_dim = 0

    if angular_kwargs.get('in_dim', 0) > 0:

      in_dim = angular_kwargs['in_dim']

      c_dim = angular_kwargs['c_dim']
      hidden_size = angular_kwargs['hidden_size']
      n_blocks = angular_kwargs['nblocks']
      cond_mode = angular_kwargs['cond_mode']

      activation = angular_kwargs['activation']

      if rot_mode == "axisang":
        quat_dim = 4 * angular_kwargs.get('in_dim', 48) // 3

      fc_p = nn.Linear(quat_dim, hidden_size)
      fc_out = nn.Linear(hidden_size, quat_dim)
      activation = get_activation(activation)

      fcs_c = nn.ModuleList([ nn.Linear(c_dim, hidden_size) for i in range(n_blocks) ])
      blocks_c = nn.ModuleList([ ResnetBlockFC(hidden_size) for i in range(n_blocks) ])


      self.angular_field = AngularLinearVelocityField(
          fc_p=fc_p,
          fc_out=fc_out,
          fcs_c=fcs_c,
          blocks_c=blocks_c,
          activation=activation,
          cond_mode=cond_mode,
          )

      self.angular_dim = in_dim

    if linear_kwargs.get('in_dim', 0) > 0:
      in_dim = linear_kwargs['in_dim']
      cond_mode = linear_kwargs['cond_mode']
      activation = linear_kwargs['activation']

      if not same_cond_proj or angular_kwargs.get('in_dim', 0) < 0:
        c_dim = linear_kwargs['c_dim']
        hidden_size = linear_kwargs['hidden_size']
        n_blocks = linear_kwargs['nblocks']

        fcs_c = nn.ModuleList([ nn.Linear(c_dim, hidden_size) for i in range(n_blocks) ])
        blocks_c = nn.ModuleList([ ResnetBlockFC(hidden_size) for i in range(n_blocks) ])

      fc_p = nn.Linear(in_dim, hidden_size)
      fc_out = nn.Linear(hidden_size, in_dim)
      activation = get_activation(activation)

      self.linear_field = LinearVelocityField(
          fc_p=fc_p,
          fc_out=fc_out,
          fcs_c=fcs_c,
          blocks_c=blocks_c,
          activation=activation,
          cond_mode=cond_mode,
          )

      self.linear_dim = in_dim

  def forward(self, x, cond):
    if self.angular_dim > 0:
      theta = self.angular_field(
          x[...,:self.angular_dim],
          cond,
          )
    if self.linear_dim > 0:
      r = self.linear_field(
          x[..., -self.linear_dim:],
          cond,
          )

    if self.angular_dim > 0 and self.linear_dim > 0:
      return torch.cat(
          [theta, r],
          dim=-1,
          )
    elif self.angular_dim > 0:
      return theta
    elif self.linear_dim > 0:
      return r
    else:
      raise ValueError('WRONG')



class LinearVelocityField(nn.Module):

  def __init__(
      self,
      fc_p=None,
      fc_out=None,
      fcs_c=None,
      blocks_c=None,
      activation=None,
      cond_mode="cat",
      ):

    super().__init__()

    self.fc_p = fc_p
    self.fc_out = fc_out
    self.fcs_c = fcs_c
    self.blocks_c = blocks_c
    self.activation = activation
    self.cond_mode = cond_mode

  def forward(self, x, cond):
    """
      x: [B, V, 3] | [B, 1, 3]
      cond: [B, c_dim]
    """
    batch_size = x.shape[0]
    _size = x.shape[1:]

    # [B, V, H]
    net = self.fc_p(x)

    for fc_c, block_c in zip(self.fcs_c, self.blocks_c):
      # [B, 1, H]
      net_c = fc_c(cond).unsqueeze(1)

      if cond_mode == "cat":
        net = torch.cat([net, net_c.expand_as(net)], dim=-1) #[B, V, H'=2H]
      elif cond_mode == "add":
        net = net + net_c

      net = block(net)

    return self.fc_out(self.activation(net)) #[B, V, 3]



class AnugularVelocityField(nn.Module):

  def __init__(
      self,
      fc_p=None,
      fc_out=None,
      fcs_c=None,
      blocks_c=None,
      activation="tanh",
      cond_mode="",
      rot_mode="axisang",
      ):

    super().__init__()

    self.fc_p = fc_p
    self.fc_out= fc_out
    self.fcs_c = fcs_c
    self.blocks_c = blocks_c
    self.activation = get_activation(activation)
    self.cond_mode = 'cat'
    self.rot_mode = rot_mode

  def forward(self, theta, cond):
    batch_size = theta.shape[0]
    _size = theta.shape[1:]

    if self.rot_mode == "axisang":
      # [B, j, 3] 
      theta = theta.view(batch_size, -1, 3)
      # [B, j, 4]
      quats = axis_angle_to_quaternion(theta)
      quats = standardize_quaternion(quats)
    else:
      raise NotImplementedError()

    # [B, H]
    net = self.fc_p(quats.view(batch_size, -1))

    for fc_c, block_c in zip(self.fcs_c, self.blocks_c):
      # [B, H]
      net_c = fc_c(cond)

      if cond_mode == "cat":
        net = torch.cat([net, net_c], dim=-1) #[B, H'=2H]
      elif cond_mode == "sum":
        net = net + net_c

      net = block(net)

    # [B, j*4]
    d_quats = self.fc_out(
        self.activation(net)
        )
    # [B, j, 4]
    d_quats = nn.functional.normalize(
        d_quats.view(batch_size, -1, 4),
        dim=-1
        )

    # rotate [B, j, 4]
    quats = quaternion_multiply(quats, d_quats)

    # back to representation
    if self.rot_mode == "axisang":
      # [B, j, 3]
      theta = quaternion_to_axis_angle(quats)

    return theta.view(batch_size, *_size) #[B, 1, j*3] | [B, j*3]

