import torch
import torch.nn as nn


from methods.modelutils.layers import ResnetBlockFC, get_activation
from methods.modelutils.rotation_conversions import *



class RotationTranslationVelocityField(nn.Module):

  def __init__(
      self,
      in_dim=51,
      c_dim=512,
      hidden_size=512,
      n_blocks=1,
      cond_mode='cat',
      activation="tanh",
      angular_dim=48,
      rot_mode="axisang",
      disentangle=True,
      temperature_scaling=False,
      #positional_encoding=False,
      #positional_encoding_kwargs={},
      **kwargs,
      ):

    super().__init__()
    self.in_dim = in_dim

    self.linear_dim = in_dim - angular_dim
    self.angular_dim = angular_dim
    self.cond_mode = cond_mode
    self.rot_mode = rot_mode
    self.disentangle = disentangle

    if self.rot_mode == "axisang":
      self.quaternion_dim = self.angular_dim // 3 * 4
    else:
      raise NotImplementedError()

    # update in_dim
    in_dim = self.quaternion_dim + self.linear_dim

    # activation
    self.activation = get_activation(activation)

    # condition
    self.fc_c = nn.ModuleList([ nn.Linear(c_dim, hidden_size) for i in range(n_blocks) ])
    self.blocks_c = nn.ModuleList([ ResnetBlockFC(hidden_size) for i in range(n_blocks) ])
    if self.cond_mode == 'cat':
      self.fc_cat_c = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size) for i in range(n_blocks) ])

    # disentangle hiddens 
    if self.disentangle:
      assert self.angular_dim > 0 and self.linear_dim > 0,\
          "Expected both type inputs!"

      h_a = int(hidden_size * self.quaternion_dim / in_dim)
      h_l = hidden_size - h_a

      self.fc_p_a = nn.Linear(self.quaternion_dim, h_a)
      self.fc_p_l = nn.Linear(self.linear_dim, h_l)

      self.fc_out_a = nn.Linear(h_a, self.quaternion_dim)
      self.fc_out_l = nn.Linear(h_l, self.linear_dim)

    else:

      self.fc_p = nn.Linear(in_dim, hidden_size)
      self.fc_out = nn.Linear(hidden_size, out_dim)

    # remperature_scaling
    self.temperature_scaling = nn.Parameter(torch.ones(in_dim, )) if temperature_scaling else 1.0
    self.n_blocks = n_blocks

  def forward(self, x, cond):
    """
    X: [B, N, in_dim]
    cond: [B, c_dim]
    """
    batch_size = x.shape[0]
    N = x.shape[1]
    if self.angular_dim > 0:
      # [B*N, 3*j]
      theta = x[..., :self.angular_dim].contiguous().view(batch_size * N, -1)
      # [B*N*j, 4]
      quaternions = axis_angle_to_quaternion(theta.contiguous().view(-1, 3))
      quaternions = standardize_quaternion(quaternions)
      # [B, N, 4*j]
      quaternions = quaternions.contiguous().view(batch_size, N, -1)
      # [B, N, in_dim]
      y = torch.cat(
          [quaternions, x[..., self.angular_dim:]],
          dim=-1,
          )
    else:
      y = x

    assert self.quaternion_dim + self.linear_dim == y.shape[-1], "Wrong dimensions!"

    # project input
    if self.disentangle:
      h_a = self.fc_p_a(y[..., :self.quaternion_dim])
      h_l = self.fc_p_l(y[..., self.quaternion_dim:])
      h = torch.cat(
          [h_a, h_l],
          dim=-1,
          )
      h_a_size = h_a.shape[-1]
      h_l_size = h_l.shape[-1]

    else:
      h = self.fc_p(y)

    # condition
    for i in range(self.n_blocks):
      h_c = self.fc_c[i](cond).unsqueeze(1) # [B, 1, H]
      if self.cond_mode == 'add':
        h = h + h_c # [B, N, H]
      elif self.cond_mode == 'cat':
        h = torch.cat([h, h_c.expand_as(h)], dim=-1) # [B, N, 2H]
        h = self.fc_cat_c[i](h) # [B, N, H]
      h = self.blocks_c[i](h) # [B, N, H]

    # dy
    if self.disentangle:
      h_a = h[..., :h_a_size]
      h_l = h[..., h_a_size:]

      out_a = self.fc_out_a(self.activation(h_a)) *\
          self.temperature_scaling[:self.quaternion_dim] # [B, N, j*4]

      out_l = self.fc_out_l(self.activation(h_l)) *\
          self.temperature_scaling[self.quaternion_dim:] # [B, N, d]

    else:
      out = self.fc_out(self.activation(h)) * self.temperature_scaling
      out_a = out[..., :self.quaternion_dim]
      out_l = out[..., self.quaternion_dim:]

    # normalize Dquaternions
    if self.angular_dim > 0:
      out_a = nn.functional.normalize(
        out_a.contiguous().view(-1, 4),
        dim=-1
        )
      out_a = standardize_quaternion(out_a).contiguous().view(batch_size, N, -1)

    # outputs
    if self.angular_dim > 0 and self.linear_dim > 0:
      return {
          'dy': torch.cat(
            [out_a, out_l],
            dim=-1),
          'q':quaternions,
          }

    elif self.angular_dim > 0:
      return {'dy': out_a, 'q':quaternions}

    elif self.linear_dim > 0:
      return {'dy': out_l}

    else:
      raise ValueError()
