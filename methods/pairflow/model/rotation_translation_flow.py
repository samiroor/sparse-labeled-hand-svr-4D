import torch
import torch.nn as nn

from methods.modelutils.rotation_conversions import *


class RotationTranslationFlow(nn.Module):

  def __init__(
      self,
      velocity_field,
      T=1,
      return_multiTime=False,
      **kwargs,
      ):

    super().__init__()
    self.velocity_field = velocity_field

    self.T = T
    self.return_multiTime = return_multiTime

    self.rot_mode = velocity_field.rot_mode
    self.linear_dim = velocity_field.linear_dim
    self.angular_dim = velocity_field.angular_dim
    self.quaternion_dim = velocity_field.quaternion_dim


  def forward(self, x, cond):
    """
    x : [B, N, 3]
    cond : [B, c_dim]
    """
    y_hat = self.forward_order1(x, cond)
    return y_hat[-1]


  def forward_order1(self, x, cond):
    xs = [x]
    for i in range(self.T):
      out_dict = self.velocity_field(xs[-1], cond)

      if self.linear_dim > 0:
        r = out_dict['dy'][..., self.quaternion_dim:] / self.T + xs[-1][..., self.angular_dim:]

      if self.angular_dim > 0 :
        q = quaternion_multiply(
            out_dict['q'].contiguous().view(-1, 4),
            out_dict['dy'][..., :self.quaternion_dim].contiguous().view(-1, 4),
            )
        if self.rot_mode == "axisang":
          theta = quaternion_to_axis_angle(q).contiguous().view(*x[..., :self.angular_dim].shape)
        else:
          NotImplementedError()

      if self.linear_dim > 0 and self.angular_dim > 0:
        y = torch.cat([theta, r], dim=-1)
      elif self.linear_dim > 0:
        y = r
      elif self.angular_dim > 0:
        y = theta
      else:
        raise ValueError()

      xs += [y]
    return torch.stack(xs)
