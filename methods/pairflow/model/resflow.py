import torch
import torch.nn as nn
from data import Fields as fields

class ResFlow(nn.Module):
  def __init__(self, velocity_field, T):
    super().__init__()
    self.velocity_field = velocity_field
    self.T = T

  def forward(self, input_dict, cond):
    """x: [B,V,3]
       cond: [B,d]
    """
    x = input_dict[fields.verts3d]
    y_hat = self.forward_order4(x, cond)
    # TODO need intermediate?
    output_dict = {fields.verts3d: y_hat[-1]}
    return output_dict

  def inverse(self, input_dict):
    x = input_dict[fields.verts3d]
    y_hat_inv = self.inverse_order4(x, cond)
    # TODO need intermediate
    output_dict = {filds.verts3d: y_hat_inv[-1]}
    return output_dict

  def forward_order1(self, x, cond):
    # Defines the forward pass of the network.
    # Returns the input and output of all the layers
    xs = [x]

    # Repeat the linear layers T times
    for i in range(self.T):
      y = self.velocity_field(xs[-1], cond)/self.T + xs[-1]
      xs += [y]
    return torch.stack(xs)

  def forward_order2(self, x, cond):
    # Defines the forward pass of the network.
    # This uses an order-2 runge-kutta method
    # Returns the input and output of all the layers
    xs = [x]
    h = 1.0/self.T

    # Repeat the linear layers T times
    for i in range(self.T):
      y0 = xs[-1]
      y1 = y0 + self.velocity_field(y0, cond)*h/2
      y2 = y0 + self.velocity_field(y1, cond)*h
      xs += [y2]

    return torch.stack(xs)

  def forward_order4(self, x, cond):
    # Defines the forward pass of the network.
    # This uses an order-4 runge-kutta method
    # Returns the input and output of all the layers
    xs = [x]
    h = 1.0/self.T

    # Repeat the linear layers T times
    for i in range(self.T):
      y0 = xs[-1]
      k1 = self.velocity_field(y0, cond)
      k2 = self.velocity_field(y0 + k1*h/2, cond)
      k3 = self.velocity_field(y0 + k2*h/2, cond)
      k4 = self.velocity_field(y0 + k3*h, cond)
      yn = y0 + (k1 + 2*k2 + 2*k3 + k4, cond)*h/6
      xs += [yn]

    return torch.stack(xs)

  def inverse_order4(self, x, cond):
    # Defines the inverse flow of the network.
    # This uses an order-4 runge-kutta method
    # Returns the input and output of all the layers
    xs = [x]
    h = 1.0/self.T

    # Repeat the linear layers T times
    for i in range(self.T):
      y0 = xs[-1]
      k1 = -self.velocity_field(y0, cond)
      k2 = -self.velocity_field(y0 + k1*h/2, cond)
      k3 = -self.velocity_field(y0 + k2*h/2, cond)
      k4 = -self.velocity_field(y0 + k3*h, cond)
      yn = y0 + (k1 + 2*k2 + 2*k3 + k4, cond)*h/6
      xs += [yn]

    return torch.stack(xs)

  def inverse_order1(self, x, cond):
    # Computes an inverse mapping for the network
    # Returns the input and output of all the layers
    xs = [x]

    # Repeat the linear layers T times
    for i in range(self.T):
      y = xs[-1] - self.velocity_field(xs[-1], cond)/self.T
      xs += [y]
    return torch.stack(xs)

  def inverse_order2(self, x, cond):
    # Defines the inverse flow
    # Returns the input and output of all the layers
    xs = [x]
    h = 1.0/self.T

    # Repeat the linear layers T times
    for i in range(self.T):
      y0 = xs[-1]
      y1 = y0 - self.velocity_field(y0, cond)*h/2
      y2 = y0 - self.velocity_field(y1, cond)*h
      xs += [y2]

    return torch.stack(xs)

