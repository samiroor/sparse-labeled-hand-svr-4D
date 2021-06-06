import torch
import torch.nn as nn
from methods.modelutils.layers import ResnetBlockFC, get_activation, PositionalEncoding



class VelocityField(nn.Module):
  def __init__(
      self,
      in_dim=3,
      out_dim=3,
      c_dim=512,
      hidden_size=512,
      activation='tanh',
      n_blocks=5,
      use_positional_encoding=False,
      ):
    super().__init__()

    self.positional_encoder = PositionalEncoding() if use_positional_encoding else None
    in_dim = in_dim if self.positional_encoder is None else self.positional_encoder.get_dim(in_dim)
    out_dim = out_dim

    self.n_blocks = n_blocks
    self.fc_p = nn.Linear(in_dim, hidden_size)
    self.fc_c = nn.ModuleList([ nn.Linear(c_dim, hidden_size) for i in range(n_blocks) ])
    self.blocks = nn.ModuleList([ ResnetBlockFC(hidden_size) for i in range(n_blocks) ])
    self.fc_out = nn.Linear(hidden_size, out_dim)
    self.activation = get_activation(activation)


  def forward(self, x, cond):
    """
    x: [B,V,3]
    cond: [B,c_dim]
    """
    if self.positional_encoder:
      x = self.positional_encoder(x)
    net = self.fc_p(x)
    for i in range(self.n_blocks):
      net_c = self.fc_c[i](cond).unsqueeze(1)
      net = net + net_c
      net = self.blocks[i](net)
    out = self.fc_out(self.activation(net))
    return out
