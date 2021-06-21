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
      cond_mode='add',
      use_positional_encoding=False,
      positional_encoding_kwargs={},
      ):
    """
    Args:
      cond_mode: one of ['add', 'cat']
    """
    super().__init__()
    assert(cond_mode in ['add','cat'])
    self.cond_mode = cond_mode
    self.in_dim = in_dim

    self.positional_encoder = PositionalEncoding(**positional_encoding_kwargs) if use_positional_encoding else None
    in_dim = in_dim if self.positional_encoder is None else self.positional_encoder.get_dim(in_dim)
    out_dim = out_dim

    self.n_blocks = n_blocks
    self.fc_p = nn.Linear(in_dim, hidden_size)
    self.fc_c = nn.ModuleList([ nn.Linear(c_dim, hidden_size) for i in range(n_blocks) ])
    if cond_mode == 'cat':
      self.fc_cat = nn.ModuleList([nn.Linear(2*hidden_size,hidden_size) for i in range(n_blocks) ])
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
    net = self.fc_p(x) # [B,V,H]
    for i in range(self.n_blocks):
      net_c = self.fc_c[i](cond).unsqueeze(1) # [B,1,H]
      if self.cond_mode == 'add':
        net = net + net_c # [B,V,H]
      elif self.cond_mode == 'cat':
        net = torch.cat([net,net_c.expand_as(net)], dim=-1) # [B,V,2H]
        net = self.fc_cat[i](net) # [B,V,H]
      net = self.blocks[i](net) # [B,V,H]
    out = self.fc_out(self.activation(net)) # [B,V,O]
    return out
