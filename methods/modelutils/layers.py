import torch
from torch import nn, einsum
import torch.nn.functional as F


from einops import rearrange


def get_activation(name):
  if name == 'relu':
    return nn.ReLU()
  elif name == 'silu':
    return nn.SiLU()
  elif name == 'gelu':
    return nn.GELU()
  elif name == 'sigmoid':
    return nn.Sigmoid()
  elif name == 'softmax':
    return nn.Softmax(dim=-1)
  elif name == 'logsoftmax':
    return nn.LogSoftmax(dim=-1)
  elif name == 'softplus':
    return nn.Softplus()
  elif name == 'tanh':
    return nn.Tanh()
  else:
    raise ValueError(f'Not a valid activation "{name}"!')


def fcs(
    ch_in,
    ch_out,
    ch_hiddens,
    dropout=0.,
    activation=None,
    flatten=False,
    ):

  layers = []

  if flatten: layers.append(nn.Flatten())
  if activation is not None: activation = get_activation(activation)

  # no hiddens
  if len(ch_hiddens) == 0:

    if dropout: layers.append(nn.Dropout(p=dropout))
    if activation: layers.append(activation)
    layers.append(nn.Linear(ch_in, ch_out))
    return  nn.Sequential(*layers)

  # with hiddens
  for in_size, out_size in zip([ch_in] + ch_hiddens[:-1], ch_hiddens):
    if dropout: layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_size, out_size))
    if activation is not None: layers.append(activation)

  layers.append(nn.Linear(ch_hiddens[-1], ch_out))
  return  nn.Sequential(*layers)



# Resnet Blocks
class ResnetBlockFC(nn.Module):
  ''' Fully connected ResNet Block class.
  Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
  '''

  def __init__(
      self,
      size_in,
      size_out=None,
      size_h=None,
      ):
    super().__init__()

    # Attributes
    if size_out is None:
      size_out = size_in

    if size_h is None:
      size_h = min(size_in, size_out)

    self.size_in = size_in
    self.size_h = size_h
    self.size_out = size_out

    # Submodules
    self.fc_0 = nn.Linear(size_in, size_h)
    self.fc_1 = nn.Linear(size_h, size_out)
    self.actvn = nn.ReLU()

    if size_in == size_out:
      self.shortcut = None
    else:
      self.shortcut = nn.Linear(size_in, size_out, bias=False)

    # Initialization
    nn.init.zeros_(self.fc_1.weight)

  def forward(self, x):
    net = self.fc_0(self.actvn(x))
    dx = self.fc_1(self.actvn(net))

    if self.shortcut is not None:
      x_s = self.shortcut(x)
    else:
      x_s = x

    return x_s + dx



class PositionalEncoding(nn.Module):
  """Positional encoding layer
  """

  def __init__(
      self,
      linear=True,
      length=16,
      ):
    """Args
        - linear: if true omegas are linear
                  else they will be powers of 2
    """
    super().__init__()
    omega_range = torch.arange(1,length+1, dtype=torch.float32)[None]
    if linear:
      self.register_buffer("omega_range", omega_range)
    else:
      self.register_buffer("omega_range", 2**omega_range)

  def get_dim(self, dim):
    return 2*dim*self.omega_range.shape[-1]

  def forward(self, X):
    omega_X = X[...,None] @ self.omega_range
    omega_X = omega_X.view(*X.shape[:-1],-1)
    return torch.cat([torch.cos(omega_X),torch.sin(omega_X)], dim=-1)

class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b x y d, x u v d -> b x y u v', q, height_emb)
        width_score = einsum('b x y d, y u v d -> b x y u v', q, width_emb)

        return height_score + width_score

class CrossTransformer(nn.Module):
  """Cross Transformer layer"""

  def __init__(self,
               dim=512,
               dim_key=128,
               dim_value=512,
               qk_same_proj=False,
               positional_embedding=False,
               max_pos_size=100,
               compute_s_v=False):
    super().__init__()
    self.scale = dim_key ** -0.5
    self.qk_same_proj = qk_same_proj
    if qk_same_proj:
      self.to_q = nn.Conv2d(dim, dim_key, 1, bias=False)
      self.to_k = None
    else:
      self.to_q = nn.Conv2d(dim, dim_key, 1, bias=False)
      self.to_k = nn.Conv2d(dim, dim_key, 1, bias=False)
    self.to_v = nn.Conv2d(dim, dim_value, 1, bias=False)
    self.pos_emb = None
    if positional_embedding:
      self.pos_emb = RelPosEmb(max_pos_size, dim_key)
    self.compute_s_v = compute_s_v

  def forward(self, s, u):
    # s : b, dim, u, v
    # u : b, dim, h, w
    # 
    to_k = self.to_k or self.to_q
    q = self.to_q(s) # q: b, c, u, v
    k = to_k(u) # k: b, c, h, w
    v = self.to_v(u) # v: b, d, h, w
    b, c, u, v =  q.shape
    _,_,h,w = k.shape
    q, k = map(lambda t: rearrange(t, 'b c x y -> b x y c', h=heads), (q, k))
    sim = einsum('b u v c, b h w c -> b u v h w', q, k) * self.scale
    if self.pos_emb is not None:
      sim_pos = self.pos_emb(q)
      sim = sim + sim_pos
    sim = rearrange(sim, 'b u v h w -> b u v (h w)')
    attn = sim.softmax(dim=-1)
    attn = rearrange(attn, 'b u v (h w) -> b u v h w', u=u)
    out = einsum('b u v h w, b d h w -> b d u v', attn, v) #s_aligned from u in v-space
    if self.compute_s_v:
      s_v = self.to_v(s) # s in v-space
      return out, s_v
    return out



