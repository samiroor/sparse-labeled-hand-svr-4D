from torch import nn


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
