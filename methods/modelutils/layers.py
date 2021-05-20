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


# fcs
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
  if activation: activation = get_activation(activation)

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
    if activation: layers.append(activation)

  layers.append(nn.Linear(ch_hiddens[-1], ch_out))
  return  nn.Sequential(*layers)

