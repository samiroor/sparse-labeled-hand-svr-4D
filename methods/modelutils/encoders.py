import torch.nn as nn
from . import resnet


class Resnet18(nn.Module):
  """
  ResNet-18 encoder network for image input.
  Args:
    c_dim (int): output dimension of the latent embedding
    use_linear (bool): whether an extra fc layer should be used or not to have c_dim output
    return_spatial_features (bool): return sptial features along with reduced features
    return_layers (list) : name of expected spatial features

  return:
    out (Tensor): reduced feature in [B, c_dim]
    features (dict): features['layer4']: Tensor of [B, C, H, W]

  """
  def __init__(
      self,
      c_dim,
      use_linear=True,
      return_spatial_features=False,
      return_layer='layer4',
      ):

    super().__init__()
    self.use_linear = use_linear
    self.return_spatial_features = return_spatial_features
    self.return_layer = return_layer

    self.features = resnet.resnet18(pretrained=True)
    self.features.fc = nn.Sequential()

    if use_linear:
      self.fc = nn.Linear(512, c_dim)
    elif c_dim == 512:
      self.fc = nn.Sequential()
    else:
      raise ValueError('c_dim must be 512 if use_linear is False')

  def forward(self, x):
    features = self.features(x, [self.return_layer] if self.return_spatial_features else list())
    out = self.fc(features['avgpool'])
    if self.return_spatial_features:
      features.pop('avgpool')
      return out, features[self.return_layer]
    return out
