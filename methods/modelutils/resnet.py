import torch
import torch.nn as nn
from torch import Tensor

import torchvision
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



class NewResNet(torchvision.models.resnet.ResNet):
  def _forward_impl(self, x, return_layers=list()):
    features = {}

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    if 'layer1' in return_layers: features['layer1'] = x
    x = self.layer2(x)
    if 'layer2' in return_layers: features['layer2'] = x
    x = self.layer3(x)
    if 'layer3' in return_layers: features['layer3'] = x
    x = self.layer4(x)
    if 'layer4' in return_layers: features['layer4'] = x

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    features['avgpool'] = x

    return features

  def forward(self, x, return_layers=list()):
    return self._forward_impl(x, return_layers)



def _resnet(
    arch: str,
    block: Type[Union[torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
    ):
  model = NewResNet(block, layers, **kwargs)
  if pretrained:
    state_dict = load_state_dict_from_url(
        model_urls[arch],
        progress=progress,
        )
    model.load_state_dict(state_dict)
  return model



def resnet18(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
    ):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
