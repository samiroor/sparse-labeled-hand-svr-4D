import torch
import torch.nn as nn


class ContextHead(nn.Module):
  def __init__(
      self,
      context_base,
      ):
    super().__init__()
    self.context_base = context_base if context_base is not None else lambda x:x

  def forward(self, f1, f2):
    raise NotImplementedError


class SubtractContextHead(ContextHead):
  def __init__(
      self,
      context_base=None,
      **kwargs,
      ):
    super().__init__(context_base)

  def forward(self, f1, f2):
    f1 = self.context_base(f1)
    f2 = self.context_base(f2)
    return  f1-f2


class AttentionContextHead(ContextHead):
  pass


