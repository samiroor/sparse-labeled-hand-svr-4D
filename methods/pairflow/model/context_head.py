import torch
import torch.nn as nn


class ContextHead(nn.Module):
  def __init__(
      self,
      context_base=None,
      context_head=None,
      **kwargs,
      ):
    super().__init__()
    self.context_base = context_base if context_base is not None else lambda x:x
    self.context_head = context_head if context_head is not None else lambda x:x

  def forward(self, f1, f2):
    raise NotImplementedError


class OPContextHead(ContextHead):
  def __init__(
      self,
      context_base=None,
      context_head=None,
      op='cat',
      **kwargs,
      ):

    super().__init__(
        context_base=context_base,
        context_head=context_head,
        )

    self.op = op
    self.to_vector = nn.Sequential(*[
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten(),
      ])

  def forward(
      self,
      f1,
      f2,
      ):

    f1 = self.context_base(f1)
    f2 = self.context_base(f2)

    if op == 'subtract':
      return self.context_head(f1 - f2)

    elif op == 'cat':
      return self.context_head(torch.cat([f1, f2], dim=-1))

    elif op in ('cross_attention', ):
      assert len(f1.shape)==4 and len(f2.shape)==4, "Expected spatial features!"
      return self.to_vector( self.context_head(f1, f2) )
