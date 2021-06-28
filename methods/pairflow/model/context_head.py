from einops import rearrange

import torch
from torch import nn, einsum


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

    if self.op == 'subtract':
      return self.context_head(f1 - f2)

    elif self.op == "spatial_subtract":
      return self.to_vector(self.context_head(f1-f2))

    elif self.op == 'cat':
      return self.context_head(torch.cat([f1, f2], dim=-1))

    elif self.op == 'scaled_dot_product':
      b, c, u, v = f1.shape #q
      _, _, h, w = f2.shape #k
      scale = c ** -0.5

      f1, f2 = map(lambda t: rearrange(t, 'b c x y -> b x y c'), (f1, f2))
      sim = einsum('b u v c, b h w c -> b u v h w', f1, f2) * scale
      sim = rearrange(sim, 'b u v h w -> b u v (h w)')
      attn = sim.softmax(dim=-1)

      attn = rearrange(attn, 'b u v (h w) -> b u v h w', h=h)
      out =  einsum('b u v h w, b d h w -> b d u v', attn, f2)

      return self.to_vector(self.context_head(out))

    elif self.op in ('cross_attention', ):
      assert len(f1.shape)==4 and len(f2.shape)==4, "Expected spatial features!"
      return self.to_vector( self.context_head(f1, f2) )

    else:
      raise NotImplementedError()
