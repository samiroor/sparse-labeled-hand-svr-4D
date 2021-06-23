from einops import rearrange

import torch
from torch import nn, einsum


class CrossAttention(nn.Module):
  def __init__(
      self,
      in_dim=512,
      key_dim=128,
      value_dim=512,
      qk_same_proj=False,
      positional_encoding=False,
      max_pos_size=100,
      **kwargs,
      ):

    super().__init__()
    self.name = 'cross'
    self.scale = key_dim ** -0.5

    self.to_q = nn.Conv2d(in_dim, key_dim, 1, bias=False)
    self.to_k = self.to_q if qk_same_proj else  nn.Conv2d(in_dim, key_dim, 1, bias=False)
    self.to_v = nn.Conv2d(in_dim, value_dim, 1, bias=False)

    self.pos_emb = None
    if positional_encoding:
      pass #TODO

  def forward(
      self,
      phi_q,
      phi_k,
      ):

    # phi_q : query images spatial features [B, in_dim, u, v]
    # phi_k : key images spatial features [B, in_dim, h, w] 

    q = self.to_q(phi_q) #B, c (key_dim), u, v
    k = self.to_k(phi_k) #B, c (key_dim), h, w
    V = self.to_v(phi_k) #B, d (value_dim), h, w

    b, c, u, v =  q.shape
    _, _, h, w = k.shape

    q, k = map(lambda t: rearrange(t, 'b c x y -> b x y c'), (q, k))
    sim = einsum('b u v c, b h w c -> b u v h w', q, k) * self.scale

    if self.pos_emb is not None:
      sim_pos = self.pos_emb(q)
      sim = sim + sim_pos

    sim = rearrange(sim, 'b u v h w -> b u v (h w)')
    attn = sim.softmax(dim=-1)
    attn = rearrange(attn, 'b u v (h w) -> b u v h w', h=h)

    out = einsum('b u v h w, b d h w -> b d u v', attn, V)

    return out



class MultiHeadCrossAttention(nn.Module):
  def __init__(
      self,
      nheads=1,
      in_dim=512,
      key_dim=128,
      value_dim=512,
      qk_same_proj=False,
      positional_encoding=False,
      max_pos_size=100,
      **kwargs,
      ):

    super().__init__()

    self.name = 'cross'
    self.heads = nn.ModuleList(
        [
          CrossAttention(
            in_dim=in_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            qk_same_proj=qk_same_proj,
            positional_encoding=positional_encoding,
            max_pos_size=max_pos_size,
            **kwargs,
            ) for _ in range(nheads)
          ]
        )

    self.unify_heads = nn.Conv2d(value_dim * nheads, value_dim, 1, bias=False) if nheads > 1 else lambda x: x

  def forward(
      self,
      phi_q,
      phi_k,
      ):
    """
    Args:
      phi_q (torch.Tensor): query images spatial features [B, in_dim, u, v]
      phi_k (torch.Tensor): key images spatial features [B, in_dim, h, w]

    return
      [B, value_dim, u, v]
       Aligned query spatial features in value (key) space
    """
    return self.unify_heads(
        torch.cat(
          [head(phi_q, phi_k) for head in self.heads],
          dim=1,
          )
        )



class VectorAttention(nn.Module):
  def __init__(
      self,
      in_dim=512,
      key_dim=128,
      value_dim=512,
      qk_same_proj=False,
      positional_encoding=False,
      max_pos_size=100,
      **kwargs,
      ):

    super().__init__()
    self.name = 'vector'
    self.scale = key_dim ** -0.5

    self.to_q = nn.Linear(in_dim, key_dim, bias=False)
    self.to_k = self.to_q if qk_same_proj else  nn.Linear(in_dim, key_dim, bias=False)
    self.to_v = nn.Linear(in_dim, value_dim, bias=False)

    self.pos_emb = None
    if positional_encoding:
      pass #TODO


  def forward(
      self,
      phi_q,
      phi_k,
      ):

    # phi_q : query images vector features [B, in_dim]
    # phi_k : key images vector features [B, in_dim] 

    q = self.to_q(phi_q) #B, c, t (key_dim)
    k = self.to_k(phi_k) #B, c, t (key_dim)
    V = self.to_v(phi_k) #B, d, t (value_dim)

    # B, B
    sim = torch.matmul(q, k.transpose(-2,-1)) * self.scale

    if self.pos_emb is not None:
      sim_pos = self.pos_emb(q)
      sim = sim + sim_pos

    attn = sim.softmax(dim=-1)

    return torch.matmul(attn, V) #B, c, c



class MultiHeadVectorAttention(nn.Module):
  def __init__(
      self,
      nheads=1,
      in_dim=512,
      key_dim=128,
      value_dim=512,
      qk_same_proj=False,
      positional_encoding=False,
      max_pos_size=100,
      **kwargs,
      ):

    super().__init__()

    self.name = 'vector'
    self.heads = nn.ModuleList(
        [
          VectorAttention(
            in_dim=in_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            qk_same_proj=qk_same_proj,
            positional_encoding=positional_encoding,
            max_pos_size=max_pos_size,
            **kwargs,
            ) for _ in range(nheads)
          ]
        )

    self.unify_heads = nn.Linear(value_dim * nheads, value_dim, 1, bias=False) if nheads > 1 else lambda x: x

  def forward(
      self,
      phi_q,
      phi_k,
      ):
    """
    Args:
      phi_q (torch.Tensor): query images vector features [B, in_dim, t]
      phi_k (torch.Tensor): key images vector features [B, in_dim, t]

    return
      [B, value_dim]
       Aligned query vector features in value (key) space
    """
    return self.unify_heads(
        torch.cat(
          [head(phi_q, phi_k) for head in self.heads],
          dim=1,
          )
        )

