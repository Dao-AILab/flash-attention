# Inspired by https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py

from typing import Tuple
import math

import torch

from einops import rearrange, repeat

import rotary_emb


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1]
    cos = repeat(cos, 's d -> s 1 (2 d)')
    sin = repeat(sin, 's d -> s 1 (2 d)')
    return torch.cat([x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin,
                      x[..., rotary_dim:]], dim=-1)


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert cos.shape == (rotary_seqlen, rotary_dim // 2)
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x1, x2 = x[..., :rotary_dim].chunk(2, dim=-1)
        out = torch.empty_like(x) if not inplace else x
        o1, o2 = out[..., :rotary_dim].chunk(2, dim=-1) if not inplace else (x1, x2)
        rotary_emb.apply_rotary(x1, x2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), o1, o2, False)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do1, do2 = do[..., :rotary_dim].chunk(2, dim=-1)
        dx = torch.empty_like(do) if not inplace else do
        dx1, dx2 = dx[..., :rotary_dim].chunk(2, dim=-1) if not inplace else (do1, do2)
        rotary_emb.apply_rotary(do1, do2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), dx1, dx2, True)
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None


apply_rotary_emb_func = ApplyRotaryEmb.apply


class ApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert cos.shape == (seqlen, rotary_dim // 2)
        assert sin.shape == (seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, :, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), q1, q2, False)
        k1, k2 = qkv[:, :, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(k1, k2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), k1, k2, False)
        ctx.save_for_backward(cos, sin)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, :, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dq1, dq2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), dq1, dq2, True)
        dk1, dk2 = dqkv[:, :, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dk1, dk2, rearrange(cos[:, :seqlen], 's d -> s 1 d'),
                                rearrange(sin[:, :seqlen], 's d -> s 1 d'), dk1, dk2, True)
        return dqkv, None, None


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    """

    def __init__(self, dim: int, base=10000, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, x, seqlen_offset=0):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)
        """
        seqlen = x.shape[1] + seqlen_offset
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seqlen > self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset)
        return apply_rotary_emb_qkv_(qkv, self._cos_cached[seqlen_offset:],
                                     self._sin_cached[seqlen_offset:])
