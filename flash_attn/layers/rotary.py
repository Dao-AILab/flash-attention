# Copyright (c) 2023, Tri Dao.

from typing import Tuple
import math

import torch

from einops import rearrange, repeat

import rotary_emb


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), '... d two -> ... (d two)', two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, 's d -> s 1 (2 d)')
    sin = repeat(sin, 's d -> s 1 (2 d)')
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
                      x[..., ro_dim:]], dim=-1)


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (out_ro.chunk(2, dim=-1) if not interleaved
                      else (out_ro[..., ::2], out_ro[..., 1::2]))
        rotary_emb.apply_rotary(x1, x2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), o1, o2, False)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (do_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (do_ro[..., ::2], do_ro[..., 1::2]))
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (dx_ro.chunk(2, dim=-1) if not ctx.interleaved
                        else (dx_ro[..., ::2], dx_ro[..., 1::2]))
        rotary_emb.apply_rotary(do1, do2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), dx1, dx2, True)
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


apply_rotary_emb_func = ApplyRotaryEmb.apply


class ApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q_ro = qkv[:, :, 0, :, :rotary_dim]
        q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        rotary_emb.apply_rotary(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), q1, q2, False)
        k_ro = qkv[:, :, 1, :, :rotary_dim]
        k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        rotary_emb.apply_rotary(k1, k2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
                                rearrange(sin_k[:seqlen], 's d -> s 1 d'), k1, k2, False)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        _, seqlen, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[:, :, 0, :, :rotary_dim]
        dq1, dq2 = (dq_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dq_ro[..., ::2], dq_ro[..., 1::2]))
        rotary_emb.apply_rotary(dq1, dq2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), dq1, dq2, True)
        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dk_ro[..., ::2], dk_ro[..., 1::2]))
        rotary_emb.apply_rotary(dk1, dk2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
                                rearrange(sin_k[:seqlen], 's d -> s 1 d'), dk1, dk2, True)
        return dqkv, None, None, None, None, None


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

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, interleaved=False, scale_base=None, device=None):
        """
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                 / (1.4 * dim) if scale_base is not None else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

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
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                          - seqlen // 2) / self.scale_base)
                scale = self.scale.to(device=power.device) ** rearrange(power, 's -> s 1')
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset)
        if self.scale is None:
            return apply_rotary_emb_qkv_(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                None, None, self.interleaved
            )
        else:
            return apply_rotary_emb_qkv_(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:], self._sin_k_cached[seqlen_offset:],
                self.interleaved
            )
