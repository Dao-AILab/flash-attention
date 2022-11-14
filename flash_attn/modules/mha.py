# Copyright (c) 2022, Tri Dao.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
except ImportError:
    flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func = None, None

try:
    from flash_attn.ops.flash_attn_triton import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

try:
    from flash_attn.ops.fused_dense import FusedDenseTD, FusedDenseResidual
except ImportError:
    FusedDenseTD, FusedDenseResidual = None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 triton=False, device=None, dtype=None):
        super().__init__()
        if attention_dropout != 0.0 or not triton:
            assert flash_attn_unpadded_qkvpacked_func is not None, 'FlashAttention is not installed'
        if attention_dropout == 0.0 and triton:
            assert flash_attn_qkvpacked_func is not None, 'FlashAttention Triton is not installed'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.triton = triton

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        if self.triton and (self.dropout_p == 0 or not self.training):  # Triton version doesn't support dropout
            output = flash_attn_qkvpacked_func(qkv, None, self.causal, self.softmax_scale)
        else:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=qkv.device)
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=self.causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 triton=False, device=None, dtype=None):
        super().__init__()
        if attention_dropout != 0.0 or not triton:
            assert flash_attn_unpadded_kvpacked_func is not None, 'FlashAttention is not installed'
        if attention_dropout == 0.0 and triton:
            assert flash_attn_kvpacked_func is not None, 'FlashAttention Triton is not installed'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.triton = triton

    def forward(self, q, kv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]
        if self.triton and (self.dropout_p == 0.0 or not self.training):  # Triton version doesn't support dropout
            output = flash_attn_kvpacked_func(q, kv, None, self.causal, self.softmax_scale)
        else:
            q = rearrange(q, 'b s ... -> (b s) ...')
            kv = rearrange(kv, 'b s ... -> (b s) ...')
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q,
                                        dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k,
                                        dtype=torch.int32, device=kv.device)
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=self.causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if self.causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, kv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if self.causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen_q, seqlen_k), -10000.0,
                                                device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDenseResidual.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, cross_attn=False, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, dwconv=False, rotary_emb_dim=0,
                 fused_bias_fc=False, use_flash_attn=False, return_residual=False,
                 checkpointing=False, device=None, dtype=None) -> None:
        """
            return_residual: whether to return the input x along with the output. This is for
                performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert not cross_attn, 'MHA with rotary embedding does not support cross-attention yet'
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim)

        if fused_bias_fc and FusedDenseTD is None:
            raise ImportError('fused_dense is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDenseTD
        linear_resid_cls = LinearResidual if not fused_bias_fc else FusedDenseResidual
        if not self.cross_attn:
            if not self.return_residual:
                self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
            else:
                self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
            if self.dwconv:
                self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                            groups=3 * embed_dim)
            inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        else:
            # TODO: use the residual linear class for Wq
            self.Wq = linear_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.Wkv = linear_cls(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
            if self.dwconv:
                self.dwconv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2,
                                        groups=embed_dim)
                self.dwconv_kv = nn.Conv1d(2 * embed_dim, 2 * embed_dim, kernel_size=3, padding=2,
                                        groups=2 * embed_dim)
            inner_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout, **factory_kwargs)
        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, x_kv=None):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
        """
        if not self.cross_attn:
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                                'b d s -> b s d').contiguous()
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv)
            else:
                # context = torch.utils.checkpoint.checkpoint(self._inner_attention, qkv)
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv)
        else:
            q = rearrange(self.Wq(x), 'b s (h d) -> b s h d', h=self.num_heads)
            kv = rearrange(self.Wkv(x if x_kv is None else x_kv), 'b s (two h d) -> b s two h d',
                           two=2, h=self.num_heads)
            if self.dwconv:
                q = rearrange(self.dwconv_q(rearrange(q, 'b s d -> b d s'))[..., :-2],
                              'b d s -> b s d').contiguous()
                kv = rearrange(self.dwconv_kv(rearrange(kv, 'b s d -> b d s'))[..., :-2],
                               'b d s -> b s d').contiguous()
            if not self.checkpointing:
                context = self.inner_attn(q, kv)
            else:
                # context = torch.utils.checkpoint.checkpoint(self._inner_attention, qkv)
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, q, kv)
        out = self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))
        return out if not self.return_residual else (out, x)
