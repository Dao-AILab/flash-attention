# Copyright (c) 2022, Tri Dao.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

try:
    from flash_attn.ops.fused_dense import FusedDense, ColumnParallelLinear, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    import ft_attention
except ImportError:
    ft_attention = None


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
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, 'FlashAttention is not installed'
        assert flash_attn_qkvpacked_func is not None, 'FlashAttention is not installed'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen, self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
        else:
            return flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0,
                                             softmax_scale=self.softmax_scale, causal=causal)


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
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert flash_attn_varlen_kvpacked_func is not None, 'FlashAttention is not installed'
        assert flash_attn_kvpacked_func is not None, 'FlashAttention is not installed'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_kvpacked_func(
                q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            return flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0,
                                            causal=causal, softmax_scale=self.softmax_scale)


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
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
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
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen_k), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen_q, seqlen_k), -10000.0,
                                                device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)
    """
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size, inference_params.max_sequence_len, 2,
            num_heads, head_dim, dtype=kv.dtype, device=kv.device
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            # For FT, k_cache has shape (b, h, headdim / packsize, s, packsize)
            # where packsize = 4 if fp32, 8 if fp16 or bf16.
            # v_cache has shape (b, h, s, headdim)
            k_cache, v_cache = inference_params.key_value_memory_dict[layer_idx]
            kv_cache = None
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0])
    assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2])
    # Copy key and values.
    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        kv = kv_cache[batch_start:batch_end, :sequence_end, ...]
        return kv
    else:
        assert inference_params.sequence_len_offset == 0
        # FT kernel requires different layouts for the k_cache and v_cache.
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        packsize = 4 if kv.dtype == torch.float32 else 8
        if kv_cache is not None:
            kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
            k_cache = rearrange(kv_cache[:, :, 0], 'b s h (d packsize) -> b h d s packsize',
                                packsize=packsize).contiguous()
            v_cache = rearrange(kv_cache[:, :, 1], 'b s h d -> b h s d').contiguous()
            inference_params.key_value_memory_dict[layer_idx] = (k_cache, v_cache)
        else:
            k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
                kv[:, :, 0], 'b s h (d packsize) -> b h d s packsize', packsize=packsize
            )
            v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(
                kv[:, :, 1], 'b s h d -> b h s d'
            )
        return kv


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, cross_attn=False,
                 qkv_proj_bias=True, out_proj_bias=True,
                 dropout=0.0, softmax_scale=None, causal=False, layer_idx=None, dwconv=False,
                 rotary_emb_dim=0, rotary_emb_base=10000.0, rotary_emb_scale_base=None,
                 rotary_emb_interleaved=False, fused_bias_fc=False, use_flash_attn=False,
                 return_residual=False, checkpointing=False, device=None, dtype=None) -> None:
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
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert not cross_attn, 'MHA with rotary embedding does not support cross-attention yet'
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, base=rotary_emb_base,
                                              scale_base=rotary_emb_scale_base,
                                              interleaved=rotary_emb_interleaved, device=device)

        if fused_bias_fc and FusedDense is None:
            raise ImportError('fused_dense is not installed')
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (LinearResidual if not fused_bias_fc
                            else partial(FusedDense, return_residual=True))
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        if not self.cross_attn:
            if not self.return_residual:
                self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=qkv_proj_bias,
                                       **factory_kwargs)
            else:
                self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=qkv_proj_bias,
                                             **factory_kwargs)
            if self.dwconv:
                self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                            groups=3 * embed_dim)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            if not self.return_residual:
                self.Wkv = linear_cls(embed_dim, 2 * embed_dim, bias=qkv_proj_bias,
                                      **factory_kwargs)
            else:
                self.Wkv = linear_resid_cls(embed_dim, 2 * embed_dim, bias=qkv_proj_bias,
                                            **factory_kwargs)
            if self.dwconv:
                self.dwconv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2,
                                          groups=embed_dim)
                self.dwconv_kv = nn.Conv1d(2 * embed_dim, 2 * embed_dim, kernel_size=3, padding=2,
                                          groups=2 * embed_dim)
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                                     attention_dropout=dropout)
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, fused_ft_kernel=True):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if not fused_ft_kernel:
            return torch.empty(batch_size, max_seqlen, 2, self.num_heads, self.head_dim,
                               dtype=dtype, device=device)
        else:
            assert dtype in [torch.float16, torch.bfloat16, torch.float32]
            packsize = 4 if dtype == torch.float32 else 8
            assert self.head_dim % packsize == 0
            k_cache = torch.empty(batch_size, self.num_heads, self.head_dim // packsize, max_seqlen,
                                  packsize, dtype=dtype, device=device)
            v_cache = torch.empty(batch_size, self.num_heads, max_seqlen, self.head_dim,
                                  dtype=dtype, device=device)
            return k_cache, v_cache

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)
        """
        assert not self.dwconv, 'Generation does not support dwconv yet'
        assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None, max_seqlen=None,
                mixer_subset=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = ({'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen, **kwargs}
                  if self.use_flash_attn else {'key_padding_mask': key_padding_mask, **kwargs})
        if not self.cross_attn:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                                'b d s -> b s d').contiguous()
            qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)
            if inference_params is None:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv)
                if not self.checkpointing:
                    context = self.inner_attn(qkv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
            else:
                if (not inference_params.fused_ft_kernel) or inference_params.sequence_len_offset == 0:
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(qkv, seqlen_offset=inference_params.sequence_len_offset)
                    q = qkv[:, :, 0]
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
                    # If we're processing the prompt, causal=None (use self.causal).
                    # If we're decoding, then causal=False.
                    causal = None if inference_params.sequence_len_offset == 0 else False
                    context = self.inner_cross_attn(q, kv, causal=causal)
                else:
                    assert inference_params.fused_ft_kernel
                    assert ft_attention is not None
                    batch_start = inference_params.batch_size_offset
                    batch_end = batch_start + qkv.shape[0]
                    k_cache, v_cache = inference_params.key_value_memory_dict[self.layer_idx]
                    lengths_per_sample = (inference_params.lengths_per_sample[batch_start:batch_end]
                                          if inference_params.lengths_per_sample is not None else None)
                    rotary_emb_base = self.rotary_emb.base if self.rotary_emb_dim > 0 else 0
                    context = ft_attention.single_query_attention(
                        *rearrange(qkv, 'b 1 three h d -> b three h d').unbind(dim=1),
                        k_cache[batch_start:batch_end],
                        v_cache[batch_start:batch_end],
                        lengths_per_sample,
                        None,  # rotary_cos_
                        None,  # rotary_sin_
                        None,  # nnz_head_idx
                        inference_params.sequence_len_offset,
                        self.rotary_emb_dim, rotary_emb_base,
                        # neox_rotary_style
                        (not self.rotary_emb.interleaved) if self.rotary_emb_dim > 0 else True
                    )
                    context = rearrange(context, 'b h d -> b 1 h d')
        else:
            if not self.return_residual:
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                kv = self.Wkv(x_kv if x_kv is not None else x)
            else:
                if x_kv is not None:
                    kv, x_kv = self.Wkv(x_kv)
                else:
                    kv, x = self.Wkv(x)
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
            kv = rearrange(kv, '... (two h d) -> ... two h d', two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(self.dwconv_q(rearrange(q, 'b s d -> b d s'))[..., :-2],
                              'b d s -> b s d').contiguous()
                kv = rearrange(self.dwconv_kv(rearrange(kv, 'b s d -> b d s'))[..., :-2],
                               'b d s -> b s d').contiguous()
            if inference_params is None:
                if not self.checkpointing:
                    context = self.inner_cross_attn(q, kv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
            else:
                kv = self._update_kv_cache(kv)
                context = self.inner_cross_attn(q, kv, causal=False)
        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        return out if not self.return_residual else (out, x)


class ParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, process_group, qkv_proj_bias=True, out_proj_bias=True,
                 dropout=0.0, softmax_scale=None, causal=False, layer_idx=None,
                 rotary_emb_dim=0, rotary_emb_base=10000.0, rotary_emb_scale_base=None,
                 rotary_emb_interleaved=False, use_flash_attn=False, checkpointing=False,
                 sequence_parallel=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, base=rotary_emb_base,
                                              scale_base=rotary_emb_scale_base,
                                              interleaved=rotary_emb_interleaved, device=device)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError('fused_dense is not installed')
        self.Wqkv = ColumnParallelLinear(embed_dim, 3 * embed_dim, process_group,
                                         bias=qkv_proj_bias,
                                         sequence_parallel=sequence_parallel, **factory_kwargs)
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                                     attention_dropout=dropout)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, process_group,
                                          bias=out_proj_bias,
                                          sequence_parallel=sequence_parallel, **factory_kwargs)

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, '(b s) (three h d) -> b s three h d', s=seqlen, three=3,
                            d=self.head_dim)
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
        else:
            if (not inference_params.fused_ft_kernel) or inference_params.sequence_len_offset == 0:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=inference_params.sequence_len_offset)
                q = qkv[:, :, 0]
                assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                # If we're processing the prompt, causal=None (use self.causal).
                # If we're decoding, then causal=False.
                causal = None if inference_params.sequence_len_offset == 0 else False
                context = self.inner_cross_attn(q, kv, causal=causal)
            else:
                assert inference_params.fused_ft_kernel
                assert ft_attention is not None
                batch_start = inference_params.batch_size_offset
                batch_end = batch_start + qkv.shape[0]
                k_cache, v_cache = inference_params.key_value_memory_dict[self.layer_idx]
                lengths_per_sample = (inference_params.lengths_per_sample[batch_start:batch_end]
                                      if inference_params.lengths_per_sample is not None else None)
                rotary_emb_base = self.rotary_emb.base if self.rotary_emb_dim > 0 else 0
                context = ft_attention.single_query_attention(
                    *rearrange(qkv, 'b 1 three h d -> b three h d').unbind(dim=1),
                    k_cache[batch_start:batch_end],
                    v_cache[batch_start:batch_end],
                    lengths_per_sample,
                    None,  # rotary_cos_
                    None,  # rotary_sin_
                    None,  # nnz_head_idx
                    inference_params.sequence_len_offset,
                    self.rotary_emb_dim, rotary_emb_base,
                    # neox_rotary_style
                    (not self.rotary_emb.interleaved) if self.rotary_emb_dim > 0 else True
                )
                context = rearrange(context, 'b h d -> b 1 h d')
        if seqlen is None:
            context = rearrange(context, 'b s h d -> b s (h d)')
        else:
            context = rearrange(context, 'b s h d -> (b s) (h d)')
        out = self.out_proj(context)
        return out
