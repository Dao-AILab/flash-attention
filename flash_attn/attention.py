import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def _flash_attn(q, k, v, mask=None, bias=None):
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    k_no_heads, k_n, k_c = k.shape[-3:]

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    k_batch_size = k.shape[0]
    
    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])
    k = k.reshape(-1, *k.shape[-2:])
    v = v.reshape(-1, *v.shape[-2:])
    
    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    k_max_s = k_n
    k_cu_seqlens = torch.arange(
        0, (k_batch_size + 1) * k_n, step=k_n, dtype=torch.int32, device=k.device
    )

    if mask is not None:
        mask_heads, tgt_len, src_len = mask.shape[-3:]
        mask = mask.reshape(-1 , mask_heads, tgt_len, src_len).contiguous()

    if bias is not None:
        bias_heads, tgt_len, src_len = bias.shape[-3:]
        bias = bias.reshape(-1 , bias_heads, tgt_len, src_len).contiguous()

    out = flash_attn_unpadded_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        attn_mask=mask,
        attn_bias=bias,
        dropout_p = 0.,
        softmax_scale = 1., # q has been scaled already
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c) 
    return out
