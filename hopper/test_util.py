import math

import torch
from einops import rearrange, repeat

from padding import pad_input, unpad_input


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random", zero_lengths=False):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)

    if zero_lengths:
        # Generate zero-lengths every 5 batches and the last batch.
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, qv=None, kvpacked=False, qkvpacked=False,
    query_unused_mask=None, key_unused_mask=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d_v)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    d_v = v.shape[-1]
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q, query_padding_mask, query_unused_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if qv is not None else None
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        seqused_q = None
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...") if qv is not None else None

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(
            k, key_padding_mask, key_unused_mask
        )
        v_unpad, *rest = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            qv_unpad.detach()  if qv is not None else None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            qv.detach() if qv is not None else None,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
):
    if cp_world_size > 1:
        return construct_cp_mask(
            seqlen_q,
            seqlen_k,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqlen_k=cp_tot_seqlen_k,
            window_size=window_size,
            sink_token_length=sink_token_length,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            key_leftpad=key_leftpad,
            device=device,
        )
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            torch.logical_and(col_idx < row_idx + sk - sq - window_size[0], col_idx >= sink_token_length),
        )

def construct_cp_mask(
    seqlen_q,
    seqlen_k,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
    window_size=(-1, -1),  # -1 means infinite window size
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    """
    Construct attention mask for context parallelism (DCP).

    This function creates a mask that handles both local windowing and context parallelism.
    For DCP, each rank only sees a subset of KV tokens (interleaved), and the mask
    must account for the global positions when applying causal or windowing constraints.

    Args:
        seqlen_q: Length of query sequence
        seqlen_k: Length of key sequence (local to this rank)
        cp_world_size: Number of context parallel ranks
        cp_rank: Current rank ID (0 to cp_world_size-1)
        cp_tot_seqlen_k: Total lengths of key sequence in cp world
        window_size: (left_window, right_window), -1 = infinite
        sink_token_length: Number of "sink" tokens that can always be attended to
        query_padding_mask: Which query positions are valid
        key_padding_mask: Which key positions are valid
        key_leftpad: Left padding for keys (per batch)
        device: Device to place tensors on

    Returns:
        mask: Boolean tensor of shape [seqlen_q, seqlen_k] where True = masked out
    """
    # Create position indices
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")  # [seqlen_q, 1]
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)  # [seqlen_k]

    # Handle left padding if present
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)

    # Calculate effective sequence lengths
    sk = (
        cp_tot_seqlen_k[0]
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1") * cp_world_size
    )
    sq = (
        torch.tensor(seqlen_q, device=device, dtype=torch.long)  # Global seqlen_k for DCP
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )

    if cp_world_size > 1:
        # DCP masking logic
        # Convert local K indices to global (absolute) K positions
        # local_k_idx * cp_world_size + cp_rank gives the global position
        abs_k_idx = col_idx * cp_world_size + cp_rank  # [seqlen_k] -> global positions

        # Query global positions: row_idx + seqlen_k_global - seqlen_q
        # This handles the case where query and key sequences might have different lengths
        abs_q_idx = row_idx + sk - sq  # [seqlen_q, 1] -> global query positions

        if window_size[0] < 0:
            # Infinite left window - essentially causal masking with right window
            mask = abs_k_idx > abs_q_idx + window_size[1]
        else:
            # Finite window - sliding window attention
            # Right boundary: abs_k_idx > abs_q_idx + window_size[1]
            right_mask = abs_k_idx > torch.minimum(abs_q_idx + window_size[1], sk)

            # Left boundary: abs_k_idx < abs_q_idx - window_size[0], but exclude sink tokens
            left_mask = torch.logical_and(
                abs_k_idx < abs_q_idx - window_size[0],
                abs_k_idx >= sink_token_length
            )

            mask = torch.logical_or(right_mask, left_mask)

    else:
        # Non-DCP case: fall back to original construct_local_mask logic
        if window_size[0] < 0:
            mask = col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk_local = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk // cp_world_size
            mask = torch.logical_or(
                col_idx > torch.minimum(row_idx + sk_local - sq + window_size[1], sk_local),
                torch.logical_and(
                    col_idx < row_idx + sk_local - sq - window_size[0],
                    col_idx >= sink_token_length
                ),
            )

    return mask


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),  # -1 means infinite window size
    sink_token_length=0,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_kv, head_dim)
        v: (batch_size, seqlen_k, nheads_kv, head_dim_v)
        qv: (batch_size, seqlen_q, nheads, head_dim_v)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
        s_aux: (nheads)
        cp_world_size: Number of context parallel ranks
        cp_rank: Current rank ID (0 to cp_world_size-1)
        cp_tot_seqlen_k:  (batch_size) total seqlen of k/v in cp world
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim_v)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    batch_size = q.shape[0]
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    nheads = q.shape[2]
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        qv = qv.float() if qv is not None else None
        s_aux = s_aux.float() if s_aux is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
        qv = (qv.float() * q_descale).to(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None:
        scores = scores + torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqlen_k=cp_tot_seqlen_k,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    if s_aux is not None:
        # concatenate sink column before softmax
        s_aux = s_aux.reshape(1, nheads, 1, 1).expand(batch_size, -1, seqlen_q, -1)
        scores = torch.cat([scores, s_aux], dim=-1)
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if s_aux is not None:
        # remove sink column
        attention = attention[..., :-1]
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    # Without this we might get NaN in dv
    if key_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)
