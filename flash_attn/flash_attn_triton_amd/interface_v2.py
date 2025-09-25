import torch
import os
from typing import Optional, Union
from .fwd_prefill import attention_forward_prefill_triton_impl
from .fwd_decode import attention_forward_decode_triton_impl
from .bwd import attention_backward_triton_impl
from .utils import DEBUG, USE_EXP2, BWD_MODE, PHILOX_SEED, PHILOX_OFFSET


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
    gen_: Optional[torch.Tensor] = None,
):

    # Reject FP8 tensors (FA2 AMD path does not support FP8)
    if str(q.dtype).startswith("torch.float8"):
        raise NotImplementedError(
            "FP8 tensors are not supported in the AMD Triton FA2 interface. Use the FA3 path instead."
        )

    # Unsupported features assertions (keep behavior explicit like v3 shim)
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the AMD Triton FA2 interface (expected 0.0)."
        )

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::fwd inputs")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out, out.shape if out is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap:", softcap)
        print("return_softmax:", return_softmax)
    out = torch.zeros_like(q) if out is None else out.zero_()

    # Layout / shapes
    layout = "bshd"
    max_seqlen_q = q.shape[1]
    max_seqlen_k = k.shape[1]
    batch, _, nheads_q, _ = q.shape

    # Normalize / validate alibi
    if alibi_slopes is not None:
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        assert alibi_slopes.is_cuda and alibi_slopes.dim() == 2
        assert alibi_slopes.shape == (batch, nheads_q)

    # Dropout + RNG seed
    philox_seed, philox_offset = PHILOX_SEED, PHILOX_OFFSET
    rng_state = torch.as_tensor([philox_seed, philox_offset])

    # argument checks
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype == v.dtype
    assert out.shape[:-1] == q.shape[:-1] and out.shape[-1] == v.shape[-1]
    nheads_k = k.shape[2]
    assert (nheads_q % nheads_k) == 0

    # call implementation
    if DEBUG:
        print("Using Triton implementation")
    softmax_lse, sd_mask = attention_forward_prefill_triton_impl(
        q,
        k,
        v,
        out,
        softmax_scale,
        alibi_slopes,
        causal,
        window_size_left,
        window_size_right,
        None,
        layout,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        philox_seed,
        philox_offset,
        return_softmax,
        USE_EXP2,
        None,
        None,
        None,
    )

    if DEBUG:
        print("flash_attn_triton_amd.py::fwd outputs")
        print("o:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sd_mask:", sd_mask, sd_mask.shape if sd_mask is not None else None)
        print("rng_state:", rng_state)

    # --- Assertions (shape + dtype contracts) ---
    # out: (B, Sq, Hq, D)
    assert out.shape == q.shape, f"[fwd] out shape {out.shape} != q shape {q.shape}"
    # softmax_lse: (B, Hq, Sq)
    expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    assert (
        softmax_lse.shape == expected_lse_shape
    ), f"[fwd] softmax_lse shape {softmax_lse.shape} != {expected_lse_shape}"
    assert (
        softmax_lse.dtype == torch.float32
    ), f"[fwd] softmax_lse dtype {softmax_lse.dtype} != torch.float32"
    if return_softmax:
        # sd_mask: (B, Hq, Sq, Sk)
        assert sd_mask is not None, "[fwd] return_softmax=True but sd_mask is None"
        assert sd_mask.dim() == 4, f"[fwd] sd_mask dim {sd_mask.dim()} != 4"
        assert (
            sd_mask.shape[0] == q.shape[0]
            and sd_mask.shape[1] == q.shape[2]
            and sd_mask.shape[2] == q.shape[1]
        ), f"[fwd] sd_mask leading dims {sd_mask.shape[:3]} mismatch (B,Hq,Sq) {(q.shape[0], q.shape[2], q.shape[1])}"
    else:
        assert sd_mask is None, "[fwd] return_softmax=False but sd_mask is not None"

    return out, softmax_lse, sd_mask, rng_state


def bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_: Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
):
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the AMD Triton FA2 interface (expected 0.0)."
        )

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::bwd inputs")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)

    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()

    # get shape
    batch, _, nheads_q, _ = q.shape

    # Upstream change: base seeding logic on provided rng_state instead of dropout probability.
    if rng_state is not None:
        philox_seed, philox_offset = rng_state[0].item(), rng_state[1].item()
    else:
        philox_seed, philox_offset = None, None

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 2:
            pass
        elif alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        else:
            raise ValueError("Alibi can be (nheads,) or (batch_size, nheads).")

    # call implementation
    if DEBUG:
        print("Using Triton implementation")
    delta = attention_backward_triton_impl(
        do=dout,
        q=q,
        k=k,
        v=v,
        o=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
        sm_scale=softmax_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        layout="bshd",
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        seqused_q=None,
        seqused_k=None,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        use_exp2=USE_EXP2,
        mode=BWD_MODE,
    )

    if DEBUG:
        print("flash_attn_triton_amd.py::bwd outputs")
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)
    # --- Assertions ---
    assert dq.shape == q.shape, f"[bwd] dq shape {dq.shape} != q shape {q.shape}"
    assert dk.shape == k.shape, f"[bwd] dk shape {dk.shape} != k shape {k.shape}"
    assert dv.shape == v.shape, f"[bwd] dv shape {dv.shape} != v shape {v.shape}"
    # delta (softmax_d) : (B, Hq, Sq)
    expected_delta_shape = (q.shape[0], q.shape[2], q.shape[1])
    assert (
        delta.shape == expected_delta_shape
    ), f"[bwd] delta shape {delta.shape} != {expected_delta_shape}"
    return dq, dk, dv, delta


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    block_table_: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
    gen_: Optional[torch.Tensor] = None,
):

    if str(q.dtype).startswith("torch.float8"):
        raise NotImplementedError(
            "FP8 tensors are not supported in the AMD Triton FA2 interface (varlen_fwd). Use the FA3 path instead."
        )

    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in varlen_fwd (expected 0.0)."
        )
    if leftpad_k is not None:
        raise NotImplementedError(
            "leftpad_k is not supported in AMD Triton FA2 varlen_fwd."
        )
    if block_table_ is not None:
        raise NotImplementedError(
            "block_table / paged attention is not supported in AMD Triton FA2 varlen_fwd."
        )
    if seqused_k is not None:
        raise NotImplementedError(
            "seqused_k is not supported in AMD Triton FA2 varlen_fwd."
        )

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::varlen_fwd")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("gen_:", gen_)
    out = torch.zeros_like(q) if out is None else out.zero_()

    # Layout and basic info for varlen
    layout = "thd"
    batch = len(cu_seqlens_q) - 1
    _, nheads_q, _ = q.shape

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        assert alibi_slopes.is_cuda and alibi_slopes.dim() == 2
        assert alibi_slopes.shape == (batch, nheads_q)

    philox_seed, philox_offset = PHILOX_SEED, PHILOX_OFFSET
    rng_state = torch.as_tensor([philox_seed, philox_offset])

    # Inline checks (subset appropriate for varlen)
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype == v.dtype
    assert out.shape == q.shape
    nheads_k = k.shape[1]
    assert (nheads_q % nheads_k) == 0

    # call implementation
    if DEBUG:
        print("Using Triton implementation")
    softmax_lse, sd_mask = attention_forward_prefill_triton_impl(
        q,
        k,
        v,
        out,
        softmax_scale,
        alibi_slopes,
        causal,
        window_size_left,
        window_size_right,
        None,
        layout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        philox_seed,
        philox_offset,
        return_softmax,
        USE_EXP2,
        None,
        None,
        None,
    )

    if DEBUG:
        print("varlen_fwd outputs")
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sd_mask:", sd_mask, sd_mask.shape if sd_mask is not None else None)
    # --- Assertions ---
    # out: (Total_Q, Hq, D)
    assert (
        out.shape == q.shape
    ), f"[varlen_fwd] out shape {out.shape} != q shape {q.shape}"
    # softmax_lse: (Hq, Total_Q)
    expected_lse_shape = (q.shape[1], q.shape[0])
    assert (
        softmax_lse.shape == expected_lse_shape
    ), f"[varlen_fwd] softmax_lse shape {softmax_lse.shape} != {expected_lse_shape}"
    assert (
        softmax_lse.dtype == torch.float32
    ), f"[varlen_fwd] softmax_lse dtype {softmax_lse.dtype} != torch.float32"
    if return_softmax:
        # sd_mask expected: (B, Hq, max_seqlen_q, max_seqlen_k)
        assert (
            sd_mask is not None
        ), "[varlen_fwd] return_softmax=True but sd_mask is None"
        assert sd_mask.dim() == 4, f"[varlen_fwd] sd_mask dim {sd_mask.dim()} != 4"
        assert sd_mask.shape[0] == (
            len(cu_seqlens_q) - 1
        ), f"[varlen_fwd] sd_mask batch {sd_mask.shape[0]} != {len(cu_seqlens_q)-1}"
        assert (
            sd_mask.shape[1] == q.shape[1]
        ), f"[varlen_fwd] sd_mask nheads {sd_mask.shape[1]} != {q.shape[1]}"
    else:
        assert (
            sd_mask is None
        ), "[varlen_fwd] return_softmax=False but sd_mask is not None"
    return out, softmax_lse, sd_mask, rng_state


def varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_: Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
):
    if str(q.dtype).startswith("torch.float8"):
        raise NotImplementedError(
            "FP8 tensors are not supported in the AMD Triton FA2 interface (varlen_bwd). Use the FA3 path instead."
        )
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in varlen_bwd (expected 0.0)."
        )

    if DEBUG:
        print()
        print("varlen_bwd")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("out:", out)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)

    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()

    # get shape
    batch = len(cu_seqlens_q) - 1
    _, nheads_q, _ = q.shape

    # Upstream change: base seeding logic on provided rng_state instead of dropout probability.
    if rng_state is not None:
        philox_seed, philox_offset = rng_state[0].item(), rng_state[1].item()
    else:
        philox_seed, philox_offset = None, None

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 2:
            pass
        elif alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        else:
            raise ValueError("Alibi can be (nheads,) or (batch_size, nheads).")

    # call implementation
    if DEBUG:
        print("Using Triton implementation")
    delta = attention_backward_triton_impl(
        do=dout,
        q=q,
        k=k,
        v=v,
        o=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
        sm_scale=softmax_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        layout="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=None,
        seqused_k=None,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        use_exp2=USE_EXP2,
        mode=BWD_MODE,
    )

    if DEBUG:
        print("varlen_bwd outputs")
        print("delta:", delta, delta.shape)
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)
    # --- Assertions ---
    assert dq.shape == q.shape, f"[varlen_bwd] dq shape {dq.shape} != q shape {q.shape}"
    assert dk.shape == k.shape, f"[varlen_bwd] dk shape {dk.shape} != k shape {k.shape}"
    assert dv.shape == v.shape, f"[varlen_bwd] dv shape {dv.shape} != v shape {v.shape}"
    expected_delta_shape = (q.shape[1], q.shape[0])  # (Hq, Total_Q)
    assert (
        delta.shape == expected_delta_shape
    ), f"[varlen_bwd] delta shape {delta.shape} != {expected_delta_shape}"
    return dq, dk, dv, delta


def fwd_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: Optional[Union[(int, torch.Tensor)]],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    rotary_interleaved: bool,
    num_splits: int,
):

    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in fwd_kvcache (expected 0.0)."
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "num_splits > 1 not supported in AMD Triton FA2 fwd_kvcache."
        )

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::fwd_kvcache inputs")
        print("q:", q, q.shape)
        print("k_cache:", k_cache, k_cache.shape)
        print("v_cache:", v_cache, v_cache.shape)
        print("k:", k, k.shape if k is not None else None)
        print("v:", v, v.shape if v is not None else None)
        print("cache_seqlens:", cache_seqlens)
        print("rotary_cos:", rotary_cos)
        print("rotary_sin:", rotary_sin)
        print("cache_batch_idx:", cache_batch_idx)
        print("cache_leftpad:", cache_leftpad)
        print("block_table:", block_table)
        print("alibi_slopes:", alibi_slopes)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap:", softcap)
        print("rotary_interleaved:", rotary_interleaved)
        print("num_splits:", num_splits)

    # output
    out = torch.zeros_like(q) if out is None else out.zero_()

    # Basic layout info for decode path
    layout = "bshd"
    max_seqlen_q = q.shape[1]
    max_seqlen_k = k_cache.shape[1]
    cache_seqlens_tensor = (
        torch.tensor(cache_seqlens, device=q.device)
        if isinstance(cache_seqlens, int)
        else cache_seqlens
    )
    window_left = (
        int(window_size_left.item())
        if isinstance(window_size_left, torch.Tensor)
        else window_size_left
    )
    window_right = (
        int(window_size_right.item())
        if isinstance(window_size_right, torch.Tensor)
        else window_size_right
    )

    k_new = k
    v_new = v

    # get shape
    batch, _, nheads_q, _ = q.shape

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        assert alibi_slopes.is_cuda and alibi_slopes.dim() == 2
        assert alibi_slopes.shape == (batch, nheads_q)

    # launch kernel
    if DEBUG:
        print("Using Triton implementation")
    softmax_lse = attention_forward_decode_triton_impl(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        out,
        softmax_scale,
        causal,
        window_left,
        window_right,
        alibi_slopes,
        layout,
        cache_seqlens_tensor,
        cache_batch_idx,
        block_table,
        None,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=rotary_interleaved,
    )

    if DEBUG:
        print("out:", out, out.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
    # --- Assertions ---
    assert (
        out.shape == q.shape
    ), f"[fwd_kvcache] out shape {out.shape} != q shape {q.shape}"
    expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    assert (
        softmax_lse.shape == expected_lse_shape
    ), f"[fwd_kvcache] softmax_lse shape {softmax_lse.shape} != {expected_lse_shape}"
    assert (
        softmax_lse.dtype == torch.float32
    ), f"[fwd_kvcache] softmax_lse dtype {softmax_lse.dtype} != torch.float32"
    return out, softmax_lse
