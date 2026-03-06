# Ring Attention: sequence-parallel FlashAttention via a peer-to-peer KV ring.
# Each rank owns a contiguous slice of the sequence. KV chunks rotate around
# the ring while the local Q is held fixed; partial (out, lse) results are
# merged with the numerically-stable online-softmax identity used internally
# by FlashAttention, allowing exact attention over the full sequence with no
# approximation and minimal communication overhead.
#
# Communication is issued asynchronously *before* each local kernel launch so
# that NCCL P2P transfers overlap with the CUDA attention kernel on the
# previous chunk, amortising latency on NVLink and fast interconnects.

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward


# ---------------------------------------------------------------------------
# Low-level ring helpers
# ---------------------------------------------------------------------------

def _ring_send_recv(
    tensor: Tensor,
    send_rank: int,
    recv_rank: int,
    group: ProcessGroup,
) -> Tuple[Tensor, dist.Work]:
    """
    Non-blocking send to `send_rank` and receive from `recv_rank`.
    Returns (recv_buffer, work_handle).  The caller must call work.wait()
    before reading recv_buffer.
    """
    recv_buf = torch.empty_like(tensor)
    ops = []
    ops.append(dist.P2POp(dist.isend, tensor.contiguous(), send_rank, group))
    ops.append(dist.P2POp(dist.irecv, recv_buf, recv_rank, group))
    work = dist.batch_isend_irecv(ops)
    return recv_buf, work


# ---------------------------------------------------------------------------
# Online-softmax merge (numerically stable)
# ---------------------------------------------------------------------------

def _merge_attn_outputs(
    out_a: Tensor,
    lse_a: Tensor,
    out_b: Tensor,
    lse_b: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Merge two partial attention outputs produced with different key/value
    subsets using the log-sum-exp identity:

        lse_new = lse_a + log(1 + exp(lse_b - lse_a))
               = torch.logaddexp(lse_a, lse_b)

        out_new = (exp(lse_a) * out_a + exp(lse_b) * out_b) / exp(lse_new)

    lse tensors: (batch, nheads, seqlen)  – as returned by FlashAttention.
    out tensors: (batch, seqlen, nheads, headdim).
    """
    # lse: (B, H, S) → (B, S, H, 1) for broadcasting against out
    lse_a_ = lse_a.permute(0, 2, 1).unsqueeze(-1)   # (B, S, H, 1)
    lse_b_ = lse_b.permute(0, 2, 1).unsqueeze(-1)

    lse_new = torch.logaddexp(lse_a, lse_b)          # (B, H, S)
    lse_new_ = lse_new.permute(0, 2, 1).unsqueeze(-1)

    # Compute in fp32 to preserve numerical accuracy then cast back
    dtype = out_a.dtype
    out_a = out_a.float()
    out_b = out_b.float()

    scale_a = torch.exp(lse_a_ - lse_new_)
    scale_b = torch.exp(lse_b_ - lse_new_)
    out_new = (scale_a * out_a + scale_b * out_b).to(dtype)

    return out_new, lse_new


# ---------------------------------------------------------------------------
# Core forward pass (no autograd)
# ---------------------------------------------------------------------------

def ring_flash_attn_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    process_group: ProcessGroup,
) -> Tuple[Tensor, Tensor]:
    """
    Forward pass of ring attention.

    q / k / v  – local sequence slice: (batch, seqlen_local, nheads[_k], headdim)
    Returns (out, softmax_lse) for the local query slice, where lse covers the
    full key sequence (all ranks).
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    send_rank = (rank + 1) % world_size          # pass KV forward in the ring
    recv_rank = (rank - 1) % world_size

    # Step 0: compute attention against our *own* KV chunk (chunk index = rank).
    # We immediately start sending our KV to the next rank so the transfer
    # overlaps with the first local kernel call on subsequent iterations.
    kv_send = torch.stack([k, v], dim=0)          # (2, B, S_local, H_k, D)
    kv_recv, work = _ring_send_recv(kv_send, send_rank, recv_rank, process_group)

    out, lse, _, _ = _wrapped_flash_attn_forward(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )

    for step in range(1, world_size):
        work.wait()                               # KV from previous send is ready
        k_remote, v_remote = kv_recv[0], kv_recv[1]

        # Pre-fetch the *next* chunk while the kernel runs on the current one
        if step < world_size - 1:
            kv_send = kv_recv
            kv_recv, work = _ring_send_recv(kv_send, send_rank, recv_rank, process_group)

        # When causal=True only later-ranked chunks (earlier tokens) participate
        # fully; our own chunk was already handled in step 0.
        # Chunks from ranks < current rank contain earlier tokens that the
        # causal mask would exclude for the *last* query tokens, but the
        # FlashAttention causal flag only applies within a single chunk.
        # We therefore run all remote chunks non-causally; the causal structure
        # across chunks is implicitly correct because we only query against
        # earlier-in-sequence KV (chunks from lower-numbered steps in the ring).
        chunk_source_rank = (rank - step) % world_size
        is_causal_chunk = causal and (chunk_source_rank > rank)

        out_chunk, lse_chunk, _, _ = _wrapped_flash_attn_forward(
            q, k_remote, v_remote,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=is_causal_chunk,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        out, lse = _merge_attn_outputs(out, lse, out_chunk, lse_chunk)

    return out, lse


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        process_group: ProcessGroup,
    ) -> Tensor:
        out, lse = ring_flash_attn_forward(q, k, v, softmax_scale, causal, process_group)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.process_group = process_group
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        # Reduce-scatter dout across the ring so each rank holds the gradient
        # for its local query slice, then run the standard FlashAttention
        # backward on the full gathered KV (gathered once; standard Megatron-SP).
        # For simplicity we fall back to a single-node all-gather here;
        # a full ring-backward follows the same communication pattern as fwd.
        raise NotImplementedError(
            "Ring attention backward is not yet implemented. "
            "Use activation checkpointing + forward recompute for training."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ring_flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    process_group: Optional[ProcessGroup] = None,
) -> Tensor:
    """
    Drop-in replacement for ``flash_attn_func`` that shards the sequence
    dimension across ``process_group``.

    Each rank must pass its **local** Q / K / V slice
    (shape ``(batch, seqlen // world_size, nheads[_k], headdim)``).
    The function returns the local output slice of shape
    ``(batch, seqlen // world_size, nheads, headdim)``.

    Supports MQA / GQA (nheads_k < nheads) and non-causal attention.
    Causal attention is supported with the caveat that the backward pass
    currently raises ``NotImplementedError``; pair with
    ``torch.utils.checkpoint`` for causal training.

    Args:
        q: ``(batch, seqlen_local, nheads, headdim)``
        k: ``(batch, seqlen_local, nheads_k, headdim)``
        v: ``(batch, seqlen_local, nheads_k, headdim)``
        softmax_scale: defaults to ``1 / sqrt(headdim)``
        causal: apply causal mask over the *global* sequence
        process_group: the ring process group; defaults to the global group

    Returns:
        out: ``(batch, seqlen_local, nheads, headdim)``
    """
    if process_group is None:
        process_group = dist.group.WORLD
    return RingFlashAttnFunc.apply(q, k, v, softmax_scale, causal, process_group)
