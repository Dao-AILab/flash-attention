"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Callable, Optional
from functools import partial
import math
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

# Import data structures from block_sparsity
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute import utils


@cute.jit
def load_block_list(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    first_block_preloaded: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    use_tma_q: cutlass.Constexpr,
    tma_q_bytes: cutlass.Constexpr,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Iterate over the sparse blocks and load K, V (and Q) into the pipeline.
    for the intra_wg_overlap case, we overlap the loads of K and V. And this
    means we need to pipeline the last V load from the partial block case,
    with the loads for the full blocks. Set first_block_preloaded when the
    caller has already issued the first K load for the list.

    Note:
        we iterate along the block_n indices in reverse.

    Returns:
        Updated kv_producer_state after processing the block list.

    """
    if block_count > 0:
        if const_expr(not intra_wg_overlap):
            # Peel first iteration: the first block may need to load Q alongside K,
            # Parameters are already Constexpr, so no need to wrap in const_expr()
            n_block_first = block_indices[block_count - 1]
            extra_tx = tma_q_bytes if const_expr(load_q_with_first) and const_expr(use_tma_q) else 0
            pipeline_k.producer_acquire(kv_producer_state, extra_tx_count=extra_tx)

            if const_expr(load_q_with_first and use_tma_q):
                load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))

            load_K(src_idx=n_block_first, producer_state=kv_producer_state)
            pipeline_v.producer_acquire(kv_producer_state)
            load_V(src_idx=n_block_first, producer_state=kv_producer_state)
            kv_producer_state.advance()

            for offset in cutlass.range(1, block_count):
                n_block = block_indices[block_count - 1 - offset]
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
        else:
            n_block_first = block_indices[block_count - 1]
            if const_expr(not first_block_preloaded):
                extra_tx = (
                    tma_q_bytes if const_expr(load_q_with_first) and const_expr(use_tma_q) else 0
                )
                pipeline_k.producer_acquire(kv_producer_state, extra_tx_count=extra_tx)

                if const_expr(load_q_with_first and use_tma_q):
                    load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))

                load_K(src_idx=n_block_first, producer_state=kv_producer_state)

            for idx in cutlass.range(block_count - 1, unroll=1):
                n_block_prev = block_indices[block_count - 1 - idx]
                n_block = block_indices[block_count - 2 - idx]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)

    return kv_producer_state


@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor,
    block_count,
    load_V,
    pipeline_v,
    kv_producer_state,
):
    """Load the final V block after overlapped K/V loads."""
    if block_count > 0:
        n_block_last = block_indices[0]
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_last, producer_state=kv_producer_state)
        kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def produce_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    use_tma_q: cutlass.Constexpr,
    tma_q_bytes: cutlass.Constexpr,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Iterate over the mask and full block lists for a single tile.

    The masked (partial) list may leave the last V load pending when intra-warp-group
    overlap is enabled. The first full block must consume that pending V while
    issuing its own K load on the next pipeline stage.

    In the intra-wg-overlap path, the last masked block leaves its V copy in flight
    while we advance the producer state to start the next full K. Either the full list
    overlaps that pending V load, or, if no full blocks exist, we explicitly drain it.

    """

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    if mask_empty:
        # No masked blocks: the full list owns the initial Q+K load.
        kv_producer_state = load_block_list(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            use_tma_q=use_tma_q,
            tma_q_bytes=tma_q_bytes,
            intra_wg_overlap=intra_wg_overlap,
        )

        if const_expr(intra_wg_overlap) and curr_full_block_cnt > 0:
            kv_producer_state = finish_overlap_v_load(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_V,
                pipeline_v,
                kv_producer_state,
            )
    else:
        # Masked blocks present: load Q together with the first masked K so consumers can
        # start immediately. When overlap is disabled this fully drains the list.
        kv_producer_state = load_block_list(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            use_tma_q=use_tma_q,
            tma_q_bytes=tma_q_bytes,
            intra_wg_overlap=intra_wg_overlap,
        )

        if full_empty:
            if const_expr(intra_wg_overlap):
                kv_producer_state = finish_overlap_v_load(
                    curr_mask_block_idx,
                    curr_mask_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
        else:
            if const_expr(intra_wg_overlap):
                # Bridge the masked list to the full list by overlapping the pending masked V
                # with the first full K load.
                n_block_mask_last = curr_mask_block_idx[0]
                n_block_full_first = curr_full_block_idx[curr_full_block_cnt - 1]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_full_first, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_mask_last, producer_state=kv_producer_state_prev)

                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_q_with_first=False,
                    first_block_preloaded=True,
                    kv_producer_state=kv_producer_state,
                    load_Q=load_Q,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    use_tma_q=use_tma_q,
                    tma_q_bytes=tma_q_bytes,
                    intra_wg_overlap=intra_wg_overlap,
                )

                kv_producer_state = finish_overlap_v_load(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
            else:
                # Non-overlap path with both lists: run the full list normally (skipping the Q
                # reload because the masked list already issued it).
                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_q_with_first=False,
                    first_block_preloaded=False,
                    kv_producer_state=kv_producer_state,
                    load_Q=load_Q,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    use_tma_q=use_tma_q,
                    tma_q_bytes=tma_q_bytes,
                    intra_wg_overlap=intra_wg_overlap,
                )

    return kv_producer_state


@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
):
    """Consume the mask and full block lists for a single tile on the consumer side.

    Mirrors `produce_block_sparse_loads` so that the consumer pipeline
    """

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]
    curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
    curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]

    processed_any = curr_mask_block_cnt + curr_full_block_cnt > 0

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=mask_n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None,
                ),
                is_first_n_block=True,
            )
            O_should_accumulate = True
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            if curr_full_block_cnt == 0:
                warp_scheduler_barrier_arrive()

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first_n_block=True,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            kv_consumer_state = process_first_half_block(
                n_block=mask_n_block,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None,
                ),
                score_mod_fn=score_mod_fn,
                is_first_block=True,
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=full_n_block,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    score_mod_fn=score_mod_fn,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                )
                O_should_accumulate = True
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_mask_block_cnt + curr_full_block_cnt > 0:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    m_block,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_count - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=q_stage * m_block + 0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=q_stage * m_block + 1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_count - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.
    """
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            m_block=m_block,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            m_block=m_block,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_q_with_first=False,
                m_block=m_block,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
):
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors
    if const_expr(full_block_cnt is not None):
        return (
            mask_block_cnt[batch_idx, head_idx, m_block]
            + full_block_cnt[batch_idx, head_idx, m_block]
        )
    else:
        return mask_block_cnt[batch_idx, head_idx, m_block]


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.core.ThrMma,
    tOtOs: tuple[cute.Tensor],
    sO: cute.Tensor,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    mbar_corr_epi_full_offset: Int32,
    mbar_corr_epi_empty_offset: Int32,
    softmax_corr_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle the block-sparse case where a tile is fully masked:
    * zero staged results
    * seed stats
    * satisfy the usual barrier protocol so downstream warps continue to make progress.
    """
    LOG2_E = Float32(math.log2(math.e))

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (const_expr(not is_split_kv) or split_idx == 0):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = Float32(1.0)
                else:
                    row_sum_value = row_sum_value + utils.exp2f(
                        sink_val * LOG2_E - row_max_value * softmax_scale_log2
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + m_block_size * 2] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        cute.arch.mbarrier_wait(
            mbar_ptr + mbar_softmax_corr_full_offset + stage,
            softmax_corr_consumer_phase,
        )
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_empty_offset + stage)

        if const_expr(gmem_tiled_copy_O is None):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_corr_epi_empty_offset + stage,
                corr_epi_producer_phase,
            )
        correction_epilogue(
            thr_mma_pv,
            tOtOs[stage],
            tidx,
            stage,
            m_block,
            seqlen.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO,
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_corr_epi_full_offset + stage)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_O_rescaled_offset + stage)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_2_offset + stage)

    softmax_corr_consumer_phase ^= 1
    o_corr_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        softmax_corr_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    mbar_ptr,
    mbar_softmax_corr_full_offset: Int32,
    mbar_softmax_corr_empty_offset: Int32,
    mbar_P_full_O_rescaled_offset: Int32,
    mbar_P_full_2_offset: Int32,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
):
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors

    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]

    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None

    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_full_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_O_rescaled_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_P_full_2_offset + stage_idx)
        cute.arch.mbarrier_arrive(mbar_ptr + mbar_softmax_corr_empty_offset + stage_idx)
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True),  # last block could oob
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False),
                )

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=partial(mask_fn_none, mask_seqlen=True),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=partial(mask_fn_none, mask_seqlen=False),
                )
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=partial(mask_fn_none, mask_seqlen=False),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


# =============================================================================
# Backward-specific block-sparse helpers (SM100)
# =============================================================================
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward.

    Args:
        m_block_max: Maximum m_block index from causal/local masking constraints.
            Computed by block_info.get_m_block_min_max() based on sequence lengths
            and attention mask type. When > 0, caps the result to ensure we don't
            count sparse blocks that fall outside the valid causal/local window.

    Returns min(sparse_block_count * subtile_factor, m_block_max) when m_block_max > 0.
    """
    q_block_cnt, _, full_q_block_cnt, _ = blocksparse_tensors
    total = q_block_cnt[batch_idx, head_idx, n_block]
    if const_expr(full_q_block_cnt is not None):
        total = total + full_q_block_cnt[batch_idx, head_idx, n_block]
    result = total * subtile_factor
    if m_block_max > 0:
        result = cutlass.min(result, m_block_max)
    return result


@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    # Pipeline states (will be returned after advancing)
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    # Pipelines
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    # Load functions
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    # Global tensors for LSE/dPsum
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    # TMA copy bytes for extra_tx_count
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    # Flags for which loads to perform
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    # Subtiling factor and bounds
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """SM100 backward block sparse loading with subtiling.

    Returns updated (producer_state_Q_LSE, producer_state_dO_dPsum).
    First iteration loads K/V alongside Q/dO; subsequent iterations load only Q/dO.
    """
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        loop_count,
    ) = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors, batch_idx, head_idx, n_block, subtile_factor, m_block_max
    )

    for iter_idx in cutlass.range(loop_count, unroll=1):
        m_block, _ = get_m_block_from_iter_bwd(
            iter_idx,
            curr_q_cnt,
            curr_q_idx,
            curr_full_cnt,
            curr_full_idx,
            subtile_factor,
        )

        if iter_idx == 0:
            # First block: load K/V alongside Q/dO
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K)
                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                load_Q(m_block, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(
                    producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V
                )
                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()
        else:
            # Subsequent blocks: just load Q/dO (K/V already loaded)
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                load_Q(m_block, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Args:
        m_block_max: Maximum m_block index from causal/local masking constraints.
            Computed by block_info.get_m_block_min_max() based on sequence lengths
            and attention mask type. When > 0, caps total_count to ensure we don't
            process sparse blocks that fall outside the valid causal/local window.
            This combines block sparsity with causal/local masking.

    Returns (curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    sparse_block_count = curr_q_cnt
    if const_expr(full_cnt is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt

    total_count = sparse_block_count * subtile_factor
    if m_block_max > 0:
        total_count = cutlass.min(total_count, m_block_max)

    return curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
):
    """Derive m_block index and is_full_block flag from iteration index.

    In backward, we iterate in FORWARD order: masked blocks first (low to high),
    then full blocks (low to high). This ensures that when loop_count is capped
    to m_block_max, we skip the high (potentially out-of-bounds) m_blocks at the
    end of iteration rather than in the middle.

    With subtiling (subtile_factor > 1):
    - sparse_iter_idx = iter_idx // subtile_factor (which sparse block)
    - subtile_offset = iter_idx % subtile_factor (which subtile within sparse block)
    - m_block = sparse_m_block * subtile_factor + subtile_offset

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index (after subtiling)
        - is_full_block: True if this is a full block (no mask_mod needed)
          Note: All subtiles within a sparse block share the same is_full_block status
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False

    # Forward order: process low sparse block indices first
    if sparse_iter_idx < curr_q_cnt:
        sparse_m_block = curr_q_idx[sparse_iter_idx]
        is_full_block = False
    else:
        full_iter = sparse_iter_idx - curr_q_cnt
        sparse_m_block = curr_full_idx[full_iter]
        is_full_block = True

    m_block = sparse_m_block * subtile_factor + subtile_offset

    return m_block, is_full_block
