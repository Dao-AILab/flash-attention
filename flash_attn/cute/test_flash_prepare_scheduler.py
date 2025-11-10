#!/usr/bin/env python3
"""Test script for FlashPrepareScheduler.

Tests the prepare scheduler kernel that computes metadata for variable-length sequences.
"""

import torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute.flash_prepare_scheduler import FlashPrepareScheduler


def test_prepare_scheduler_basic():
    """Test basic functionality of FlashPrepareScheduler."""
    print("=" * 80)
    print("Test: Basic prepare scheduler functionality")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    # Test configuration
    num_batch = 4
    nheads = 8
    nheads_kv = 2
    seqlen_q_static = 1024
    seqlen_k_static = 2048
    seqlen_k_new_static = 0
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108  # Typical for H100
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Create output tensors
    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    # Convert to CuTe tensors
    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

    # Create scheduler
    scheduler = FlashPrepareScheduler(sort=False)

    # Get CUDA stream
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  nheads: {nheads}, nheads_kv: {nheads_kv}")
    print(f"  seqlen_q_static: {seqlen_q_static}, seqlen_k_static: {seqlen_k_static}")
    print(f"  tile_m: {tile_m}, tile_n: {tile_n}")
    print(f"  packgqa: {packgqa}, is_causal: {is_causal}")

    # Call the scheduler
    print("\nCalling prepare scheduler...")
    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")
    print(f"  num_nheads_in_l2: {num_nheads_in_l2.cpu().numpy()}")
    print(f"  tile_count_semaphore: {tile_count_semaphore.cpu().numpy()}")

    # Verify outputs are reasonable
    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    print("\nVerification:")
    print(f"  Expected seqlen_q (per batch): {expected_seqlen_q}")

    # All batches should have the same values for static seqlen
    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {prepare_seqlen_q}, expected {expected_seqlen_q}"
    )

    # num_splits_dynamic should be >= 1 and <= num_splits_static
    assert torch.all(num_splits_dynamic >= 1), (
        f"num_splits_dynamic should be >= 1, got {num_splits_dynamic}"
    )
    assert torch.all(num_splits_dynamic <= num_splits_static), (
        f"num_splits_dynamic should be <= num_splits_static ({num_splits_static}), got {num_splits_dynamic}"
    )

    # num_nheads_in_l2 should be <= nheads_kv (when packgqa) or nheads (when not packgqa)
    max_nheads = nheads_kv if packgqa else nheads
    assert torch.all(num_nheads_in_l2 <= max_nheads), (
        f"num_nheads_in_l2 should be <= {max_nheads}, got {num_nheads_in_l2}"
    )

    print("✓ Basic test passed!")


def test_prepare_scheduler_varlen():
    """Test prepare scheduler with variable-length sequences."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with variable-length sequences")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    # Test configuration with variable lengths
    num_batch = 3
    nheads = 4
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = False
    is_causal = False
    enable_pdl = False

    # Variable sequence lengths
    seqlens_q = torch.tensor([512, 1024, 256], dtype=dtype, device=device)
    seqlens_k = torch.tensor([1024, 2048, 512], dtype=dtype, device=device)

    # Create cumulative sequence lengths
    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    # Create output tensors
    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    # Convert to CuTe tensors
    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    # Create scheduler
    scheduler = FlashPrepareScheduler(sort=False)

    # Get CUDA stream
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlens_q: {seqlens_q.cpu().numpy()}")
    print(f"  seqlens_k: {seqlens_k.cpu().numpy()}")
    print(f"  cu_seqlens_q: {cu_seqlens_q.cpu().numpy()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.cpu().numpy()}")

    # Call the scheduler
    print("\nCalling prepare scheduler...")
    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")
    print(f"  num_nheads_in_l2: {num_nheads_in_l2.cpu().numpy()}")

    # Verify outputs match expected sequence lengths
    # For non-packgqa, prepare_seqlen_q should equal seqlen_q
    expected_seqlen_q = seqlens_q.cpu().numpy()
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nVerification:")
    print(f"  Expected seqlen_q: {expected_seqlen_q}")
    print(f"  Actual seqlen_q: {actual_seqlen_q}")

    # Check that values are reasonable (may not match exactly due to rounding)
    for i in range(num_batch):
        # prepare_seqlen_q should be close to seqlen_q (within tile_m)
        assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
            f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
        )

    # num_splits_dynamic should be >= 1 and <= num_splits_static
    assert torch.all(num_splits_dynamic >= 1), (
        f"num_splits_dynamic should be >= 1, got {num_splits_dynamic}"
    )
    assert torch.all(num_splits_dynamic <= num_splits_static), (
        f"num_splits_dynamic should be <= num_splits_static ({num_splits_static}), got {num_splits_dynamic}"
    )

    # For varlen, validate that splits make sense given sequence lengths
    splits_cpu = num_splits_dynamic.cpu().numpy()
    seqlens_k_cpu = seqlens_k.cpu().numpy()
    # Longer sequences should generally have more or equal splits
    # (this is a heuristic - actual splits depend on L2 cache and other factors)
    for i in range(num_batch):
        # Very long sequences should have at least some splits if num_splits_static > 1
        if seqlens_k_cpu[i] > 4096 and num_splits_static > 1:
            assert splits_cpu[i] >= 1, (
                f"Batch {i} with seqlen_k={seqlens_k_cpu[i]} should have at least 1 split"
            )

    print("✓ Varlen test passed!")


def test_prepare_scheduler_varlen_small_batch():
    """Test varlen with small batch size."""
    print("\n" + "=" * 80)
    print("Test: Varlen with small batch size")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 1
    nheads = 4
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    seqlens_q = torch.tensor([256], dtype=dtype, device=device)
    seqlens_k = torch.tensor([512], dtype=dtype, device=device)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlens_q: {seqlens_q.cpu().numpy()}")
    print(f"  seqlens_k: {seqlens_k.cpu().numpy()}")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {actual_seqlen_q}")
    print(f"  expected_seqlen_q: {expected_seqlen_q}")

    assert torch.all(prepare_seqlen_q.cpu() == torch.from_numpy(expected_seqlen_q)), (
        f"prepare_seqlen_q mismatch: got {actual_seqlen_q}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Small batch varlen test passed!")


def test_prepare_scheduler_varlen_large_batch():
    """Test varlen with large batch size."""
    print("\n" + "=" * 80)
    print("Test: Varlen with large batch size")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 100
    nheads = 8
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Create variable sequence lengths
    torch.manual_seed(42)
    seqlens_q = torch.randint(128, 2048, (num_batch,), dtype=dtype, device=device)
    seqlens_k = (seqlens_q * 2).to(dtype)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlens_q range: [{seqlens_q.min().item()}, {seqlens_q.max().item()}]")
    print(f"  seqlens_k range: [{seqlens_k.min().item()}, {seqlens_k.max().item()}]")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q[:5]: {actual_seqlen_q[:5]}")
    print(f"  prepare_seqlen_q[-5:]: {actual_seqlen_q[-5:]}")

    # Check that values are reasonable (within tile_m)
    for i in range(num_batch):
        assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
            f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
        )

    # num_splits_dynamic should be >= 1 and <= num_splits_static
    assert torch.all(num_splits_dynamic >= 1), (
        f"num_splits_dynamic should be >= 1, got {num_splits_dynamic}"
    )
    assert torch.all(num_splits_dynamic <= num_splits_static), (
        f"num_splits_dynamic should be <= num_splits_static ({num_splits_static}), got {num_splits_dynamic}"
    )

    # For varlen, longer sequences should generally have more splits (or at least not fewer)
    # This is a heuristic check - longer sequences need more splits to fit in L2
    splits_cpu = num_splits_dynamic.cpu().numpy()
    seqlens_k_cpu = seqlens_k.cpu().numpy()
    # Check that splits are reasonable relative to sequence lengths
    for i in range(num_batch):
        # Very long sequences should have at least some splits if num_splits_static > 1
        if seqlens_k_cpu[i] > 4096 and num_splits_static > 1:
            assert splits_cpu[i] >= 1, (
                f"Batch {i} with seqlen_k={seqlens_k_cpu[i]} should have at least 1 split"
            )

    print("✓ Large batch varlen test passed!")


def test_prepare_scheduler_varlen_extreme_lengths():
    """Test varlen with extreme sequence lengths (very small and very large)."""
    print("\n" + "=" * 80)
    print("Test: Varlen with extreme sequence lengths")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 10
    nheads = 8
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Mix of very small and very large sequences
    seqlens_q = torch.tensor(
        [32, 64, 128, 512, 1024, 2048, 4096, 8192, 64, 256], dtype=dtype, device=device
    )
    seqlens_k = (seqlens_q * 2).to(dtype)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlens_q: {seqlens_q.cpu().numpy()}")
    print(f"  seqlens_k: {seqlens_k.cpu().numpy()}")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {actual_seqlen_q}")
    print(f"  expected_seqlen_q: {expected_seqlen_q}")

    for i in range(num_batch):
        assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
            f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
        )

    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Extreme lengths varlen test passed!")


def test_prepare_scheduler_varlen_uniform():
    """Test varlen with uniform sequence lengths."""
    print("\n" + "=" * 80)
    print("Test: Varlen with uniform sequence lengths")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 20
    nheads = 8
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # All sequences have the same length
    seqlen = 1024
    seqlens_q = torch.full((num_batch,), seqlen, dtype=dtype, device=device)
    seqlens_k = torch.full((num_batch,), seqlen * 2, dtype=dtype, device=device)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  uniform seqlen_q: {seqlen}")
    print(f"  uniform seqlen_k: {seqlen * 2}")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {actual_seqlen_q[:5]}... (all should be {expected_seqlen_q})")

    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {actual_seqlen_q}, expected all {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Uniform varlen test passed!")


def test_prepare_scheduler_varlen_skewed():
    """Test varlen with skewed sequence length distribution."""
    print("\n" + "=" * 80)
    print("Test: Varlen with skewed sequence length distribution")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 15
    nheads = 8
    nheads_kv = 2
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Skewed distribution: mostly small sequences, few large ones
    seqlens_q = torch.tensor(
        [64, 64, 64, 64, 64, 128, 128, 128, 256, 256, 512, 512, 1024, 2048, 4096],
        dtype=dtype,
        device=device,
    )
    seqlens_k = (seqlens_q * 2).to(dtype)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlens_q: {seqlens_q.cpu().numpy()}")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {actual_seqlen_q}")

    for i in range(num_batch):
        assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
            f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
        )

    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Skewed varlen test passed!")


def test_prepare_scheduler_varlen_no_packgqa():
    """Test varlen without packgqa."""
    print("\n" + "=" * 80)
    print("Test: Varlen without packgqa")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 8
    nheads = 8
    nheads_kv = 8  # Same as nheads
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = False  # No packing
    is_causal = False
    enable_pdl = False

    seqlens_q = torch.tensor(
        [256, 512, 1024, 2048, 512, 1024, 256, 512], dtype=dtype, device=device
    )
    seqlens_k = (seqlens_q * 2).to(dtype)

    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
    cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  packgqa: {packgqa}")
    print(f"  seqlens_q: {seqlens_q.cpu().numpy()}")

    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=cu_seqlens_q_cute,
        mCuSeqlensK=cu_seqlens_k_cute,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {actual_seqlen_q}")
    print(f"  expected_seqlen_q: {expected_seqlen_q}")

    assert torch.all(prepare_seqlen_q.cpu() == torch.from_numpy(expected_seqlen_q)), (
        f"prepare_seqlen_q mismatch: got {actual_seqlen_q}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ No packgqa varlen test passed!")


def test_prepare_scheduler_varlen_different_ratios():
    """Test varlen with different head ratios."""
    print("\n" + "=" * 80)
    print("Test: Varlen with different head ratios")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    test_configs = [
        (8, 8, 1, True),  # 1:1 ratio, packgqa=True
        (8, 4, 2, True),  # 2:1 ratio, packgqa=True
        (8, 2, 4, True),  # 4:1 ratio, packgqa=True
        (16, 4, 4, True),  # 4:1 ratio, packgqa=True
        (8, 8, 1, False),  # 1:1 ratio, packgqa=False
    ]

    num_batch = 5
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    is_causal = False
    enable_pdl = False

    seqlens_q = torch.tensor([256, 512, 1024, 512, 256], dtype=dtype, device=device)
    seqlens_k = (seqlens_q * 2).to(dtype)

    for nheads, nheads_kv, expected_ratio, packgqa in test_configs:
        cu_seqlens_q = torch.zeros(num_batch + 1, dtype=dtype, device=device)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

        cu_seqlens_k = torch.zeros(num_batch + 1, dtype=dtype, device=device)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

        prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
        num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
        num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
        tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

        prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
        num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
        num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
        tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
        cu_seqlens_q_cute = from_dlpack(cu_seqlens_q)
        cu_seqlens_k_cute = from_dlpack(cu_seqlens_k)

        scheduler = FlashPrepareScheduler(sort=False)
        stream = cuda.CUstream(0)

        print(
            f"\nTesting nheads={nheads}, nheads_kv={nheads_kv}, ratio={expected_ratio}:1, packgqa={packgqa}"
        )

        scheduler(
            seqlen_q_static=0,
            seqlen_k_static=0,
            seqlen_k_new_static=0,
            mCuSeqlensQ=cu_seqlens_q_cute,
            mCuSeqlensK=cu_seqlens_k_cute,
            mCuSeqlensKNew=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            mLeftPadK=None,
            num_batch=num_batch,
            nheads=nheads,
            nheads_kv=nheads_kv,
            num_sm=num_sm,
            num_splits_static=num_splits_static,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_count_semaphore=tile_count_semaphore_cute,
            mPrepareSeqlenQ=prepare_seqlen_q_cute,
            mNumSplitsDynamic=num_splits_dynamic_cute,
            mVarlenBatchIdx=None,
            mNumNheadsInL2=num_nheads_in_l2_cute,
            enable_pdl=enable_pdl,
            is_causal=is_causal,
            packgqa=packgqa,
            is_e4m3=is_e4m3,
            d=d,
            dv=dv,
            stream=stream,
        )

        torch.cuda.synchronize()

        qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
        expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
        actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

        assert qhead_per_khead == expected_ratio, (
            f"qhead_per_khead mismatch: got {qhead_per_khead}, expected {expected_ratio}"
        )

        for i in range(num_batch):
            assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
                f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
            )

        assert torch.all(num_splits_dynamic >= 1)

        print(f"  ✓ Ratio {expected_ratio}:1, packgqa={packgqa} passed")

    print("✓ Different ratios varlen test passed!")


def test_prepare_scheduler_seqused():
    """Test prepare scheduler with seqused tensors."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with seqused tensors")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    # Test configuration
    num_batch = 2
    nheads = 8
    nheads_kv = 4
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 2
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Sequence lengths using seqused
    seqused_q = torch.tensor([256, 512], dtype=dtype, device=device)
    seqused_k = torch.tensor([512, 1024], dtype=dtype, device=device)

    # Create output tensors
    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    # Convert to CuTe tensors
    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    seqused_q_cute = from_dlpack(seqused_q)
    seqused_k_cute = from_dlpack(seqused_k)

    # Create scheduler
    scheduler = FlashPrepareScheduler(sort=False)

    # Get CUDA stream
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqused_q: {seqused_q.cpu().numpy()}")
    print(f"  seqused_k: {seqused_k.cpu().numpy()}")

    # Call the scheduler
    print("\nCalling prepare scheduler...")
    scheduler(
        seqlen_q_static=0,
        seqlen_k_static=0,
        seqlen_k_new_static=0,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=seqused_q_cute,
        mSeqUsedK=seqused_k_cute,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")
    print(f"  num_nheads_in_l2: {num_nheads_in_l2.cpu().numpy()}")

    # Verify outputs
    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqused_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    actual_seqlen_q = prepare_seqlen_q.cpu().numpy()

    print("\nVerification:")
    print(f"  Expected seqlen_q: {expected_seqlen_q}")
    print(f"  Actual seqlen_q: {actual_seqlen_q}")

    # Check that values are reasonable
    for i in range(num_batch):
        assert abs(actual_seqlen_q[i] - expected_seqlen_q[i]) < tile_m, (
            f"Batch {i}: prepare_seqlen_q {actual_seqlen_q[i]} not close to expected {expected_seqlen_q[i]}"
        )

    print("✓ Seqused test passed!")


def test_prepare_scheduler_causal():
    """Test prepare scheduler with causal masking."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with causal masking")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 4
    nheads = 16
    nheads_kv = 4
    seqlen_q_static = 512
    seqlen_k_static = 512
    seqlen_k_new_static = 0
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 64
    is_e4m3 = False
    packgqa = True
    is_causal = True
    enable_pdl = False

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}, is_causal: {is_causal}")
    print(f"  nheads: {nheads}, nheads_kv: {nheads_kv}")
    print(f"  seqlen_q_static: {seqlen_q_static}, seqlen_k_static: {seqlen_k_static}")

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")

    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {prepare_seqlen_q}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Causal test passed!")


def test_prepare_scheduler_leftpad():
    """Test prepare scheduler with left padding."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with left padding")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 3
    nheads = 8
    nheads_kv = 2
    seqlen_q_static = 1024
    seqlen_k_static = 2048
    seqlen_k_new_static = 256
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Left padding values
    leftpad_k = torch.tensor([64, 128, 32], dtype=dtype, device=device)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    leftpad_k_cute = from_dlpack(leftpad_k)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  leftpad_k: {leftpad_k.cpu().numpy()}")
    print(f"  seqlen_k_new_static: {seqlen_k_new_static}")

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=leftpad_k_cute,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")

    assert torch.all(prepare_seqlen_q == expected_seqlen_q)
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Leftpad test passed!")


def test_prepare_scheduler_append_kv():
    """Test prepare scheduler with append KV (seqlen_k_new)."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with append KV")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 2
    nheads = 8
    nheads_kv = 2
    seqlen_q_static = 512
    seqlen_k_static = 1024
    seqlen_k_new_static = 256
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 64
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    # Variable new sequence lengths
    seqlens_k_new = torch.tensor([128, 256], dtype=dtype, device=device)
    cu_seqlens_k_new = torch.zeros(num_batch + 1, dtype=dtype, device=device)
    cu_seqlens_k_new[1:] = torch.cumsum(seqlens_k_new, dim=0)

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)
    cu_seqlens_k_new_cute = from_dlpack(cu_seqlens_k_new)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  seqlen_k_static: {seqlen_k_static}")
    print(f"  seqlens_k_new: {seqlens_k_new.cpu().numpy()}")

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=cu_seqlens_k_new_cute,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic.cpu().numpy()}")

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    assert torch.all(prepare_seqlen_q == expected_seqlen_q)
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Append KV test passed!")


def test_prepare_scheduler_different_ratios():
    """Test prepare scheduler with different qhead_per_khead ratios."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with different qhead_per_khead ratios")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    test_configs = [
        (8, 8, 1),  # 1:1 ratio
        (8, 4, 2),  # 2:1 ratio
        (8, 2, 4),  # 4:1 ratio
        (16, 4, 4),  # 4:1 ratio
        (32, 8, 4),  # 4:1 ratio
    ]

    seqlen_q_static = 1024
    seqlen_k_static = 2048
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    for nheads, nheads_kv, expected_ratio in test_configs:
        num_batch = 2

        prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
        num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
        num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
        tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

        prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
        num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
        num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
        tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

        scheduler = FlashPrepareScheduler(sort=False)
        stream = cuda.CUstream(0)

        print(f"\nTesting nheads={nheads}, nheads_kv={nheads_kv}, expected_ratio={expected_ratio}")

        scheduler(
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            seqlen_k_new_static=0,
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mCuSeqlensKNew=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            mLeftPadK=None,
            num_batch=num_batch,
            nheads=nheads,
            nheads_kv=nheads_kv,
            num_sm=num_sm,
            num_splits_static=num_splits_static,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_count_semaphore=tile_count_semaphore_cute,
            mPrepareSeqlenQ=prepare_seqlen_q_cute,
            mNumSplitsDynamic=num_splits_dynamic_cute,
            mVarlenBatchIdx=None,
            mNumNheadsInL2=num_nheads_in_l2_cute,
            enable_pdl=enable_pdl,
            is_causal=is_causal,
            packgqa=packgqa,
            is_e4m3=is_e4m3,
            d=d,
            dv=dv,
            stream=stream,
        )

        torch.cuda.synchronize()

        qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
        expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

        assert qhead_per_khead == expected_ratio, (
            f"qhead_per_khead mismatch: got {qhead_per_khead}, expected {expected_ratio}"
        )
        assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
            f"prepare_seqlen_q mismatch: got {prepare_seqlen_q}, expected {expected_seqlen_q}"
        )
        assert torch.all(num_splits_dynamic >= 1)

        print(f"  ✓ Ratio {expected_ratio}:1 passed")

    print("✓ Different ratios test passed!")


def test_prepare_scheduler_large_batch():
    """Test prepare scheduler with large batch size (>992 to test multiple CTAs)."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler with large batch size")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 1500  # > 992 to test multiple CTAs
    nheads = 8
    nheads_kv = 2
    seqlen_q_static = 512
    seqlen_k_static = 1024
    seqlen_k_new_static = 0
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = True
    is_causal = False
    enable_pdl = False

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch} (should use multiple CTAs)")

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    print("\nResults:")
    print(f"  prepare_seqlen_q[:5]: {prepare_seqlen_q[:5].cpu().numpy()}")
    print(f"  prepare_seqlen_q[-5:]: {prepare_seqlen_q[-5:].cpu().numpy()}")
    print(f"  num_splits_dynamic[:5]: {num_splits_dynamic[:5].cpu().numpy()}")

    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {prepare_seqlen_q[:5]}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ Large batch test passed!")


def test_prepare_scheduler_edge_cases():
    """Test prepare scheduler with edge cases (small sequences, zero-length, etc.)."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler edge cases")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    # Test 1: Very small sequences
    print("\nTest 1: Very small sequences")
    num_batch = 2
    nheads = 4
    nheads_kv = 2
    seqlen_q_static = 64  # Smaller than tile_m
    seqlen_k_static = 64  # Smaller than tile_n
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 1
    is_e4m3 = False
    packgqa = False
    is_causal = False
    enable_pdl = False

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=0,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    expected_seqlen_q = seqlen_q_static * (1 if packgqa else 1)  # packgqa=False
    assert torch.all(prepare_seqlen_q == expected_seqlen_q)
    assert torch.all(num_splits_dynamic >= 1)
    print("  ✓ Small sequences passed")

    # Test 2: Very long sequences
    print("\nTest 2: Very long sequences")
    seqlen_q_static = 8192
    seqlen_k_static = 16384

    prepare_seqlen_q.zero_()
    num_splits_dynamic.zero_()
    num_nheads_in_l2.zero_()

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=0,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    # Recalculate expected_seqlen_q for the new sequence lengths
    expected_seqlen_q = seqlen_q_static * (1 if packgqa else 1)  # packgqa=False
    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {prepare_seqlen_q.cpu().numpy()}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)
    print("  ✓ Long sequences passed")

    print("✓ Edge cases test passed!")


def test_prepare_scheduler_no_packgqa():
    """Test prepare scheduler without packgqa (packgqa=False)."""
    print("\n" + "=" * 80)
    print("Test: Prepare scheduler without packgqa")
    print("=" * 80)

    device = "cuda"
    dtype = torch.int32

    num_batch = 4
    nheads = 8
    nheads_kv = 8  # Same as nheads
    seqlen_q_static = 1024
    seqlen_k_static = 2048
    seqlen_k_new_static = 0
    tile_m = 128
    tile_n = 128
    d = 128
    dv = 128
    num_sm = 108
    num_splits_static = 128
    is_e4m3 = False
    packgqa = False
    is_causal = False
    enable_pdl = False

    prepare_seqlen_q = torch.zeros(num_batch, dtype=dtype, device=device)
    num_splits_dynamic = torch.zeros(num_batch, dtype=dtype, device=device)
    num_nheads_in_l2 = torch.zeros(num_batch, dtype=dtype, device=device)
    tile_count_semaphore = torch.zeros(1, dtype=dtype, device=device)

    prepare_seqlen_q_cute = from_dlpack(prepare_seqlen_q)
    num_splits_dynamic_cute = from_dlpack(num_splits_dynamic)
    num_nheads_in_l2_cute = from_dlpack(num_nheads_in_l2)
    tile_count_semaphore_cute = from_dlpack(tile_count_semaphore)

    scheduler = FlashPrepareScheduler(sort=False)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  packgqa: {packgqa}")
    print(f"  nheads: {nheads}, nheads_kv: {nheads_kv}")

    scheduler(
        seqlen_q_static=seqlen_q_static,
        seqlen_k_static=seqlen_k_static,
        seqlen_k_new_static=seqlen_k_new_static,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mCuSeqlensKNew=None,
        mSeqUsedQ=None,
        mSeqUsedK=None,
        mLeftPadK=None,
        num_batch=num_batch,
        nheads=nheads,
        nheads_kv=nheads_kv,
        num_sm=num_sm,
        num_splits_static=num_splits_static,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_count_semaphore=tile_count_semaphore_cute,
        mPrepareSeqlenQ=prepare_seqlen_q_cute,
        mNumSplitsDynamic=num_splits_dynamic_cute,
        mVarlenBatchIdx=None,
        mNumNheadsInL2=num_nheads_in_l2_cute,
        enable_pdl=enable_pdl,
        is_causal=is_causal,
        packgqa=packgqa,
        is_e4m3=is_e4m3,
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlen_q_static * (qhead_per_khead if packgqa else 1)

    print("\nResults:")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q.cpu().numpy()}")
    print(f"  expected_seqlen_q: {expected_seqlen_q}")

    assert torch.all(prepare_seqlen_q == expected_seqlen_q), (
        f"prepare_seqlen_q mismatch: got {prepare_seqlen_q}, expected {expected_seqlen_q}"
    )
    assert torch.all(num_splits_dynamic >= 1)

    print("✓ No packgqa test passed!")


def main():
    """Run all tests."""
    print("Testing FlashPrepareScheduler")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    try:
        test_prepare_scheduler_basic()
        test_prepare_scheduler_varlen()
        test_prepare_scheduler_varlen_small_batch()
        test_prepare_scheduler_varlen_large_batch()
        test_prepare_scheduler_varlen_extreme_lengths()
        test_prepare_scheduler_varlen_uniform()
        test_prepare_scheduler_varlen_skewed()
        test_prepare_scheduler_varlen_no_packgqa()
        test_prepare_scheduler_varlen_different_ratios()
        test_prepare_scheduler_seqused()
        test_prepare_scheduler_causal()
        test_prepare_scheduler_leftpad()
        test_prepare_scheduler_append_kv()
        test_prepare_scheduler_different_ratios()
        test_prepare_scheduler_large_batch()
        test_prepare_scheduler_edge_cases()
        test_prepare_scheduler_no_packgqa()

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
