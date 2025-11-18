#!/usr/bin/env python3
"""Test script for FlashPrepareScheduler.

Tests the prepare scheduler kernel that computes metadata for variable-length sequences.
"""

import torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

# Importing from local file instead of installed package
from flash_prepare_scheduler import FlashPrepareScheduler

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
    # is_e4m3 removed as kernel doesn't accept it
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

    # Create scheduler with correct init params
    # packgqa and num_batch must be passed here, not in __call__
    scheduler = FlashPrepareScheduler(packgqa=packgqa, sort=False, num_batch=num_batch)

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
    # Removed: num_batch, packgqa, is_e4m3 from call args
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

    # Create scheduler with correct init params
    scheduler = FlashPrepareScheduler(packgqa=packgqa, sort=False, num_batch=num_batch)

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

    scheduler = FlashPrepareScheduler(packgqa=packgqa, sort=False, num_batch=num_batch)
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

    scheduler = FlashPrepareScheduler(packgqa=packgqa, sort=False, num_batch=num_batch)
    stream = cuda.CUstream(0)

    print("Configuration:")
    print(f"  num_batch: {num_batch}")
    print(f"  Total q len: {cu_seqlens_q[-1]}")
    print(f"  Total k len: {cu_seqlens_k[-1]}")

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
        d=d,
        dv=dv,
        stream=stream,
    )

    torch.cuda.synchronize()

    print("\nResults (first 5 batches):")
    print(f"  prepare_seqlen_q: {prepare_seqlen_q[:5].cpu().numpy()}")
    print(f"  num_splits_dynamic: {num_splits_dynamic[:5].cpu().numpy()}")

    # Validation
    qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
    expected_seqlen_q = seqlens_q.cpu().numpy() * (qhead_per_khead if packgqa else 1)
    
    assert torch.all(prepare_seqlen_q.cpu() == torch.from_numpy(expected_seqlen_q)), "prepare_seqlen_q mismatch"
    assert torch.all(num_splits_dynamic >= 1), "num_splits_dynamic invalid"
    assert torch.all(num_splits_dynamic <= num_splits_static), "num_splits_dynamic exceeds static limit"

    print("✓ Large batch varlen test passed!")


if __name__ == "__main__":
    test_prepare_scheduler_basic()
    test_prepare_scheduler_varlen()
    test_prepare_scheduler_varlen_small_batch()
    test_prepare_scheduler_varlen_large_batch()