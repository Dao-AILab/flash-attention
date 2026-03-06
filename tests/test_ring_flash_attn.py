"""
Correctness test for ring_flash_attn_func.

Run with torchrun:
    torchrun --nproc_per_node=2 tests/test_ring_flash_attn.py
"""
import os
import torch
import torch.distributed as dist

from flash_attn import flash_attn_func, ring_flash_attn_func
from flash_attn.utils.distributed import split_sequence_for_ring


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(0)
    B, S, H, D = 2, 512, 8, 64
    assert S % world_size == 0

    # Build the full tensors on rank 0 and broadcast so all ranks agree
    q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    for t in (q, k, v):
        dist.broadcast(t, src=0)

    # Reference: single-GPU flash attention on full sequence (rank 0 only)
    ref_out = flash_attn_func(q, k, v, causal=False)

    # Ring attention: each rank computes on its local slice
    q_local = split_sequence_for_ring(q, dist.group.WORLD)
    k_local = split_sequence_for_ring(k, dist.group.WORLD)
    v_local = split_sequence_for_ring(v, dist.group.WORLD)

    ring_out = ring_flash_attn_func(q_local, k_local, v_local, causal=False)

    # The local ring output should match the corresponding slice of the reference
    chunk = S // world_size
    ref_local = ref_out[:, rank * chunk : (rank + 1) * chunk]

    max_diff = (ring_out - ref_local).abs().max().item()
    atol = 1e-2  # bfloat16 accumulation tolerance
    status = "PASS" if max_diff < atol else "FAIL"
    print(f"[rank {rank}] max_diff={max_diff:.6f}  {status}", flush=True)
    assert max_diff < atol, f"Ring attention output diverges: max_diff={max_diff}"

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
