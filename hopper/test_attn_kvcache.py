import pytest
import torch
import flash_attn
import flash_attn_interface
import itertools
import time

@pytest.mark.parametrize("nheads_q", [48//2, 64//8, 16, 48, 64//4])
@pytest.mark.parametrize("nheads_kv", [4//2, 8//8, 1, 4, 8//4])
@pytest.mark.parametrize("num_requests", [1, 4, 16])
@pytest.mark.parametrize("query_seqlen", [1, 16])
@pytest.mark.parametrize("context_seqlen", [16384, 65536])
@pytest.mark.parametrize("headdim", [64, 128, 256])
def test_flash_attn_kvcach_output(nheads_q, nheads_kv, num_requests, query_seqlen, context_seqlen, headdim):
    device = "cuda"
    num_caches = 16
    cache_seqlen = 65536

    k_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=torch.bfloat16
    )
    # print(f"***{model_name}***")
    q = torch.randn((num_requests, query_seqlen, nheads_q, headdim), device="cuda", dtype=torch.bfloat16)
    cache_idxs = torch.randperm(num_caches, dtype=torch.int32, device="cuda")[:num_requests]
    cache_seqlens = torch.tensor([context_seqlen] * num_requests, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    out_fa2, lse_fa2 = flash_attn_interface.flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_idxs,
                    causal=True,
                    num_splits=1,
                    return_softmax_lse=True,
                )
    for i in range(1, 128):
                out_fa3, lse_fa3 = flash_attn_interface.flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_idxs,
                    causal=True,
                    num_splits=i,
                    return_softmax_lse=True,
                )

                out_fa3_gqa, lse_fa3_gqa = flash_attn_interface.flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_idxs,
                    causal=True,
                    num_splits=i,
                    return_softmax_lse=True,
                    gqa_decoding=True
                )

                torch.cuda.synchronize()
                print ('output-max-diff', i, context_seqlen, (out_fa2 - out_fa3).abs().max().item())
                print ('output-mean-diff',i, context_seqlen, (out_fa2 - out_fa3).abs().mean().item())
                print ('output-max-diff gqa', i, context_seqlen, (out_fa2 - out_fa3_gqa).abs().max().item())
                print ('output-mean-diff gqa',i, context_seqlen, (out_fa2 - out_fa3_gqa).abs().mean().item())
                print ('lse-max-diff',i, context_seqlen, (lse_fa2 - lse_fa3).abs().max().item())
                print ('lse-mean-diff',i,  context_seqlen, (lse_fa2 - lse_fa3).abs().mean().item())
                print ('lse-max-diff gqa',i, context_seqlen, (lse_fa2 - lse_fa3_gqa).abs().max().item())
                print ('lse-mean-diff gqa',i, context_seqlen, (lse_fa2 - lse_fa3_gqa).abs().mean().item())

                assert ((out_fa2 - out_fa3).abs().max().item() <= 1e-3)
                assert ((out_fa2 - out_fa3).abs().mean().item() <= 1e-4)
                assert ((out_fa2 - out_fa3_gqa).abs().max().item() <= 1e-3)
                assert ((out_fa2 - out_fa3_gqa).abs().mean().item() <= 1e-4)
                assert ((lse_fa2 - lse_fa3).abs().max().item() <= 1e-3)
                assert ((lse_fa2 - lse_fa3).abs().mean().item() <= 1e-4)
                assert ((lse_fa2 - lse_fa3_gqa).abs().max().item() <= 1e-3)
                assert ((lse_fa2 - lse_fa3_gqa).abs().mean().item() <= 1e-4)

if __name__ == "__main__":
    main()
