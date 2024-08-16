import torch
#from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
import flash_attn_interface as fa3
import flash_attn as fa2


def main():
    # *SAMPLE CONFIG*
    # Model arch params:
    nheads_q = 64
    nheads_kv = 8
    headdim = 128
    dtype = torch.bfloat16

    # Cache settings:
    num_caches = 8
    cache_seqlen = 1024 * 16

    # Batching settings
    ntokens = 1024
    max_queries_per_batch = 4
    small_request_ntokens = 16

    # Input settings
    query_seqlens = [900, 12, 3]
    num_queries = len(query_seqlens)
    # Need to add empty queries to fill out `max_queries_per_batch`
    num_padding_queries = max_queries_per_batch - num_queries
    context_seqlens = [4096, 5120*2, 6145*2]

    # Validation
    assert sum(query_seqlens) <= ntokens
    assert all(s < small_request_ntokens for s in query_seqlens[1:])
    assert num_queries <= max_queries_per_batch
    assert all(s < cache_seqlen for s in context_seqlens)

    # Allocate some tensors
    k_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
    )
    v_cache = torch.randn(
        (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
    )

    q_buf_large = torch.randn(
        (1, ntokens, nheads_q, headdim), device="cuda", dtype=dtype
    )
    cache_seqlen_large = torch.tensor(
        [context_seqlens[0]], dtype=torch.int32, device="cuda"
    )
    cache_idx_large = torch.tensor([1], dtype=torch.int32, device="cuda")

    q_buf_small = torch.randn(
        (max_queries_per_batch - 1, small_request_ntokens, nheads_q, headdim),
        device="cuda",
        dtype=dtype,
    )
    cache_seqlens_small = torch.tensor(
        context_seqlens[1:] + [0] * num_padding_queries, dtype=torch.int32, device="cuda"
    )
    cache_idxs_small = torch.randperm(num_caches, dtype=torch.int32, device="cuda")[
        : max_queries_per_batch - 1
    ]

    # Call flash attn
    # First for the single full-sized query
    out0 = fa3.flash_attn_with_kvcache(
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=True,
    )

    # Second for n-1 small queries
    out1 = fa3.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=True,
    )

      # Call flash attn
    # First for the single full-sized query
    out2 = fa2.flash_attn_with_kvcache(
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=True,
    )

    # Second for n-1 small queries
    out3 = fa2.flash_attn_with_kvcache(
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=True,
    )

    print ((out0 - out2).abs().max().item());
    print ((out1 - out3).abs().max().item());

if __name__ == "__main__":
    main()
