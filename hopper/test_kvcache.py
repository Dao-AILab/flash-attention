import torch
#from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
import flash_attn_interface as fa3
import flash_attn as fa2
import torch.utils.benchmark as benchmark
import time

import argparse
import math

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--causal', action='store_true')
parser.add_argument('--splits', type=int, default=1)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--validate', action='store_true')
parser.add_argument('--gqa', action='store_true')

args = parser.parse_args()

def benchmark_fa_kv_old(fn, repeats=10, desc='', verbose=True, **kwinputs):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(**kwinputs)',
            globals={'fn': fn, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(desc, m)
    return t, m

def benchmark_fa_kv(fn, repeats=10, *args, **kwargs):
    # warmup
    for _ in range(5):
        fn(*args, **kwargs)
    niters = repeats
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(niters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / niters

def main():
    # *SAMPLE CONFIG*
    # Model arch params:
    nheads_q = 64
    nheads_kv = 8
    headdim = 128
    #dtype = torch.bfloat16
    dtype = torch.float16

    # Cache settings:
    num_caches = 8
    cache_seqlen = 1024 * 16

    # Batching settings
    ntokens = 1024
    max_queries_per_batch = 4
    small_request_ntokens = 16

    # Input settings
    query_seqlens = [900, 12, 1]
    num_queries = len(query_seqlens)
    # Need to add empty queries to fill out `max_queries_per_batch`
    num_padding_queries = max_queries_per_batch - num_queries
    context_seqlens = [4096, 5120*2, 6145*2]
    #context_seqlens = [4096, 5120*2, 6152*2]

    # Validation
    assert sum(query_seqlens) <= ntokens
    assert all(s < small_request_ntokens for s in query_seqlens[1:])
    assert num_queries <= max_queries_per_batch
    assert all(s < cache_seqlen for s in context_seqlens)

    torch.manual_seed(5434)

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

    if args.validate:
        # Call flash attn
        # First for the single full-sized query
        out0, lse0 = fa3.flash_attn_with_kvcache(
            q=q_buf_large,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlen_large,
            cache_batch_idx=cache_idx_large,
            causal=bool(args.causal),
            num_splits=args.splits,
            return_softmax_lse=True,
           #num_splits=1
        )   

         # Second for n-1 small queries
        out1_split1, lse1_split1 = fa3.flash_attn_with_kvcache(
            q=q_buf_small,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_small,
            cache_batch_idx=cache_idxs_small,
            causal=bool(args.causal),
            num_splits=1,
            gqa_decoding=bool(args.gqa),
            return_softmax_lse=True,
        )

        # Second for n-1 small queries
        out1, lse1 = fa3.flash_attn_with_kvcache(
            q=q_buf_small,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_small,
            cache_batch_idx=cache_idxs_small,
            causal=bool(args.causal),
            num_splits=args.splits,
            gqa_decoding=bool(args.gqa),
            return_softmax_lse=True,
        )

        # Call flash attn
        # First for the single full-sized query
        out2 = fa2.flash_attn_with_kvcache(
            q=q_buf_large,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlen_large,
            cache_batch_idx=cache_idx_large,
            causal=bool(args.causal),
            num_splits=args.splits,
        )

        print ('big')
        print ('diff-max', (out0 - out2).abs().max().item(), cache_seqlens_small)
        print ('diff-mean', (out0 - out2).abs().mean().item())


        # Second for n-1 small queries
        out3, lse_fa2 = fa2.flash_attn_with_kvcache(
            q=q_buf_small,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_small,
            cache_batch_idx=cache_idxs_small,
            causal=bool(args.causal),
            num_splits=args.splits,
            return_softmax_lse=True,
            #num_splits=1
        )

        print ('small') #, out1)
        print ('lse', lse1, lse_fa2, (lse1 - lse_fa2).abs(), out1.shape)
        print ('lse-dif-max', (lse1 - lse_fa2).abs().max().item())
        print ('diff-max', (out1 - out3).abs().max().item())
        print ('diff-mean', (out1 - out3).abs().mean().item())


    print ('fa3', args.repeats)
    time_fa3_big = benchmark_fa_kv(fa3.flash_attn_with_kvcache, repeats=args.repeats, 
        q=q_buf_large,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlen_large,
        cache_batch_idx=cache_idx_large,
        causal=bool(args.causal),
        num_splits=args.splits,
    )

    time_fa3_small = benchmark_fa_kv(fa3.flash_attn_with_kvcache, repeats=args.repeats,
        q=q_buf_small,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens_small,
        cache_batch_idx=cache_idxs_small,
        causal=bool(args.causal),
        num_splits=args.splits,
    )

    print ('fa2 ')

    time_fa2_big = benchmark_fa_kv(fa2.flash_attn_with_kvcache, repeats=args.repeats, 
            q=q_buf_large,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlen_large,
            cache_batch_idx=cache_idx_large,
            causal=bool(args.causal),
            num_splits=args.splits
    )

    time_fa2_small = benchmark_fa_kv(fa2.flash_attn_with_kvcache, repeats=args.repeats, 
            q=q_buf_small,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_small,
            cache_batch_idx=cache_idxs_small,
            causal=bool(args.causal),
            num_splits=args.splits
    )

    print ('big (split, fa3, fa2, ratio):', args.splits, time_fa3_big * 1000000, time_fa2_big * 1000000, time_fa3_big / time_fa2_big)
    print ('small (split, fa3, fa2, ratio):', args.splits, time_fa3_small * 1000000, time_fa2_small * 1000000, time_fa3_small / time_fa2_small)

if __name__ == "__main__":
    main()
