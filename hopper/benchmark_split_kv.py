import torch
import flash_attn
import flash_attn_interface
import itertools
import time
import math

import torch.utils.benchmark as benchmark

def round_up_to_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def timeit(fn, *args, **kwargs):
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        fn(*args, **kwargs)

    # Benchmark using PyTorch Timer
    t = benchmark.Timer(
        stmt='fn(*args, **kwargs)',
        globals={'fn': fn, 'args': args, 'kwargs': kwargs}
    )

    # Measure execution time
    measurement = t.timeit(20)  # Runs the function 20 times
    # measurement = t.blocked_autorange(min_run_time=1)
    avg_time = measurement.mean  # Average time in seconds

    return avg_time

def main():
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    max_splits = 129
    check_all_splits = True

    causal = True
    # causal = False
    # dtype=torch.float16
    dtype=torch.bfloat16
    tp_degree = 1

    torch.manual_seed(42)

    model_configs = [
        # ("Gemma-2-2B", 8, 4, 256),
        # ("Gemma-2-9B", 16, 8, 256),
        # ("Gemma-2-27B", 32, 16, 128),
        # ("Qwen-2.5-0.5B", 14, 2, 64),
        # ("Qwen-2.5-1.5B", 12, 2, 128),
        # ("Qwen-2.5-7B", 28, 4, 128),
        # ("Llama-3.1-8B", 32, 8, 128),
        ("Llama-3.1-70B", 64, 8, 128),
        # ("Mistral Large", 96, 8, 128),
        # ("Llama-3.1-405B", 128, 8, 128),
        # ("Llama-3.2-1B", 32, 8, 64),
        # ("Llama-3.2-3B", 24, 8, 128),
        # ("Nemotron-4-15B", 48, 8, 128),
    ]

    all_batch_configs = []

    all_batch_configs.extend(itertools.product(
        # [1024, 2048, 4096, 8192, 16384, 32768, 131072],  # context_seqlen
        # [4096, 16384, 65536],  # context_seqlen
        [131072],  # context_seqlen
        # [i for i in range(1, (num_sms) + 1)], # num_requests
        [1, 4, 8, 16],  # num_requests
        # [1],  # num_requests
        # [1, 4, 8, 16],  # query_seqlen
        [1],  # query_seqlen
    ))

    num_caches = max(reqs for _, reqs, _ in all_batch_configs)
    cache_seqlen = max(seqlen for seqlen, _, _ in all_batch_configs)

    for model_name, nheads_q, nheads_kv, headdim in model_configs:
        assert nheads_kv % tp_degree == 0
        print(f"***{model_name}***")
        print(f"QHEADS:{nheads_q}, KVHEADS:{nheads_kv}, HEADDIM:{headdim}, TP:{tp_degree}")
        nheads_q //= tp_degree
        nheads_kv //= tp_degree

        k_cache = torch.randn(
            (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
        )
        v_cache = torch.randn(
            (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
        )

        if check_all_splits is False:
            print(f"{'CONTEXT':<9}{'BSZ':<5}{'QLEN':<6}{'FA2':<10}{'FA3':<9}{'RATIO':<7}{'GB/s':<10}")

        for context_seqlen, num_requests, query_seqlen in all_batch_configs:
            bytes_kv = (context_seqlen * num_requests * nheads_kv * headdim * 4)
            bytes_q = (query_seqlen * num_requests * nheads_q * headdim * 4)
            blockH = round_up_to_power_of_2(nheads_q//nheads_kv)
            blockM = 128 # true for hdim 128 causal and hdim 64
            blockM_div_H = blockM//blockH
            num_work_tiles = nheads_kv * num_requests * math.ceil(query_seqlen/blockM_div_H)

            q = torch.randn((num_requests, query_seqlen, nheads_q, headdim), device="cuda", dtype=dtype)
            cache_idxs = torch.randperm(num_caches, dtype=torch.int32, device="cuda")[:num_requests]
            cache_seqlens = torch.tensor(
                [context_seqlen] * num_requests, dtype=torch.int32, device="cuda"
            )

            fa2_time_heuristic = timeit(
                flash_attn.flash_attn_with_kvcache,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_idxs,
                causal=causal,
            ) * 1000. * 1000.
            # fastest_splitk_time = float("inf")
            # fastest_splitk = 0
            # for i in range(1, max_splits):
            #     t = timeit(
            #         flash_attn.flash_attn_with_kvcache,
            #         q=q,
            #         k_cache=k_cache,
            #         v_cache=v_cache,
            #         cache_seqlens=cache_seqlens,
            #         cache_batch_idx=cache_idxs,
            #         causal=causal,
            #         num_splits=i,
            #     ) * 1000. * 1000.
            #     if t < fastest_splitk_time:
            #         fastest_splitk_time = t
            #         fastest_splitk = i

            fa3_time_one_split = timeit(
                flash_attn_interface.flash_attn_with_kvcache,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_idxs,
                causal=causal,
                pack_gqa=False,
                num_splits=1,
            ) * 1000. * 1000.

            fa3_time_gqa_heuristic = timeit(
                flash_attn_interface.flash_attn_with_kvcache,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_idxs,
                causal=causal,
                pack_gqa=True,
                num_splits=0,
                # max_seqlen_k_hint=context_seqlen
            ) * 1000. * 1000.

            if check_all_splits:

                fa3_fastest_num_splits = 0
                fa3_fastest_splitk_time = float("inf")

                for num_splits in range(1, max_splits):
                    t = timeit(
                        flash_attn_interface.flash_attn_with_kvcache,
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=False,
                        num_splits=num_splits
                    ) * 1000. * 1000.

                    out0 = flash_attn_interface.flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=False,
                        num_splits=num_splits
                    )

                    out1 = flash_attn_interface.flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=False,
                        num_splits=1
                    )

                    max_diff = (out0 - out1).abs().max().item()
                    mean_diff = (out0 - out1).abs().mean().item()
                    # print (f"splits {num_splits}, out diff-max, {max_diff}, out diff-mean, {mean_diff}, time {t:.2f}")
                    # print (f"splits {num_splits}, time {t:.2f}")

                    if math.isnan(max_diff) or math.isnan(mean_diff) or max_diff > 2e-3 or mean_diff > 1e-4:
                        print(f"Numerical error too high: Splits: {num_splits}, Max: {max_diff}, Mean: {mean_diff}")

                    if t < fa3_fastest_splitk_time:
                        fa3_fastest_splitk_time = t
                        fa3_fastest_num_splits = num_splits

                fa3_fastest_num_splits_gqa = 0
                fa3_fastest_splitk_time_gqa = float("inf")
                for num_splits in range(1, max_splits):

                    t = timeit(
                        flash_attn_interface.flash_attn_with_kvcache,
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=True,
                        num_splits=num_splits
                    ) * 1000. * 1000.

                    out0 = flash_attn_interface.flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=True,
                        num_splits=num_splits
                    )

                    out1 = flash_attn_interface.flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=True,
                        num_splits=1
                    )

                    max_diff = (out0 - out1).abs().max().item()
                    mean_diff = (out0 - out1).abs().mean().item()
                    # print (f"gqa splits {num_splits}, out gqa diff-max {max_diff}, out gqa diff-mean {mean_diff}, time {t:.2f}")
                    # print (f"gqa splits {num_splits}, time {t:.2f}")

                    if math.isnan(max_diff) or math.isnan(mean_diff) or max_diff > 2e-3 or mean_diff > 1e-4:
                        print(f"Numerical error too high (gqa): Splits: {num_splits}, Max: {max_diff}, Mean: {mean_diff}")

                    if t < fa3_fastest_splitk_time_gqa:
                        fa3_fastest_splitk_time_gqa = t
                        fa3_fastest_num_splits_gqa = num_splits

                efficiency = (num_work_tiles * fa3_fastest_num_splits_gqa)/num_sms
                heuristic_ratio = fa3_time_gqa_heuristic/fa3_fastest_splitk_time_gqa
                # remeasure to smooth anomalies
                if heuristic_ratio > 1.1:

                    fa3_time_gqa_heuristic = timeit(
                        flash_attn_interface.flash_attn_with_kvcache,
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=True,
                        # num_splits=num_splits_select,
                        # num_splits=1,
                        num_splits=0,
                        # max_seqlen_k_hint=context_seqlen
                    ) * 1000. * 1000.

                    fa3_fastest_splitk_time_gqa = timeit(
                        flash_attn_interface.flash_attn_with_kvcache,
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        cache_seqlens=cache_seqlens,
                        cache_batch_idx=cache_idxs,
                        causal=causal,
                        pack_gqa=True,
                        num_splits=fa3_fastest_num_splits_gqa
                    ) * 1000. * 1000.

            if check_all_splits is True:
                print(
                    f"CONTEXT:{context_seqlen}, BSZ:{num_requests}, QLEN:{query_seqlen}, "
                    f"FA2:{fa2_time_heuristic:.2f}, "
                    # f"FA2 MANUAL:{fastest_splitk_time:.2f}, "
                    # f"FA2 NUM SPLITS:{fastest_splitk}, "
                    # f"FA3 NOGQA NOSPLIT:{fa3_time_one_split:.2f}, "
                    # f"FA3 NOGQA SPLIT MANUAL:{fa3_fastest_splitk_time:.2f}, "
                    # f"FA3 NOSPLIT:{fa3_time_one_split_gqa:.2f}, "
                    f"FA3 SPLIT MANUAL:{fa3_fastest_splitk_time_gqa:.2f}, "
                    f"FA3:{fa3_time_gqa_heuristic:.2f}, "
                    # f"FA3 RATIO (NONSPLIT/SPLIT):{fa3_time_one_split_gqa/fa3_time_gqa_heuristic:.2f}, "
                    # f"FA2 NUM SPLITS:{fastest_splitk}, "
                    # f"FA3 NOGQA NUM SPLITS:{fa3_fastest_num_splits}, "
                    f"FA3 NUM SPLITS:{fa3_fastest_num_splits_gqa}, "
                    # f"RATIO (FA2/3):{fa2_time_heuristic/fa3_time_gqa_heuristic:.2f}, "
                    f"RATIO:{fa3_time_gqa_heuristic/fa3_fastest_splitk_time_gqa:.2f}, "
                    f"EFF:{efficiency:.2f}, "
                    f"GB/s:{bytes_kv/fa3_time_gqa_heuristic * 1e-3:.2f}"
                )

            if check_all_splits is False:
                print(
                    f"{context_seqlen:<9}{num_requests:<5}{query_seqlen:<6}"
                    f"{fa2_time_heuristic:<10.2f}{fa3_time_gqa_heuristic:<9.2f}"
                    f"{fa2_time_heuristic/fa3_time_gqa_heuristic:<7.2f}"
                    f"{bytes_kv/fa3_time_gqa_heuristic * 1e-3:<10.2f}"
                )



if __name__ == "__main__":
    main()
