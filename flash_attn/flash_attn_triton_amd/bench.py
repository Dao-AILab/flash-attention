import argparse
import torch
import triton
import time
import pandas as pd
from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_fp8_func,
    flash_attn_qkvpacked_fp8_func,
    flash_attn_varlen_fp8_func,
    flash_attn_varlen_qkvpacked_fp8_func,
)

from flash_attn.flash_attn_triton_amd.utils import input_helper
from typing import Literal, Optional
from functools import lru_cache

ARGS_TO_TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

FUNCTIONS = {
    "flash_attn_with_kvcache": flash_attn_with_kvcache,
    "flash_attn_func": flash_attn_func,
    "flash_attn_fp8_func": flash_attn_fp8_func,
    "flash_attn_kvpacked_func": flash_attn_kvpacked_func,
    "flash_attn_varlen_func": flash_attn_varlen_func,
    "flash_attn_varlen_fp8_func": flash_attn_varlen_fp8_func,
    "flash_attn_varlen_kvpacked_func": flash_attn_varlen_kvpacked_func,
    "flash_attn_qkvpacked_func": flash_attn_qkvpacked_func,
    "flash_attn_qkvpacked_fp8_func": flash_attn_qkvpacked_fp8_func,
    "flash_attn_varlen_qkvpacked_func": flash_attn_varlen_qkvpacked_func,
    "flash_attn_varlen_qkvpacked_fp8_func": flash_attn_varlen_qkvpacked_fp8_func,
}

MODES = {
        "fwd": "for forward pass only", 
        "full": "for forward and backward pass"
        }

def estimate_memory(config):
    batch, hq, hk, sq, sk, d_head, causal, dropout = config
    memory_estimate = batch * (hq * sq + hk * sk) * d_head * 4  # bytes
    return memory_estimate

@lru_cache(maxsize=100)
def generate_benchmark_configs(is_varlen: bool, packing: Optional[Literal["kv", "qkv"]]):
    """
    generates a small number of configs that cover the parameter space well
    """

    # define all parameter options as lists
    batch_sizes = [1, 64]
    if packing == "qkv":
        hq_values = hk_values = [2, 8]
        sq_values = sk_values = [256, 8192]
    else:
        hq_values = [64, 128] # test mqa/gqa
        hk_values = [16, 64]
        if is_varlen: # make sure the seqlen is greater than the batchsize so that subsequences are greater than 0
            sq_values = [128, 512]
            sk_values = [512, 2024]
        else:
            sq_values = [4, 4096]
            sk_values = [4096, 16384] # test large k values for inference perf
    d_head_values = [64, 128]
    causal_values = [False]
    dropout_values = [0.0]
    
    # generate all possible configs
    configs = []
    
    # one big loop to generate configs
    for batch in batch_sizes:
        for hq in hq_values:
            for hk in hk_values:
                for sq in sq_values:
                    for sk in sk_values:
                        for d_head in d_head_values:
                            for causal in causal_values:
                                for dropout in dropout_values:
                                    # filter configs
                                    config = (batch, hq, hk, sq, sk, d_head, causal, dropout)

                                    # skip if memory usage would be too high
                                    if estimate_memory(config) > 8 * 1024 * 1024 * 1024:  # 8 GB limit
                                        continue

                                    # we need hq to be a multiple of hk
                                    if hq % hk != 0:
                                        continue

                                    # for qkvpacked functions, q and k must have same dimensions
                                    if packing == "qkv" and (sq != sk or hq != hk):
                                        continue
                                        
                                    # add config
                                    configs.append(config)
    
    # sort by memory usage (smallest to largest) for better visualization
    configs.sort(key=estimate_memory)
    
    return configs


def create_benchmark_fn(
    fn_name: str,
    BATCH: int,
    HQ: int,
    HK: int,
    N_CTX_Q: int,
    N_CTX_K: int,
    D_HEAD: int,
    causal: bool,
    dropout_p: float,
    dtype: torch.dtype,
    mode: Literal["fwd", "full"],
    device: Literal["cpu", "cuda"],
):
    if fn_name == "flash_attn_func":
        q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_bench_fn():
            out, lse, S_dmask = flash_attn_func(
                q,
                k,
                v,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        return flash_attn_bench_fn

    elif fn_name == "flash_attn_kvpacked_func":
        q, kv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="kv", device=device)
        def flash_attn_kvpacked_bench_fn():
            out, lse, S_dmask = flash_attn_kvpacked_func(
                q,
                kv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq, dkv = torch.autograd.grad(out, (q, kv), do)

        return flash_attn_kvpacked_bench_fn
    elif fn_name == "flash_attn_qkvpacked_func":
        qkv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="qkv", device=device)
        def flash_attn_qkvpacked_bench_fn():
            out, lse, S_dmask = flash_attn_qkvpacked_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv = torch.autograd.grad(out, (qkv), do)

        return flash_attn_qkvpacked_bench_fn   
    elif fn_name == "flash_attn_varlen_func":
        q_unpad, k_unpad, v_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", device=device)
        def flash_attn_varlen_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad)
        return flash_attn_varlen_bench_fn
    elif fn_name == "flash_attn_varlen_kvpacked_func":
        q_unpad, kv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="kv", device=device)
        def flash_attn_varlen_kvpacked_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dkv_unpad = torch.autograd.grad(out_unpad, (q_unpad, kv_unpad), do_unpad)
        return flash_attn_varlen_kvpacked_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_func":
        qkv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="qkv", device=device)
        def flash_attn_varlen_qkvpacked_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_qkvpacked_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad)
        return flash_attn_varlen_qkvpacked_bench_fn
    elif fn_name == "flash_attn_with_kvcache":
        q, k_cache, v_cache, _, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_with_kvcache_bench_fn():
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                None,
                None,
                rotary_cos=None,
                rotary_sin=None,
                cache_seqlens=None,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_table=None,
                causal=causal,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
                num_splits=0,
            )
        return flash_attn_with_kvcache_bench_fn
    elif fn_name == "flash_attn_fp8_func":
        q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_f8_bench_fn():
            out, lse, S_dmask = flash_attn_fp8_func(
                q,
                k,
                v,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        return flash_attn_f8_bench_fn
    elif fn_name == "flash_attn_qkvpacked_fp8_func":
        qkv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="qkv", device=device)
        def flash_attn_qkvpacked_fp8_bench_fn():
            out, lse, S_dmask = flash_attn_qkvpacked_fp8_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv = torch.autograd.grad(out, (qkv), do)

        return flash_attn_qkvpacked_fp8_bench_fn   
    elif fn_name == "flash_attn_varlen_fp8_func":
        q_unpad, k_unpad, v_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", device=device)
        def flash_attn_varlen_fp8_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_fp8_func(
                q_unpad,
                k_unpad,
                v_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad)
        return flash_attn_varlen_fp8_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_fp8_func":
        qkv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="qkv", device=device)
        def flash_attn_varlen_qkvpacked_fp8_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_qkvpacked_fp8_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad)
        return flash_attn_varlen_qkvpacked_fp8_bench_fn
    else:
        valid_fn_names = ", ".join(FUNCTIONS.keys())
        raise ValueError(f"{fn_name} should be one of the following functions. {valid_fn_names}")


def get_packing_type(fn_name: str) -> Optional[Literal["kv", "qkv"]]:
    if "_kvpacked" in fn_name:
        packing = "kv"
    elif "_qkvpacked" in fn_name:
        packing = "qkv"
    else:
        packing = None

    return packing

def run_benchmark(args, fn_name, fn, mode):
    """
    Runs the benchmark for the provided function based on the provided arguments.
    """
    # start timing the benchmark
    start_time = time.time()
    
    # check mode
    if mode not in MODES:
        raise ValueError(f"{mode} not in {MODES}")

    # get configs
    packing = get_packing_type(fn_name)
    is_varlen = True if "varlen" in fn_name else False
    is_fp8 = True if "fp8" in fn_name else False
    dtype =  torch.float32 if is_fp8 else ARGS_TO_TORCH_DTYPE[args.dtype]
    if args.custom_config:
        # handle custom config case
        configs = [(args.b, args.hq, 
                    args.hq if not args.hk else args.hk, 
                    args.sq, 
                    args.sq if not args.sk else args.sk, 
                    args.d, 
                    args.causal, 
                    args.dropout)]
    else:
        configs = generate_benchmark_configs(is_varlen, packing)

    # check that we have at least one config
    assert len(configs) > 0

    # print bench fn
    print(f"Benchmarking {fn_name} with {len(configs)} configs in {mode} mode ...")

    # Setup benchmark configurations
    bench_configs = [
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "CAUSAL", "DROPOUT"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Time (ms)"],
            styles=[("red", "-")],
            ylabel="ms",
            plot_name=f"benchmark-{fn_name}-{dtype}-{mode}",
            args={
                "dtype": dtype,
                "mode": mode
            },
        )
    ]

    @triton.testing.perf_report(bench_configs)
    def bench_function(
        BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT, dtype, mode, provider, device="cuda"
    ):
        benchmark_fn = create_benchmark_fn(fn_name,
                                           BATCH, 
                                           HQ, 
                                           HK, 
                                           N_CTX_Q, 
                                           N_CTX_K, 
                                           D_HEAD,
                                           CAUSAL,
                                           DROPOUT,
                                           dtype,
                                           mode,
                                           device
                                           )

        # run the benchmark
        ms = triton.testing.do_bench(benchmark_fn, warmup=25, rep=100)
        return ms

    df = bench_function.run(save_path=".", print_data=True, return_df=True)[0]
    
    # calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total time for benchmarking {fn_name} in {mode} mode: {elapsed_time:.2f} seconds")

    return df

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0, help=f"Batch size")
    parser.add_argument("-hq", type=int, default=0, help=f"Q Number of heads")
    parser.add_argument("-hk", type=int, default=0, help=f"K and V Number of heads")
    parser.add_argument("-sq", type=int, default=0, help=f"Q Sequence Length")
    parser.add_argument("-sk", type=int, default=0, help=f"K and V Sequence Length")
    parser.add_argument("-d", type=int, default=0, help=f"Head Dimension")
    parser.add_argument("-causal", action="store_true", default=False, help=f"Causal")
    parser.add_argument("-dropout", type=float, default=0.0, help=f"Dropout")
    parser.add_argument("-dtype", choices=ARGS_TO_TORCH_DTYPE.keys(), default="fp16", help=f"Datatype")
    parser.add_argument(
        "-benchmark_fn",
        type=str,
        nargs="*",
        choices=FUNCTIONS.keys(),
        help=f"Function(s) to benchmark",
    )
    parser.add_argument(
        "-mode",
        type=str,
        nargs='*',
        default= ["full"],
        choices= MODES.keys(),
        help=f"Mode(s) to benchmark kernels. ",
    )
    return parser.parse_args()

def main():
    """
    Main function to run benchmarks.
    """
    # start timing the entire benchmarking process
    total_start_time = time.time()
    
    args = parse_args()

    # Validate arguments
    args.custom_config = False
    if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
        args.custom_config = True
        assert args.b and args.hq and args.sq and args.d, (
            "if custom config is specified, please provide batch, number of Q heads, Q sequence length, and head size."
        )
    assert args.dtype in ARGS_TO_TORCH_DTYPE, "only fp16, bf16 and fp32 types currently supported."

    # determine the functions to benchmark
    if args.benchmark_fn is None or len(args.benchmark_fn) == 0:
        bench_fn_list = FUNCTIONS.keys()
    else:
        bench_fn_list = args.benchmark_fn

    # run benchmarks
    combined_dfs = {}
    for fn_name in bench_fn_list:
        if fn_name not in FUNCTIONS:
            raise ValueError(f"invalid benchmark function specified: {fn_name}")
        for mode in args.mode:
            if fn_name in ["flash_attn_with_kvcache"] and mode == "full":
                mode = "fwd"
                print(f"{fn_name} does not have a backward pass.")
            
            # run bench mark
            df = run_benchmark(args, fn_name, FUNCTIONS[fn_name], mode)
            config_cols = [col for col in df.columns if col != "Time (ms)"]
            df = df.rename(columns={"Time (ms)": f"{fn_name}_{mode}_ms"})

            # add to combined dataframe
            if mode not in combined_dfs:
                combined_dfs[mode] = df
            else:
                combined_dfs[mode] = combined_dfs[mode].merge(df, on=config_cols, how="outer")
    
    # print total time for all benchmarks
    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal time for all benchmarks: {total_elapsed_time:.2f} seconds")

    if len(bench_fn_list) > 1:
        # print and save combined results
        for mode, combined_df in combined_dfs.items():
            # save the combined results
            combined_filename = f"combined_{mode}.csv"
            combined_df.to_csv(combined_filename, index=False)
            
            # print summary info
            print(f"\nCombined benchmark results for {mode} mode:")
            print(f"{combined_filename}")
            print(combined_df)

if __name__ == "__main__":
    main()
