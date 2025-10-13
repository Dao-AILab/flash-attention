import torch
from flash_attn.cute.interface import flash_attn_func, flash_attn_combine


def attn_ref(q, k, v):
    s = torch.einsum("bqhd,bkhd->bhqk", q, k)
    s *= q.shape[-1] ** -0.5
    s = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", s, v)


def main(should_log: bool = False):
    from triton.testing import do_bench
    from cutlass.base_dsl.utils.logger import setup_log
    import faulthandler
    import logging

    faulthandler.enable()

    if should_log:
        setup_log("cutlass", log_to_console=True, log_level=logging.INFO)

    B, Q, K, H, D = 1, 256, 32768, 32, 64

    q = torch.randn(B, Q, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, K, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, K, H, D, device="cuda", dtype=torch.bfloat16)

    # def fn():
    #     return flash_attn_func(q, k, v)[0]

    def fn():
        out_partial, lse_partial = flash_attn_func(q, k, v, num_splits=4)
        out, _ = flash_attn_combine(
            out_partial,
            lse_partial.transpose(2, 3),
            out_dtype=torch.bfloat16,
            return_lse=False,
        )
        return out

    results = do_bench(fn)
    flops = 2 * 2 * B * Q * K * H * D
    print("Flash Attention Benchmark:")
    print(f"  B: {B}")
    print(f"  Q: {Q}")
    print(f"  K: {K}")
    print(f"  H: {H}")
    print(f"  D: {D}")
    print(f"  Avg time: {results} ms")
    print(f"  TFLOPS: {flops / results * 1e-9} TFLOPS")

    o = fn()
    o_ref = attn_ref(q, k, v)

    correctness = torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)

    if not correctness:
        print("Output does not match reference")
        print(f"  Max diff: {(o - o_ref).abs().max().item()}")
        print(f"  Mean diff: {(o - o_ref).abs().mean().item()}")
        print(f"  First 10 elements: {o.flatten()[:10]}")
    else:
        print("Output matches reference")

    return results

if __name__ == "__main__":
    main()