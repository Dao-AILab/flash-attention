import argparse
import time
import torch

try:
    import cudnn
except ImportError:
    cudnn = None

from einops import rearrange

from flash_attn.cute.bench_utils import (
    flops, attention_ref,
    cudnn_fwd_setup, cudnn_bwd_setup,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None
try:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
    from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python
except ImportError:
    flash_attn_func_python = None
    flash_attn_varlen_func_python = None
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

if torch.cuda.get_device_capability()[0] != 9:
    flash_attn_func_v3 = None

from triton.testing import do_bench


# ── Autograd backward helper ────────────────────────────────────────────────

def _make_bwd_fn(fwd_fn, g, inputs):
    """Run fwd once, return a closure that benchmarks backward.

    Args:
        fwd_fn: zero-arg callable that runs the forward pass (with autograd).
        g: gradient tensor (b, seqlen, nheads, headdim_v).
        inputs: list of input tensors whose .grad should be cleared each iteration.
    """
    out = fwd_fn()
    if isinstance(out, tuple):
        out = out[0]
    g_match = g[:out.shape[0]] if g.shape[0] != out.shape[0] else g  # handle varlen
    def bwd_fn():
        for x in inputs:
            x.grad = None
        out.backward(g_match, retain_graph=True)
    return bwd_fn


# ── Backend definitions ─────────────────────────────────────────────────────
# Each setup_* function takes a context dict and returns (fwd_fn, bwd_fn).
# Either can be None if the backend doesn't support that direction for the
# given config.  fwd_fn / bwd_fn are zero-arg callables suitable for do_bench.

def setup_standard(ctx):
    if ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    fwd_fn = lambda: attention_ref(q, k, v, causal=causal)
    bwd_fn = _make_bwd_fn(fwd_fn, g, [q, k, v]) if ctx["has_backward"] else None
    return fwd_fn, bwd_fn


def setup_fa2(ctx):
    if flash_attn_func is None or ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    if ctx["headdim"] != ctx["headdim_v"]:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    dropout_p, window_size_fa, softcap = ctx["dropout_p"], ctx["window_size_fa"], ctx["softcap"]
    deterministic = ctx["deterministic"]
    if ctx["varlen"]:
        qu, ku, vu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"]
        csq, csk, sq, sk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"], ctx["seqlen_q"], ctx["seqlen"]
        fwd_fn = lambda: flash_attn_varlen_func(qu, ku, vu, csq, csk, sq, sk, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap)
        bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func(qu, ku, vu, csq, csk, sq, sk, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic), g, [qu, ku, vu]) if ctx["has_backward"] else None
    else:
        fwd_fn = lambda: flash_attn_func(q, k, v, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap)
        bwd_fn = _make_bwd_fn(lambda: flash_attn_func(q, k, v, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic), g, [q, k, v]) if ctx["has_backward"] else None
    return fwd_fn, bwd_fn


def setup_cudnn(ctx):
    if cudnn is None or ctx["headdim"] > 256 or ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    window_size_left = ctx["window_size"][0]
    # cuDNN expects (batch, nheads, seqlen, headdim) layout
    qt, kt, vt, gt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2)
    fwd_fn, o_gpu, lse_gpu = cudnn_fwd_setup(qt, kt, vt, causal=causal, window_size_left=window_size_left)
    bwd_fn = None
    if ctx["has_backward"]:
        fwd_fn()  # populate o and lse for bwd graph
        bwd_fn = cudnn_bwd_setup(qt, kt, vt, o_gpu, gt, lse_gpu, causal=causal, window_size_left=window_size_left)
    return fwd_fn, bwd_fn


def setup_fa3(ctx):
    if flash_attn_func_v3 is None:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    window_size_fa, softcap = ctx["window_size_fa"], ctx["softcap"]
    num_splits, pack_gqa, deterministic = ctx["num_splits"], ctx["pack_gqa"], ctx["deterministic"]
    k_use = ctx.get("k_paged", k) if ctx["page_size"] is not None else k
    v_use = ctx.get("v_paged", v) if ctx["page_size"] is not None else v
    if ctx["varlen"]:
        qu, ku, vu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"]
        csq, csk, sq, sk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"], ctx["seqlen_q"], ctx["seqlen"]
        fwd_fn = lambda: flash_attn_varlen_func_v3(qu, ku, vu, csq, csk, sq, sk, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa)
    else:
        fwd_fn = lambda: flash_attn_func_v3(q, k_use, v_use, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa)
    # FA3 bwd only supports headdim == headdim_v and non-fp8
    bwd_fn = None
    if ctx["has_backward"] and ctx["dtype"] != torch.float8_e4m3fn and ctx["headdim"] == ctx["headdim_v"]:
        if ctx["varlen"]:
            bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_v3(qu, ku, vu, csq, csk, sq, sk, causal=causal, window_size=ctx["window_size"], softcap=softcap, deterministic=deterministic), g, [qu, ku, vu])
        else:
            bwd_fn = _make_bwd_fn(lambda: flash_attn_func_v3(q, k, v, causal=causal, softcap=softcap), g, [q, k, v])
    return fwd_fn, bwd_fn


def setup_fa4(ctx):
    if flash_attn_func_python is None:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    window_size, softcap = ctx["window_size"], ctx["softcap"]
    pack_gqa, deterministic = ctx["pack_gqa"], ctx["deterministic"]
    sinks = ctx["sinks"]
    k_use = ctx.get("k_paged", k) if ctx["page_size"] is not None else k
    v_use = ctx.get("v_paged", v) if ctx["page_size"] is not None else v
    if ctx["varlen"]:
        qu = ctx["q_unpad"]
        ku = ctx.get("k_paged", ctx["k_unpad"]) if ctx["page_size"] is not None else ctx["k_unpad"]
        vu = ctx.get("v_paged", ctx["v_unpad"]) if ctx["page_size"] is not None else ctx["v_unpad"]
        csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]
        pt = ctx["page_table"]
        fwd_fn = lambda: flash_attn_varlen_func_python(qu, ku, vu, csq, csk, page_table=pt, causal=causal, window_size=window_size, softcap=softcap, pack_gqa=pack_gqa)
    else:
        fwd_fn = lambda: flash_attn_func_python(q, k_use, v_use, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, pack_gqa=pack_gqa)
    bwd_fn = None
    if ctx["has_backward"] and ctx["dtype"] != torch.float8_e4m3fn:
        if ctx["varlen"]:
            qu, ku, vu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"]
            csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]
            bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_python(qu, ku, vu, csq, csk, causal=causal, softcap=softcap, deterministic=deterministic), g, [qu, ku, vu])
        else:
            bwd_fn = _make_bwd_fn(lambda: flash_attn_func_python(q, k, v, causal=causal, softcap=softcap, deterministic=deterministic), g, [q, k, v])
    return fwd_fn, bwd_fn


# Ordered list of (display_name, cli_name, setup_fn)
BACKENDS = [
    ("Standard", "standard", setup_standard),
    ("FA2",      "fa2",      setup_fa2),
    ("cuDNN",    "cudnn",    setup_cudnn),
    ("FA3",      "fa3",      setup_fa3),
    ("FA4",      "fa4",      setup_fa4),
]


def get_peak_flops(device_index: int = 0, dtype: torch.dtype = torch.bfloat16) -> float | None:
    """Return peak BF16 dense TFLOPS for the given device. Returns None if unknown.

    Scaling by dtype:
      FP16 / BF16 : 1x  (identical hardware throughput)
      FP8         : 2x
    """
    # BF16 dense peak FLOPS from official NVIDIA spec sheets.
    # Checked against: num_SMs * tensor_core_clock * ops_per_SM_per_cycle.
    # The tensor core clock is NOT queryable (it differs from the max boost SM
    # clock reported by nvidia-smi), so we hardcode the spec-sheet values.
    _PEAK_BF16_FLOPS = {
        # Ampere
        "A100":  312e12,
        "A6000": 309.7e12,
        # Ada Lovelace
        "L40S":  362e12,
        # Hopper
        "H100 SXM": 989e12,
        "H100 NVL": 835e12,
        "H100 PCIe": 756e12,
        "H200":  989e12,
        "H20":   148e12,
        # Blackwell
        "GB200": 2.5e15,
        "GB300": 2.5e15,
        "B300":  2.25e15,
        "B200":  2.25e15,
    }

    device_name = torch.cuda.get_device_name(device_index)
    # Match longest key first so "H100 SXM" matches before "H100"
    peak = None
    for key in sorted(_PEAK_BF16_FLOPS, key=len, reverse=True):
        if key.lower() in device_name.lower():
            peak = _PEAK_BF16_FLOPS[key]
            break
    if peak is None:
        return None

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak *= 2
    return peak


def parse_int_k(s):
    """Parse an integer with optional k/K suffix, e.g. '8k' -> 8192."""
    s = s.strip().lower()
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def csv_ints(s):
    """Parse comma-separated integers with optional k suffix, e.g. '512,1k,2k'."""
    return [parse_int_k(x) for x in s.split(",")]


def parse_headdims(s):
    """Parse comma-separated headdim specs. Each entry is hdim or hdim-hdim_v.

    Examples:
        '128'           -> [(128, 128)]
        '192-128'       -> [(192, 128)]
        '64,128,192'    -> [(64, 64), (128, 128), (192, 192)]
        '64,128,192-128,192' -> [(64, 64), (128, 128), (192, 128), (192, 192)]
    """
    result = []
    for item in s.split(","):
        if "-" in item:
            parts = item.split("-")
            result.append((int(parts[0]), int(parts[1])))
        else:
            hdim = int(item)
            result.append((hdim, hdim))
    return result


def csv_strs(s):
    """Parse comma-separated strings, e.g. 'fa3,fa4'."""
    return [x.strip() for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark FlashAttention')
    parser.add_argument('--headdim', type=parse_headdims, default=[(128, 128)],
                        help='Head dim(s), comma-separated. Each is hdim or hdim-hdim_v. E.g. 64,128,192-128')
    parser.add_argument('--fwd', action='store_true', help='Run forward only')
    parser.add_argument('--bwd', action='store_true', help='Run backward only')
    parser.add_argument('--varlen', action='store_true', default=False)
    parser.add_argument('--causal', type=str.lower, choices=['true', 'false', 'both'], default='both',
                        help='Causal mode (default: both)')
    parser.add_argument('--seqlen', type=csv_ints, default=[8192],
                        help='Sequence length(s), comma-separated. Supports k suffix, e.g. 1k,2k,8k')
    parser.add_argument('--total-seqlen', type=parse_int_k, default='32k',
                        help='Total sequence length for batch sizing (default: 32k)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: total_seqlen // seqlen)')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--nheads', type=int, default=None,
                        help='Number of Q heads (default: 32 for hdim<=64, 16 for hdim<=192, 8 for hdim>192)')
    parser.add_argument('--nheads-kv', type=int, default=None,
                        help='Number of KV heads (default: nheads)')
    parser.add_argument('--gqa-ratio', type=int, default=None,
                        help='GQA ratio (nheads // nheads_kv). Ignored if --nheads-kv is set.')
    parser.add_argument('--backend', type=csv_strs, default=['all'],
                        help='Which backends to benchmark, comma-separated (choices: all,standard,fa2,fa3,fa4,cudnn)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup iterations (default: 5)')
    parser.add_argument('--rep', type=int, default=10,
                        help='Repetitions per benchmark (default: 10)')
    return parser.parse_args()


def main():
    args = parse_args()

    headdim_pairs = args.headdim  # list of (hdim, hdim_v) tuples

    # Parse fwd/bwd: if neither specified, do fwd only
    has_forward = args.fwd or not args.bwd
    has_backward = args.bwd

    # Parse causal
    if args.causal == 'true':
        causal_vals = [True]
    elif args.causal == 'false':
        causal_vals = [False]
    else:
        causal_vals = [False, True]

    seqlen_list = args.seqlen
    varlen = args.varlen

    # Filter backends to those requested and available
    enabled = set(args.backend)
    if 'all' in enabled:
        enabled = {cli for _, cli, _ in BACKENDS}
    active_backends = [(name, cli, fn) for name, cli, fn in BACKENDS if cli in enabled]

    # Parameters
    torch.manual_seed(0)
    dropout_p = 0.0
    dtype = torch.bfloat16
    dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    device = 'cuda'
    peak_flops = get_peak_flops(0, dtype=dtype)
    page_size = None
    softcap = 0.0
    deterministic = args.deterministic
    warmup, rep = args.warmup, args.rep

    time_f = {}
    time_b = {}

    for headdim, headdim_v in headdim_pairs:
        nheads = args.nheads if args.nheads is not None else (32 if headdim <= 64 else 16 if headdim <= 192 else 8)
        if args.nheads_kv is not None:
            nheads_kv = args.nheads_kv
        elif args.gqa_ratio is not None:
            nheads_kv = nheads // args.gqa_ratio
        else:
            nheads_kv = nheads
        has_qv = headdim == 64 and headdim_v == 512
        sinks = None

        num_splits = 0
        window_size = (None, None)
        window_size_fa = (-1, -1)
        pack_gqa = None

        for seqlen in seqlen_list:
            batch_size = args.batch_size if args.batch_size is not None else max(1, args.total_seqlen // seqlen)
            seqlen_q = seqlen

            q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
            k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
            v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)
            q, k, v = [x.detach().to(dtype).requires_grad_(has_backward) for x in [q, k, v]]
            g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)

            # Varlen tensors
            q_unpad = k_unpad = v_unpad = cu_seqlens_q = cu_seqlens_k = None
            if varlen:
                q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(has_backward) for x in [q, k, v]]
                cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
                cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen if page_size is None else None

            # Paged KV tensors
            k_paged = v_paged = page_table = None
            if page_size is not None:
                assert seqlen % page_size == 0
                k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
                page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                       "(b s) -> b s", s=seqlen // page_size)

            for causal in causal_vals:
                cfg = (headdim, headdim_v, causal, seqlen, batch_size, nheads)

                # Build context dict shared by all backends
                ctx = dict(
                    q=q, k=k, v=v, g=g, causal=causal,
                    headdim=headdim, headdim_v=headdim_v, dtype=dtype,
                    has_backward=has_backward,
                    varlen=varlen, q_unpad=q_unpad, k_unpad=k_unpad, v_unpad=v_unpad,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    seqlen_q=seqlen_q, seqlen=seqlen,
                    page_size=page_size, k_paged=k_paged, v_paged=v_paged, page_table=page_table,
                    dropout_p=dropout_p, window_size=window_size, window_size_fa=window_size_fa,
                    softcap=softcap, deterministic=deterministic,
                    num_splits=num_splits, pack_gqa=pack_gqa, sinks=sinks,
                )

                for display_name, cli_name, setup_fn in active_backends:
                    fwd_fn, bwd_fn = setup_fn(ctx)
                    if fwd_fn is not None and has_forward:
                        time.sleep(1.0)
                        print(f"Benchmarking {display_name} fwd, hdim={headdim}, seqlen={seqlen}, causal={causal}")
                        ms = do_bench(fwd_fn, warmup=warmup, rep=rep) * 1e-3
                        time_f[cfg, display_name] = ms
                    if bwd_fn is not None and has_backward:
                        time.sleep(1.0)
                        print(f"Benchmarking {display_name} bwd, hdim={headdim}, seqlen={seqlen}, causal={causal}")
                        ms = do_bench(bwd_fn, warmup=warmup, rep=rep) * 1e-3
                        time_b[cfg, display_name] = ms

    # ── Print results table ──────────────────────────────────────────────────
    backend_names = [name for name, _, _ in BACKENDS]
    shown_backends = [b for b in backend_names if any(b == k[1] for k in list(time_f) + list(time_b))]

    if not shown_backends:
        return

    col_w = 20 if peak_flops is not None else 16

    for direction, times, flops_mult in [("FWD", time_f, 1.0), ("BWD", time_b, 2.5)]:
        if not times:
            continue
        configs = sorted(set(k[0] for k in times))
        if not configs:
            continue

        col_label = "ms / TFLOPS / MFU%" if peak_flops is not None else "ms / TFLOPS"
        header = f"{'hdim':>9} {'causal':>6} {'batch':>5} {'seqlen':>6}"
        for b in shown_backends:
            header += f" {b:>{col_w}}"
        print(f"\n{'=' * len(header)}")
        print(f"  {direction} ({col_label})")
        print(f"{'=' * len(header)}")
        print(header)
        print("-" * len(header))

        for cfg in configs:
            headdim, headdim_v, causal, seqlen, batch_size, nheads = cfg
            nFLOPS = flops(batch_size, nheads, seqlen, seqlen, headdim, headdim_v, causal=causal)
            hdim_str = str(headdim) if headdim == headdim_v else f"{headdim}-{headdim_v}"
            row = f"{hdim_str:>9} {str(causal):>6} {batch_size:>5} {seqlen:>6}"
            for b in shown_backends:
                t = times.get((cfg, b))
                if t is not None:
                    tflops = flops_mult * nFLOPS / t * 1e-12
                    ms = t * 1e3
                    if peak_flops is not None:
                        mfu = flops_mult * nFLOPS / t / peak_flops * 100
                        cell = f"{ms:.2f}/{tflops:.0f}/{mfu:.1f}%"
                    else:
                        cell = f"{ms:.2f}/{tflops:.0f}"
                    row += f" {cell:>{col_w}}"
                else:
                    row += f" {'—':>{col_w}}"
            print(row)


if __name__ == '__main__':
    main()
