"""dQ of the MXFP8 hdim-256 backward's MX dS mode as a block-scaled GEMM.

NOTE [MXFP8 hd256 dS for dQ]
`bwd_mode="mxds"` (opt-in; MxHd256BwdOperands.mxds) replaces the recomputing dQ kernel. Its workspace
budget (default 4 GiB) counts every workspace buffer: the dS payload of step 1, its compact scales, and the
expanded scales of step 2 (config.mxds_slice_bytes per (batch, head) slice); config.select_mxds_chunk raises
ValueError if one slice exceeds it. The other backward operands are outside it. Per chunk of slices that fits:
  1. the dK / dV kernel (sm100_hd256_2cta_fmha_backward_dkdvkernel.py) runs as in exact mode and also
     writes dS in E4M3 to a (slice, keys, queries) workspace, queries contiguous. Each compute warp holds
     a 32 x 32 (keys x queries) block of FP32 dS x 2^8 and quantizes it with one UE8M0 scale, the max of
     the dS^T scale exponents it already computes (one redux), and stores one compact scale byte per
     block. Non-causal stores go through a 64 x 128 SMEM stage that the otherwise idle warp TMA-stores;
     causal has no SMEM left for it and stores each lane's 32 bytes directly. Causal only writes the
     visited tiles (queries >= 256 floor(key / 256)), whose masked elements are zeros;
  2. `expand_ds_sf` copies the compact bytes into the GEMM's A-scale layout;
  3. the GEMM computes dQ[b, :, h] = alpha * dS_mx . K_mx with B = K-hat requantized along keys
     (`kt_operands`; softmax_scale's power of two rides in its scales, alpha is the remainder). Causal:
     the M tile of queries [256 m, 256 (m + 1)) reads only keys < 256 (m + 1), exactly the tiles step 1
     wrote. The GEMM is the CUTLASS 4.8 persistent block-scaled GEMM (cutlass_blockscaled_gemm_sm100.py;
     cutlass_blockscaled_gemm_sm107.py on sm_107), tile 256 x 256 (all of head_dim).
dK and dV are bitwise those of exact mode; only dQ changes.

Numerics. dQ reads dS through a second MXFP8 quantization (along keys) and K-hat requantized along keys
instead of dQ = dS K-hat in BF16. On random inputs, dQ's relative error against the oracle that
differentiates the quantized forward rises from ~0.027 (exact) to ~0.038. That is the error of
per-(query, 32 keys) scales too: E4M3 spans ~17 binades, so one scale per 32 x 32 block adds no error over
per-query scaling while the block's row magnitudes stay within ~2^12 of each other (beyond that, smaller
rows fall into E4M3's subnormal range); the E4M3 rounding itself remains either way. Smaller rows of a
block whose rows span more than that lose precision: in a measured adversarial example, one query whose
dO row is 2^14x its 31 neighbours' raises the error of their dQ rows from ~0.037 to ~0.12. Requantizing
K-hat with power-of-two scales along keys is exact down to E4M3's normal range (NOTE [MXFP8 hd256
backward]).
Causal. A key's dQ contribution shares a K scale with the other keys of its 32-key group, so keys
>= t0 can change the dQ rows in [32 floor(t0 / 32), t0) (an accuracy effect of the shared scale, of
the order of dQ's own rounding); rows below 32 floor(t0 / 32) are bitwise independent of keys >= t0.
The forward is unaffected.
"""

import functools
import math

import cuda.bindings.driver as cuda
import torch
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Int32
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_ptr

from flash_attn.cute import sm100_hd256_mx_utils as mxu

_SF_VEC = 32


@triton.jit
def _expand_sf_kernel(src_ptr, dst_ptr, n_tiles, TILES: tl.constexpr):
    t = tl.program_id(0) * TILES + tl.arange(0, TILES)
    byte = tl.arange(0, 16)
    m = t[:, None] < n_tiles
    v = tl.load(src_ptr + t[:, None] * 16 + byte[None, :], mask=m)
    row = tl.arange(0, 32)
    offs = t[:, None, None] * 512 + row[None, :, None] * 16 + byte[None, None, :]
    tl.store(dst_ptr + offs, tl.broadcast_to(v[:, None, :], (TILES, 32, 16)), mask=m[:, :, None])


def expand_ds_sf(ws_sf: torch.Tensor, n: int, s_q: int, s_k: int) -> torch.Tensor:
    """The dK / dV kernel's compact block scales of n slices (16 B per (slice, 128-query, 128-key)
    tile, byte 4 ((q // 32) % 4) + (k // 32) % 4) -> the GEMM's A-scale chunks (512 B per tile:
    BlockScaledBasicChunk order, whose 32 rows are equal for block scales)."""
    tiles = n * (s_q // 128) * (s_k // 128)
    out = torch.empty(tiles * 512, dtype=torch.uint8, device=ws_sf.device)
    _expand_sf_kernel[(triton.cdiv(tiles, 8),)](ws_sf, out, tiles, TILES=8, num_warps=4)
    return out


@functools.lru_cache(maxsize=None)
def _gemm_obj(sm107: bool):
    if sm107:
        from flash_attn.cute.cutlass_blockscaled_gemm_sm107 import (
            Sm107BlockScaledPersistentDenseGemmKernel,
        )

        return Sm107BlockScaledPersistentDenseGemmKernel(
            _SF_VEC, (256, 256, 64), (256, 256, 128), (2, 1), raster_order="m"
        )
    from flash_attn.cute.cutlass_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    return Sm100BlockScaledPersistentDenseGemmKernel(_SF_VEC, (256, 256), (2, 1))


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    s_q: Int32,
    s_k: Int32,
    hc: Int32,
    bc: Int32,
    h_total: Int32,
    causal: cutlass.Constexpr,
    alpha: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """dQ (M = queries, N = d, L = (h, b)) = alpha * A . B with A = dS (M, K = keys, L), M-major
    (the workspace (slice, keys, queries)), B = K-hat requantized along keys (N, K, L), N-major
    (b, s, h, d), C in (b, s, h, d)."""
    d = 256
    n = hc * bc
    a = cute.make_tensor(
        a_ptr, cute.make_layout((s_q, cute.assume(s_k, 128), n), stride=(1, s_q, s_q * s_k))
    )
    hd = h_total * d
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout((d, cute.assume(s_k, 128), (hc, bc)), stride=(1, hd, (d, s_k * hd))),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout((cute.assume(s_q, 128), d, (hc, bc)), stride=(hd, 1, (d, s_q * hd))),
    )
    sfa = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF((s_q, s_k, n), _SF_VEC))
    sfb = cute.make_tensor(sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF((d, s_k, n), _SF_VEC))
    layouts = (OperandMajorMode.MN, OperandMajorMode.MN, utils.LayoutEnum.ROW_MAJOR)
    if cutlass.const_expr(alpha == 1.0):
        gemm(a, b, sfa, sfb, c, layouts, max_active_clusters, stream, causal=causal)
    else:
        # (the GEMM applies its epilogue op to the BF16-converted accumulator: alpha rounds twice)
        epilogue = lambda x: (x * alpha).to(cutlass.BFloat16)  # noqa: E731
        gemm(a, b, sfa, sfb, c, layouts, max_active_clusters, stream, epilogue, causal)


_compiled = {}


def mxds_dq_gemm(ws, ws_sf, kt, sfkt, dq, base: int, n: int, causal: bool, alpha: float) -> None:
    """dQ of the n slices from (batch, head) slice `base` (ws / ws_sf: the chunk's workspace and
    compact scales; kt / sfkt: MxHd256DsOperands). A chunk lies within one batch (n divides the head
    count) or spans whole batches (the head count divides n), so B / C's slice index (h, b) has fixed
    strides."""
    b_tot, s_q, h_tot, d = dq.shape
    s_k = kt.shape[1]
    assert d == 256 and kt.shape[2] == h_tot
    if n <= h_tot:
        assert h_tot % n == 0 and base % n == 0
        hc, bc = n, 1
    else:
        assert n % h_tot == 0 and base % h_tot == 0
        hc, bc = h_tot, n // h_tot
    b0, h0 = divmod(base, h_tot)
    sfa = expand_ds_sf(ws_sf, n, s_q, s_k)
    sfb_slice = (d // 128) * (s_k // 128) * 512
    sm107 = mxu.is_sm107()
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(cutlass.Float8E4M3FN, ws.data_ptr(), gmem, assumed_align=256),
        make_ptr(cutlass.Float8E4M3FN, kt[b0, :, h0].data_ptr(), gmem, assumed_align=256),
        make_ptr(cutlass.Float8E8M0FNU, sfa.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, sfkt.data_ptr() + base * sfb_slice, gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, dq[b0, :, h0].data_ptr(), gmem, assumed_align=256),
    )
    sizes = (Int32(s_q), Int32(s_k), Int32(hc), Int32(bc), Int32(h_tot))
    stream = cuda.CUstream(torch.cuda.current_stream(dq.device).cuda_stream)
    key = (sm107, causal, alpha, dq.device.index)
    if key not in _compiled:
        clusters = utils.HardwareInfo().get_max_active_clusters(2)
        _compiled[key] = cute.compile(
            _launch, _gemm_obj(sm107), *ptrs, *sizes, causal, alpha, clusters, stream
        )
    _compiled[key](*ptrs, *sizes, stream)


def alpha_split(alpha: float) -> tuple[int, float]:
    """(the power-of-two exponent of alpha folded into the K scales, the residual multiplier)."""
    e = math.frexp(alpha)[1] - 1
    return e, alpha / 2.0**e
