"""MXFP8 helpers for the dedicated hdim-256 backward kernels.

See NOTE [MXFP8 hd256 backward] in sm100_hd256_2cta_fmha_backward.py.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_layout
from cutlass.cutlass_dsl import BaseDSL

from flash_attn.cute import utils as fa_utils

MX_DTYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC = 32


def is_sm107() -> bool:
    arch = BaseDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor >= 7


def make_mx_tiled_mma(
    a_major, b_major, cta_group, tiler_mn, a_source=tcgen05.OperandSource.SMEM, k32=False
):
    """E4M3 x E4M3 block-scaled tiled MMA (UE8M0 per 32); sm_107 uses the K=64 instruction
    unless k32: K=64 is only legal for M=256 with cta_group::2 (PTX shape table) and the SM107
    op only takes K=64, so the dK/dV kernel's M=128 pair MMAs (and their scale-layout proxies and
    partition forms) use the SM100 K=32 op on sm_107 too."""
    if is_sm107() and not k32:
        from cutlass.utils import rubin_helpers

        return rubin_helpers.make_blockscaled_trivial_tiled_mma(
            MX_DTYPE,
            MX_DTYPE,
            a_major,
            b_major,
            SF_DTYPE,
            SF_VEC,
            cta_group,
            (*tiler_mn, 64),
            a_source=a_source,
        )
    return sm100_utils.make_blockscaled_trivial_tiled_mma(
        MX_DTYPE,
        a_major,
        b_major,
        SF_DTYPE,
        SF_VEC,
        cta_group,
        tiler_mn,
        a_source=a_source,
    )


def sf_gmem_tensor(sf: cute.Tensor, mn, k, h_r, h_k, b) -> cute.Tensor:
    """View of a raw interleaved UE8M0 buffer (blockscaled_utils.interleave_sf storage, heads
    ordered h = h_k * h_r_count + h_r) as (mn, k, ((h_r, h_k), b))."""
    atom = blockscaled_layout.BlockScaledBasicChunk(SF_VEC).layout
    L = cute.tile_to_shape(atom, (mn, k, h_r, h_k, b), (2, 1, 3, 4, 5))
    layout = cute.make_layout(
        (L.shape[0], L.shape[1], ((L.shape[2], L.shape[3]), L.shape[4])),
        stride=(L.stride[0], L.stride[1], ((L.stride[2], L.stride[3]), L.stride[4])),
    )
    return cute.make_tensor(cute.recast_ptr(sf.iterator, dtype=SF_DTYPE), layout)


@cute.jit
def split_wg_contiguous(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """split_wg with contiguous t2r chunks per warp group (rank-3 t2r partitions): a thread
    then holds adjacent TMEM columns, e.g. 32 contiguous queries of one key for the MXFP8
    1x32 groups."""
    assert cute.rank(t.layout) == 3
    p = cute.composition(
        t,
        cute.make_layout(
            (
                t.shape[0],
                t.shape[1],
                (cute.size(t, mode=[2]) // num_warp_groups, num_warp_groups),
            )
        ),
    )
    return p[None, None, (None, wg_idx)]


def mx_sd_view(t: cute.Tensor, hb, bcast_r: bool) -> cute.Tensor:
    """MXFP8: (b, s, h_k, h_r, d) E4M3 copy -> (s, d, ((h_r, h_k), b)), broadcast over h_r for
    K / V."""
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (t.shape[1], t.shape[4], hb),
            stride=(
                cute.assume(t.stride[1], divby=64),
                t.stride[4],
                (
                    (0 if bcast_r else t.stride[3], t.stride[2]),
                    cute.assume(t.stride[0], divby=64),
                ),
            ),
        ),
    )


def mx_sf_tile_view(t: cute.Tensor, rows: int, n_tiles, hb, h_r, h_k, bcast_r: bool) -> cute.Tensor:
    """MXFP8: flat (b, h, n_tiles, 16 * rows) scale-tile bytes (NOTE [MXFP8 hd256 backward]) as
    (16, rows, tile, ((h_r, h_k), b)) for TMA; K / V tiles broadcast over h_r."""
    tile_b = 16 * rows
    head_b = n_tiles * tile_b
    heads = h_k if bcast_r else h_k * h_r
    return cute.make_tensor(
        cute.recast_ptr(t.iterator, dtype=cutlass.Uint8),
        cute.make_layout(
            (16, rows, n_tiles, hb),
            stride=(
                1,
                16,
                tile_b,
                ((0 if bcast_r else head_b, head_b if bcast_r else head_b * h_r), head_b * heads),
            ),
        ),
    )


@cute.jit
def make_sf_s2t(
    sSF: cute.Tensor,
    tmem_layout: cute.Layout,
    tmem_col: cutlass.Int32,
    cta_group,
    lane_halves: cutlass.Constexpr[bool] = False,
):
    """TMEM scale tensor at column tmem_col and its SMEM -> TMEM copy (tcgen05.cp, issued by the
    leader CTA for both CTAs): (tSF, copy, src (per stage), dst). The copy broadcasts one
    128-row chunk to all four lane quadrants (.32x128b.warpx4), or with lane_halves two chunks
    to quadrants 0/1 and 2/3 (.64x128b.warpx2::01_23: the B scales of an N=256 pair MMA with
    M=128, whose accumulator holds columns N/2.. in lanes 64..127)."""
    tmem_ptr = cute.make_ptr(cutlass.Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
    tSF = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_col, dtype=SF_DTYPE), tmem_layout)
    op = (
        tcgen05.Cp2x64x128b0123Op(cta_group)
        if cutlass.const_expr(lane_halves)
        else tcgen05.Cp4x32x128bOp(cta_group)
    )
    atom = cute.make_copy_atom(op, SF_DTYPE)
    copy = tcgen05.make_s2t_copy(atom, cute.filter_zeros(tSF))
    thr = copy.get_slice(0)
    src = tcgen05.get_s2t_smem_desc_tensor(copy, thr.partition_S(cute.filter_zeros(sSF)))
    dst = thr.partition_D(cute.filter_zeros(tSF))
    return tSF, copy, src, dst


@cute.jit
def mx_scale(x: cute.Tensor):
    """E8M0 scale of the 32 values of x, 2^ceil(log2(amax / 448)) (blockscaled_utils' rule):
    (biased exponent, FP32 reciprocal); amax is a depth-5 tree of xorsign-abs max."""
    m1 = [cute.arch.fmax(x[i], x[i + 1], abs=True) for i in range(0, 32, 2)]
    m2 = [cute.arch.fmax(m1[i], m1[i + 1], abs=True) for i in range(0, 16, 2)]
    m3 = [cute.arch.fmax(m2[i], m2[i + 1], abs=True) for i in range(0, 8, 2)]
    m4 = [cute.arch.fmax(m3[i], m3[i + 1], abs=True) for i in range(0, 4, 2)]
    return fa_utils.e8m0_ceil_scale(cute.arch.fmax(m4[0], m4[1], abs=True))


@cute.jit
def mx_apply(x: cute.Tensor, inv_scale: cutlass.Float32, dst: cute.Tensor):
    """dst = E4M3(x * inv_scale); scales x in place."""
    for i in cutlass.range_constexpr(0, cute.size(x), 2):
        x[i], x[i + 1] = cute.arch.mul_packed_f32x2((x[i], x[i + 1]), (inv_scale, inv_scale))
    dst.store(x.load().to(dst.element_type))


def lane_halves_smem_layout_sfb(tmem_layout: cute.Layout, stages: int = 1) -> cute.Layout:
    """SMEM source (1024 B per stage) of a .64x128b.warpx2::01_23 copy into the TMEM SFB layout of an
    M=128 pair MMA with N=128, ((((32, 2), 2, 2), (32, 1)), 1, (4, k_chunks)): 16 B rows, rows
    32..63 for lane quadrants 2/3, no SMEM copy of the quadrant replication (stride-0 mode)."""
    (((mn, halves, _rep), vec), mma_m, k) = tmem_layout.shape
    k_chunks = k[1] if isinstance(k, tuple) else 1
    layout = cute.make_layout(
        ((((32, 2), 2), vec), mma_m, (4, k_chunks), stages),
        stride=((((16, 4), 512), (0, 0)), 0, (1, 8), 1024),
    )
    return layout
