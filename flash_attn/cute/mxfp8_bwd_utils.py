"""MXFP8 helpers for the SM100 hdim-256 backward (NOTE [MXFP8 hd256 backward] in
flash_bwd_sm100.py)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_layout
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op

MX_DTYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC = 32
# UE8M0 byte of the constant P^T scale 2^-8 (P~ <= 2^8 is stored unscaled in E4M3)
SF_P_BYTE = 127 - 8
# A-operand scale tile of 64 keys (K, V) and B-operand tile of 128 queries (Q, dO), bytes
SF_TILE_BYTES = 1024


def is_sm107() -> bool:
    arch = BaseDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor >= 7


def make_mx_tiled_mma(a_major, b_major, tiler_mn, cta_group=tcgen05.CtaGroup.TWO):
    """E4M3 x E4M3 block-scaled tiled MMA with one UE8M0 scale per 32 elements, K = 32 per
    instruction. The backward's pair MMAs are M = 128, and sm_107's K = 64 form is legal only at
    M = 256 (cta_group::2), so every arch uses the SM100 op."""
    return sm100_utils.make_blockscaled_trivial_tiled_mma(
        MX_DTYPE, a_major, b_major, SF_DTYPE, SF_VEC, cta_group, tiler_mn
    )


def sf_gmem_tensor(sf: cute.Tensor, mn, k, h, b) -> cute.Tensor:
    """View of an interleave_sf UE8M0 buffer (blockscaled_utils.interleave_sf, batch-major heads)
    as (mn, k, h, b)."""
    atom = blockscaled_layout.BlockScaledBasicChunk(SF_VEC).layout
    layout = cute.tile_to_shape(atom, (mn, k, h, b), (2, 1, 3, 4))
    return cute.make_tensor(cute.recast_ptr(sf.iterator, dtype=SF_DTYPE), layout)


def sf_tile_tensor(sf: cute.Tensor, n_tiles, h, b) -> cute.Tensor:
    """View of flat (b, h, n_tiles, 1024) scale-tile bytes (the pair tiles of NOTE [MXFP8 hd256
    backward]) as (16, 64, tile, h, b) for a (16, 64)-byte TMA box."""
    head_b = n_tiles * SF_TILE_BYTES
    return cute.make_tensor(
        cute.recast_ptr(sf.iterator, dtype=cutlass.Uint8),
        cute.make_layout(
            (16, 64, n_tiles, h, b), stride=(1, 16, SF_TILE_BYTES, head_b, head_b * h)
        ),
    )


@cute.jit
def make_sf_s2t(
    sSF: cute.Tensor,
    tmem_layout: cute.Layout,
    tmem_col: cutlass.Constexpr[int],
    lane_halves: cutlass.Constexpr[bool] = False,
):
    """TMEM scale tensor at column tmem_col and its SMEM -> TMEM copy (tcgen05.cp, issued by the
    leader CTA for both CTAs): (tSF, copy, src (per stage), dst). The copy broadcasts one
    128-row chunk to all four lane quadrants (.32x128b.warpx4), or with lane_halves two chunks
    to quadrants 0/1 and 2/3 (.64x128b.warpx2::01_23: the B scales of an N-split pair MMA with
    M = 128, whose accumulator holds columns N/2.. in lanes 64..127)."""
    tmem_ptr = cute.make_ptr(cutlass.Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
    tSF = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_col, dtype=SF_DTYPE), tmem_layout)
    cta_group = tcgen05.CtaGroup.TWO
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


def lane_halves_smem_layout_sfb(tmem_layout: cute.Layout, stages: int = 1) -> cute.Layout:
    """SMEM source (1024 B per stage) of a .64x128b.warpx2::01_23 copy into the TMEM SFB layout of
    an M = 128 pair MMA with N = 128, ((((32, 2), 2, 2), (32, 1)), 1, (4, k_chunks)): 16 B rows,
    rows 32..63 for lane quadrants 2/3, no SMEM copy of the quadrant replication (stride-0 mode)."""
    (((mn, halves, _rep), vec), mma_m, k) = tmem_layout.shape
    k_chunks = k[1] if isinstance(k, tuple) else 1
    return cute.make_layout(
        ((((32, 2), 2), vec), mma_m, (4, k_chunks), stages),
        stride=((((16, 4), 512), (0, 0)), 0, (1, 8), 1024),
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """PTX max.f32: ptxas fuses a dependent pair max(max(a, b), c) into one FMNMX3 and folds |x|
    operands (cute.arch.fmax(abs=True) lowers to max.xorsign.abs, which it does not fuse)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def e8m0_ceil_scale(amax: Float32, *, loc=None, ip=None) -> tuple[Int32, Float32]:
    """MX scale of a group with absolute maximum |amax| (the sign bit is ignored):
    2^ceil(log2(|amax| / 448)), the rule of blockscaled_utils, as the biased E8M0 exponent
    (clamped to [0, 254]) and the FP32 reciprocal scale. |amax| = m * 2^e with m in [1, 2) gives
    the exponent e - 8 + (m > 1.75). amax = 0 gives exponent 0 (blockscaled_utils uses scale 1);
    the group's payload is zero either way."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 b, e, m;\n\t.reg .pred p;\n\t"
        "mov.b32 b, $2;\n\tand.b32 b, b, 0x7FFFFFFF;\n\t"
        "shr.u32 e, b, 23;\n\tand.b32 m, b, 0x7FFFFF;\n\t"
        "setp.gt.u32 p, m, 0x600000;\n\tselp.s32 m, 1, 0, p;\n\t"
        "add.s32 e, e, m;\n\tsub.s32 e, e, 8;\n\t"
        "max.s32 e, e, 0;\n\tmin.s32 e, e, 254;\n\t"
        "mov.b32 $0, e;\n\tsub.s32 m, 254, e;\n\tshl.b32 m, m, 23;\n\tmov.b32 $1, m;\n\t}\n",
        "=r,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


def mx_scale(x: cute.Tensor):
    """E8M0 scale of the 32 values of x, 2^ceil(log2(amax / 448)) (blockscaled_utils' rule):
    (biased exponent, FP32 reciprocal); amax is a ternary max.f32 tree on |x| (FMNMX3). Plain
    Python: traced inline."""
    v = [cute.math.abs(x[i]) for i in range(32)]
    while len(v) > 1:
        nxt = []
        for i in range(0, len(v), 3):
            acc = v[i]
            for g in v[i + 1 : i + 3]:
                acc = fmax_f32(acc, g)
            nxt.append(acc)
        v = nxt
    return e8m0_ceil_scale(v[0])


@dsl_user_op
def cvt_e4m3x16_scaled(vals, sf_byte: Int32, *, loc=None, ip=None):
    """sm_107: 16 FP32 -> 16 E4M3 bytes (element i in byte i of four Int32 words), each multiplied
    by 2^(127 - sf_byte) inside cvt.rn.satfinite.scaled::n1::ue8m0 (no FMUL). Bitwise equal to an
    FP32 multiply by the reciprocal scale and a plain satfinite convert for normal FP32 inputs; FP32
    subnormals flush to zero. sf_byte must be per-lane data (a constant or kernel parameter fails in
    ptxas)."""
    asm = "{ .reg .b16 lo, hi; .reg .b8 sf;\n cvt.u8.u32 sf, $20;\n"
    for w in range(4):
        a0, a1, a2, a3 = (4 + 4 * w + i for i in range(4))
        asm += (
            f" cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32 lo, ${a1}, ${a0}, sf;\n"
            f" cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32 hi, ${a3}, ${a2}, sf;\n"
            f" mov.b32 ${w}, {{lo, hi}};\n"
        )
    asm += "}"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in vals]
        + [Int32(sf_byte).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,=r,=r,=r," + ",".join(["f"] * 16) + ",r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@cute.jit
def quantize_e4m3x32(x: cute.Tensor, sf: Int32, inv_scale: Float32, dst: cute.Tensor):
    """dst = E4M3(x * 2^(127 - sf)) for the 32 values of x (dst: 32 E4M3); x is scaled in place
    off sm_107, where cvt_e4m3x16_scaled fuses the scale into the convert."""
    if cutlass.const_expr(is_sm107()):
        words = cute.recast_tensor(dst, Int32)
        for h in cutlass.range_constexpr(2):
            w4 = cvt_e4m3x16_scaled([x[16 * h + i] for i in range(16)], sf)
            for i in cutlass.range_constexpr(4):
                words[4 * h + i] = w4[i]
    else:
        for i in cutlass.range_constexpr(0, 32, 2):
            x[i], x[i + 1] = cute.arch.mul_packed_f32x2((x[i], x[i + 1]), (inv_scale, inv_scale))
        dst.store(x.load().to(dst.element_type))
