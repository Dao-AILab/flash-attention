import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref


@cute.jit
def probe_p_operand_a_layout():
    cta_group = tcgen05.CtaGroup.TWO
    dtype = BFloat16
    acc_dtype = Float32
    mma_tiler_qk = (128, 128, 256)
    mma_tiler_pv = (128, 256, 128)

    tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        cta_group,
        mma_tiler_qk[:2],
    )
    tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        acc_dtype,
        cta_group,
        mma_tiler_pv[:2],
        tcgen05.OperandSource.TMEM,
    )

    for cta in cutlass.range_constexpr(2):
        thr_mma_qk = tiled_mma_qk.get_slice(cta)
        thr_mma_pv = tiled_mma_pv.get_slice(cta)

        s_acc_shape = thr_mma_qk.partition_shape_C(mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(s_acc_shape)

        tP_layout = cute.select(
            sm100_utils.make_smem_layout_a(
                tiled_mma_pv,
                mma_tiler_pv,
                dtype,
                1,
            ),
            mode=[0, 1, 2],
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)

        tSAcc = tStS[(None, None), 0, 0]
        tileP_like_f32 = mma_tiler_qk[1] // Float32.width * dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout,
            cute.make_layout((tSAcc.shape[0], tileP_like_f32)),
        )
        tStP = cute.make_tensor(tSAcc.iterator, tStP_layout)

        print("CTA", cta)
        print("  tStS:", tStS.type)
        print("  tSAcc:", tSAcc.type)
        print("  tP_layout.outer:", tP_layout.outer)
        print("  tOrP:", tOrP.type)
        print("  tileP_like_f32:", tileP_like_f32)
        print("  tStP:", tStP.type)


def test_probe_p_operand_a_layout():
    probe_p_operand_a_layout()


@cute.jit
def probe_dkdv_p_operand_a_layout():
    cta_group = tcgen05.CtaGroup.TWO
    dtype = BFloat16
    acc_dtype = Float32
    mma_tiler_kq = (128, 128, 256)
    mma_tiler_pdo = (128, 256, 128)

    tiled_mma_s = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        cta_group,
        mma_tiler_kq[:2],
    )
    tiled_mma_dv = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        acc_dtype,
        cta_group,
        mma_tiler_pdo[:2],
        tcgen05.OperandSource.TMEM,
    )

    for cta in cutlass.range_constexpr(2):
        thr_mma_s = tiled_mma_s.get_slice(cta)
        thr_mma_dv = tiled_mma_dv.get_slice(cta)

        s_shape = thr_mma_s.partition_shape_C(mma_tiler_kq[:2])
        tStS = thr_mma_s.make_fragment_C(s_shape)
        tP_layout = cute.select(
            sm100_utils.make_smem_layout_a(tiled_mma_dv, mma_tiler_pdo, dtype, 1),
            mode=[0, 1, 2],
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tdVrP = tiled_mma_dv.make_fragment_A(tP)
        dv_shape = thr_mma_dv.partition_shape_C(mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dv.make_fragment_C(dv_shape)
        sdO_layout = sm100_utils.make_smem_layout_b(tiled_mma_dv, mma_tiler_pdo, dtype, 1)
        fake_smem = cute.make_ptr(dtype, 0, mem_space=cute.AddressSpace.smem, assumed_align=1024)
        sdO = cute.make_tensor(fake_smem, sdO_layout.outer)
        tdVrdO = tiled_mma_dv.make_fragment_B(sdO)

        print("DKDV CTA", cta)
        print("  dV op:", tiled_mma_dv.op)
        print("  tdVrP:", tdVrP.type)
        print("  tdVtdV:", tdVtdV.type)
        print("  tdVrdO:", tdVrdO.type)


def test_probe_dkdv_p_operand_a_layout():
    probe_dkdv_p_operand_a_layout()


@cute.jit
def probe_dkdv_p_operand_a_layout_n128():
    cta_group = tcgen05.CtaGroup.TWO
    dtype = BFloat16
    acc_dtype = Float32
    mma_tiler_kq = (128, 128, 256)
    mma_tiler_pdo = (128, 128, 128)

    tiled_mma_s = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        cta_group,
        mma_tiler_kq[:2],
    )
    tiled_mma_dv = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        acc_dtype,
        cta_group,
        mma_tiler_pdo[:2],
        tcgen05.OperandSource.TMEM,
    )

    for cta in cutlass.range_constexpr(2):
        thr_mma_s = tiled_mma_s.get_slice(cta)
        thr_mma_dv = tiled_mma_dv.get_slice(cta)

        s_shape = thr_mma_s.partition_shape_C(mma_tiler_kq[:2])
        tStS = thr_mma_s.make_fragment_C(s_shape)
        tP_layout = cute.select(
            sm100_utils.make_smem_layout_a(tiled_mma_dv, mma_tiler_pdo, dtype, 1),
            mode=[0, 1, 2],
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tdVrP = tiled_mma_dv.make_fragment_A(tP)
        dv_shape = thr_mma_dv.partition_shape_C(mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dv.make_fragment_C(dv_shape)
        sdO_layout = sm100_utils.make_smem_layout_b(tiled_mma_dv, mma_tiler_pdo, dtype, 1)
        fake_smem = cute.make_ptr(dtype, 0, mem_space=cute.AddressSpace.smem, assumed_align=1024)
        sdO = cute.make_tensor(fake_smem, sdO_layout.outer)
        tdVrdO = tiled_mma_dv.make_fragment_B(sdO)

        print("DKDV N128 CTA", cta)
        print("  dV op:", tiled_mma_dv.op)
        print("  tdVrP:", tdVrP.type)
        print("  tdVtdV:", tdVtdV.type)
        print("  sdO_layout:", sdO_layout.outer)
        print("  tdVrdO:", tdVrdO.type)


def test_probe_dkdv_p_operand_a_layout_n128():
    probe_dkdv_p_operand_a_layout_n128()


@cute.jit
def probe_dkdv_64x128_2cta_exact_p_views():
    cta_group = tcgen05.CtaGroup.TWO
    dtype = BFloat16
    acc_dtype = Float32
    tile_n = 64
    tile_m = 128
    tile_hdim = 256

    # dkdv producer: S = K @ Q.T.
    # Per CTA this is 64x128; across 2 CTA it is logical 128x128.
    mma_tiler_kq = (2 * tile_n, tile_m, tile_hdim)
    tiled_mma_s = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        cta_group,
        mma_tiler_kq[:2],
    )

    # dkdv consumer: dV = P.T @ dO. This is the direct no-SMEM candidate.
    # It asks the following MMA to read P from TMEM as operand A.
    mma_tiler_pdo = (2 * tile_n, tile_hdim, tile_m)
    tiled_mma_dv = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        acc_dtype,
        cta_group,
        mma_tiler_pdo[:2],
        tcgen05.OperandSource.TMEM,
    )

    # Same consumer, but split to 128 hdim to show this does not change P operand ownership.
    mma_tiler_pdo_stage = (2 * tile_n, tile_hdim // 2, tile_m)
    tiled_mma_dv_stage = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        acc_dtype,
        cta_group,
        mma_tiler_pdo_stage[:2],
        tcgen05.OperandSource.TMEM,
    )

    for cta in cutlass.range_constexpr(2):
        thr_mma_s = tiled_mma_s.get_slice(cta)
        thr_mma_dv = tiled_mma_dv.get_slice(cta)
        thr_mma_dv_stage = tiled_mma_dv_stage.get_slice(cta)

        s_shape = thr_mma_s.partition_shape_C(mma_tiler_kq[:2])
        tStS = thr_mma_s.make_fragment_C(s_shape)
        tSAcc = tStS[(None, None), 0, 0]

        # This is the no-SMEM P writeback view used by the dkdv experiment:
        # 128 bf16 P columns are packed as 64 f32-like TMEM columns.
        tileP_f32_like = tile_m // Float32.width * dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout,
            cute.make_layout((tile_n, tileP_f32_like)),
        )
        tStP = cute.make_tensor(tSAcc.iterator, tStP_layout)

        tP_layout = cute.select(
            sm100_utils.make_smem_layout_a(
                tiled_mma_dv,
                mma_tiler_pdo,
                dtype,
                1,
            ),
            mode=[0, 1, 2],
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tdVrP = tiled_mma_dv.make_fragment_A(tP)

        tP_stage_layout = cute.select(
            sm100_utils.make_smem_layout_a(
                tiled_mma_dv_stage,
                mma_tiler_pdo_stage,
                dtype,
                1,
            ),
            mode=[0, 1, 2],
        )
        tP_stage = cute.make_tensor(tStS.iterator, tP_stage_layout.outer)
        tdVrP_stage = tiled_mma_dv_stage.make_fragment_A(tP_stage)

        print("DKDV exact CTA", cta)
        print("  producer tStS:", tStS.type)
        print("  producer tSAcc:", tSAcc.type)
        print("  producer tStP:", tStP.type)
        print("  consumer dV op:", tiled_mma_dv.op)
        print("  consumer tdVrP:", tdVrP.type)
        print("  consumer-stage dV op:", tiled_mma_dv_stage.op)
        print("  consumer-stage tdVrP:", tdVrP_stage.type)


def test_dkdv_64x128_2cta_p_tmem_views_are_not_same_layout(capsys):
    probe_dkdv_64x128_2cta_exact_p_views()
    out = capsys.readouterr().out

    # The writeback view is the producer's packed f32-like TMEM store view.
    assert 'producer tStP: !cute.memref<f32, tmem, align<1>, "(64,64):(65536,1)">' in out

    # The consumer view is the next MMA's operand-A ownership view.
    assert (
        'consumer tdVrP: !cute.memref<bf16, tmem, align<1>, '
        '"((128,16),1,(4,2)):((65536,1),0,(16,64))">'
    ) in out

    # Splitting dV hdim to 128 changes the accumulator shape, not this operand-A view.
    assert (
        'consumer-stage tdVrP: !cute.memref<bf16, tmem, align<1>, '
        '"((128,16),1,(4,2)):((65536,1),0,(16,64))">'
    ) in out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100",
)
def test_existing_forward_2cta_c_to_d_to_o_numeric_smoke():
    """Numerical smoke for the same C->D->O TMEM pattern in the HD256 forward kernel.

    This is intentionally written as C=A@B, D=f(C), O=D@M:
      A = q
      B = k.T
      f = softmax(scale * C)
      M = v

    The HD256 SM100 forward kernel computes C in TMEM, loads C directly from TMEM,
    writes D directly back to TMEM in packed layout, and consumes D as TMEM
    Operand A for the second 2CTA MMA. No SMEM conversion is used for C->D.
    """
    torch.manual_seed(0)
    a = torch.randn(1, 128, 16, 256, device="cuda", dtype=torch.bfloat16) / 8
    b = torch.randn(1, 128, 16, 256, device="cuda", dtype=torch.bfloat16) / 8
    m = torch.randn(1, 128, 16, 256, device="cuda", dtype=torch.bfloat16) / 8

    out, _ = flash_attn_func(a, b, m, causal=False, return_lse=True)

    scale = 1.0 / (a.shape[-1] ** 0.5)
    c = torch.einsum("bthd,bshd->bhts", a.float() * scale, b.float())
    d = torch.softmax(c, dim=-1).to(torch.bfloat16)
    out_ref = torch.einsum("bhts,bshd->bthd", d.float(), m.float()).to(torch.bfloat16)

    max_diff = (out - out_ref).abs().max().item()
    ref_scale = out_ref.abs().max().item()
    print("existing forward 2CTA C->D(TMEM Operand A)->O max_diff:", max_diff)
    assert max_diff <= 2e-2 + 2e-2 * ref_scale
