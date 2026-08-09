import math

import pytest

from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100


@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("head_dim_v", range(8, 129, 8))
def test_dkv_reduce_widths_divide_their_head_dims(head_dim, head_dim_v):
    use_2cta_instrs = head_dim >= 128
    kernel = FlashAttentionBackwardSm100(
        head_dim,
        head_dim_v,
        cluster_size=2 if use_2cta_instrs else 1,
        use_2cta_instrs=use_2cta_instrs,
    )
    kernel._setup_attributes()

    assert kernel.dK_reduce_ncol == math.gcd(32, kernel.tile_hdim // 2)
    assert kernel.dV_reduce_ncol == math.gcd(32, kernel.tile_hdimv // 2)
    assert (kernel.tile_hdim // 2) % kernel.dK_reduce_ncol == 0
    assert (kernel.tile_hdimv // 2) % kernel.dV_reduce_ncol == 0


def test_asymmetric_dv_uses_its_own_reduce_width():
    kernel = FlashAttentionBackwardSm100(
        128,
        96,
        cluster_size=2,
        use_2cta_instrs=True,
    )
    kernel._setup_attributes()

    assert kernel.dK_reduce_ncol == 32
    assert kernel.dV_reduce_ncol == 16
