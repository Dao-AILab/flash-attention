import cutlass
from cutlass.base_dsl.arch import Arch

from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120


def test_sm120_forward_uses_sm80_control_flow():
    kernel = FlashAttentionForwardSm120(
        cutlass.BFloat16,
        head_dim=64,
        head_dim_v=64,
        qhead_per_kvhead=1,
        pack_gqa=False,
        tile_m=128,
        tile_n=64,
        num_stages=1,
        num_threads=128,
    )

    assert kernel.arch == Arch.sm_80
