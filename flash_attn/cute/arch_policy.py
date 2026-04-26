from dataclasses import dataclass

from cutlass.base_dsl.arch import Arch


@dataclass(frozen=True)
class ForwardArchPolicy:
    """Forward-kernel capabilities that should not be inferred from arch numbers."""

    name: str
    smem_capacity_arch: str
    supports_tma_o: bool
    supports_tmem: bool
    supports_tcgen05: bool
    supports_wgmma: bool
    supports_warp_mma_f16bf16: bool
    supports_stmatrix_acc_store: bool
    supports_pack_gqa: bool
    native_tensor_only: bool = False


def get_forward_arch_policy(arch) -> ForwardArchPolicy:
    """Return explicit FA4 forward capabilities for a CuTe target arch."""
    major = arch.major if hasattr(arch, "major") else arch // 10
    minor = arch.minor if hasattr(arch, "minor") else arch % 10
    arch_int = major * 10 + minor
    if arch_int // 10 == 8:
        return ForwardArchPolicy(
            name="sm80",
            smem_capacity_arch="sm_80",
            supports_tma_o=False,
            supports_tmem=False,
            supports_tcgen05=False,
            supports_wgmma=False,
            supports_warp_mma_f16bf16=True,
            supports_stmatrix_acc_store=False,
            supports_pack_gqa=True,
        )
    if arch_int // 10 == 9:
        return ForwardArchPolicy(
            name="sm90",
            smem_capacity_arch="sm_90",
            supports_tma_o=True,
            supports_tmem=False,
            supports_tcgen05=False,
            supports_wgmma=True,
            supports_warp_mma_f16bf16=False,
            supports_stmatrix_acc_store=True,
            supports_pack_gqa=True,
        )
    if arch_int // 10 in (10, 11):
        return ForwardArchPolicy(
            name="sm100",
            smem_capacity_arch="sm_100",
            supports_tma_o=True,
            supports_tmem=True,
            supports_tcgen05=True,
            supports_wgmma=False,
            supports_warp_mma_f16bf16=False,
            supports_stmatrix_acc_store=True,
            supports_pack_gqa=True,
        )
    if arch in (Arch.sm_120, Arch.sm_120a, Arch.sm_121, Arch.sm_121a) or arch_int // 10 == 12:
        return ForwardArchPolicy(
            name="sm120",
            smem_capacity_arch="sm_120",
            supports_tma_o=False,
            supports_tmem=False,
            supports_tcgen05=False,
            supports_wgmma=False,
            supports_warp_mma_f16bf16=True,
            supports_stmatrix_acc_store=False,
            supports_pack_gqa=True,
            native_tensor_only=True,
        )
    raise ValueError(f"Unsupported FA4 forward architecture: {arch}")
