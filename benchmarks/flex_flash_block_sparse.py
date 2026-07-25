"""Exercise FA4 block sparsity through nightly PyTorch FlexAttention.

This path deliberately lets ``create_block_mask`` and Inductor own sparse tensor
construction. ``BACKEND='FLASH'`` is forced so unsupported cases fail instead of
silently falling back to Triton.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
Phase = Literal["discovery", "holdout"]
MaskName = Literal["block_diagonal_128", "causal_window_128"]


@dataclass(frozen=True)
class FlexBlockSparseCase:
    """One model-backed FlexAttention block-sparse workload."""

    name: str
    phase: Phase
    profile: str
    batch: int
    q_heads: int
    kv_heads: int
    d: int
    dv: int
    seqlen_q: int
    seqlen_k: int
    mask_name: MaskName


@dataclass
class FlexBlockSparseInputs:
    """Fixed tensors, block mask, and compiled nightly FlexAttention callable."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_mask: BlockMask
    call: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, BlockMask], torch.Tensor]


def block_diagonal_128(
    batch: torch.Tensor,
    head: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """Keep query and key positions in the same 128-token document block."""
    return query // 128 == key // 128


def causal_window_128(
    batch: torch.Tensor,
    head: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """Keep a causal window of 129 positions for aligned self-attention."""
    center = query
    return (key <= center) & (key >= center - 128)


MASK_MODS = {
    "block_diagonal_128": block_diagonal_128,
    "causal_window_128": causal_window_128,
}


@dataclass(frozen=True)
class FlexFlashCall:
    """Force one compiled FlexAttention call through FA4."""

    enable_gqa: bool

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        """Run nightly FlexAttention with fallback disabled."""
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=self.enable_gqa,
            kernel_options={"BACKEND": "FLASH"},
        )


def ensure_local_fa4_package() -> None:
    """Expose this checkout's FA4 package without importing legacy FA2 bindings."""
    if "flash_attn" not in sys.modules:
        package = types.ModuleType("flash_attn")
        package.__path__ = [str(REPO_ROOT / "flash_attn")]
        sys.modules["flash_attn"] = package
    from torch._inductor.kernel.flex.flex_flash_attention import ensure_flash_available

    ensure_flash_available.cache_clear()
    if not ensure_flash_available():
        raise RuntimeError("Nightly FlexAttention cannot discover this checkout's FA4")


def generate_flex_block_sparse_cases(path: Path) -> tuple[FlexBlockSparseCase, ...]:
    """Expand the bounded block-sparse section of the realistic workload spec."""
    spec = yaml.safe_load(path.read_text())
    profiles = {profile["id"]: profile for profile in spec["model_profiles"]}
    section = spec["scenario_templates"]["flex_block_sparse"]
    cases = []
    for phase in ("discovery", "holdout"):
        for raw in section[phase]:
            profile = profiles[raw["profile"]]
            for q_length, k_length in raw["qk"]:
                for batch in raw["batches"]:
                    for mask_name in raw["masks"]:
                        if mask_name == "causal_window_128" and q_length != k_length:
                            raise ValueError(
                                "Flex causal-window cases require aligned self-attention"
                            )
                        cases.append(
                            FlexBlockSparseCase(
                                name=(
                                    f"{profile['id']}__flex_sparse__{mask_name}__"
                                    f"q{q_length}_k{k_length}_b{batch}"
                                ),
                                phase=phase,
                                profile=profile["id"],
                                batch=batch,
                                q_heads=profile["q_heads"],
                                kv_heads=profile["kv_heads"],
                                d=profile["d"],
                                dv=profile["dv"],
                                seqlen_q=q_length,
                                seqlen_k=k_length,
                                mask_name=mask_name,
                            )
                        )
    return tuple(cases)


def make_flex_block_sparse_inputs(
    case: FlexBlockSparseCase,
    dtype: torch.dtype = torch.bfloat16,
) -> FlexBlockSparseInputs:
    """Compile one fixed-shape FlexAttention invocation with FA4 forced."""
    ensure_local_fa4_package()
    q = torch.randn(
        case.batch,
        case.q_heads,
        case.seqlen_q,
        case.d,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        case.batch,
        case.kv_heads,
        case.seqlen_k,
        case.d,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        case.batch,
        case.kv_heads,
        case.seqlen_k,
        case.dv,
        device="cuda",
        dtype=dtype,
    )
    block_mask = create_block_mask(
        MASK_MODS[case.mask_name],
        B=None,
        H=None,
        Q_LEN=case.seqlen_q,
        KV_LEN=case.seqlen_k,
        device="cuda",
        BLOCK_SIZE=(256, 128),
        separate_full_blocks=True,
    )
    return FlexBlockSparseInputs(
        q=q,
        k=k,
        v=v,
        block_mask=block_mask,
        call=torch.compile(
            FlexFlashCall(case.q_heads != case.kv_heads),
            fullgraph=True,
            dynamic=False,
        ),
    )


def flex_block_sparse_reference(
    case: FlexBlockSparseCase,
    inputs: FlexBlockSparseInputs,
) -> torch.Tensor:
    """Compute an independent float32 reference for a small sparse case."""
    q, k, v = inputs.q.float(), inputs.k.float(), inputs.v.float()
    if case.q_heads != case.kv_heads:
        repeat = case.q_heads // case.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(case.d)
    query = torch.arange(case.seqlen_q, device="cuda")[:, None]
    key = torch.arange(case.seqlen_k, device="cuda")[None, :]
    if case.mask_name == "block_diagonal_128":
        keep = query // 128 == key // 128
    else:
        center = query + case.seqlen_k - case.seqlen_q
        keep = (key <= center) & (key >= center - 128)
    scores.masked_fill_(~keep, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v
