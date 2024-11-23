# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

DTYPE_MAP_BWD = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [90]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [64, 96, 128, 192, 256]
PAGEDKV = ["false", "true"]
SPLIT = ["false", "true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {SPLIT}, {PAGEDKV}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim_16b<{DTYPE}, {HEAD_DIM}, {SPLIT}, {PAGEDKV}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_FP8 = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {SPLIT}, {PAGEDKV}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_fp8_hdim{HEAD_DIM}<{DTYPE}, {SPLIT}, {PAGEDKV}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    split: str
    paged_kv: str
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd" and self.dtype != "e4m3":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, SPLIT=self.split, PAGEDKV=self.paged_kv
            )
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD_FP8.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, SPLIT=self.split, PAGEDKV=self.paged_kv
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{'paged_' if self.paged_kv == 'true' else ''}{'split_' if self.split == 'true' else ''}sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim, split, paged_kv, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SPLIT, PAGEDKV, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, split=split, paged_kv=paged_kv, direction="fwd")
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP_BWD.keys(), HEAD_DIMENSIONS, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, split='false', paged_kv='false', direction="bwd")


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        default="instantiations",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
