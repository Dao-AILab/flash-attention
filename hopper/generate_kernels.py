# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

KERNEL_BATCH = namedtuple("Kernel", ["template", "filename"])

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

DTYPE_MAP_FWD_SM8x = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

DTYPE_MAP_BWD = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80, 90]  # Sm kernels support up to
HEAD_DIMENSIONS = [64, 96, 128, 192, 256]
PAGEDKV = [False, True]
SPLIT = [False, True]
SOFTCAP = [False, True]
PACKGQA = [False, True]

KERNEL_IMPL_TEMPLATE_FWD_SM90 = """#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""

KERNEL_IMPL_TEMPLATE_FWD_SM8x = """#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<80, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_<86, {DTYPE}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}, {SOFTCAP}, {PACKGQA}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
#endif
"""

KERNEL_IMPL_TEMPLATE_BWD_SM90 = """#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template<>
void run_mha_bwd_<{ARCH}, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{ARCH}, {DTYPE}, {SOFTCAP}>(params, stream);
}}
#endif
"""

KERNEL_IMPL_TEMPLATE_BWD_SM8x = """#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template<>
void run_mha_bwd_<80, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<80, {DTYPE}, {SOFTCAP}>(params, stream);
}}
template<>
void run_mha_bwd_<86, {DTYPE}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<86, {DTYPE}, {SOFTCAP}>(params, stream);
}}
#endif
#endif
"""



@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    head_dim_v: int
    split: bool
    paged_kv: bool
    softcap: bool
    packgqa: bool
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            if self.sm == 90:
                # Always enable PackGQA for PagedKV or Split to reduce compilation
                packgqa = self.packgqa or self.paged_kv or self.split
                return KERNEL_IMPL_TEMPLATE_FWD_SM90.format(
                    ARCH=str(self.sm), DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim, HEAD_DIM_V=self.head_dim_v,
                    SPLIT=str(self.split).lower(), PAGEDKV=str(self.paged_kv).lower(),
                    SOFTCAP=str(self.softcap).lower(), PACKGQA=str(packgqa).lower()
                )
            else:
                # Always enable PackGQA for Sm8x to reduce compilation
                return KERNEL_IMPL_TEMPLATE_FWD_SM8x.format(
                    DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, HEAD_DIM_V=self.head_dim_v,
                    SPLIT=str(self.split).lower(), PAGEDKV=str(self.paged_kv).lower(),
                    SOFTCAP=str(self.softcap).lower(), PACKGQA=str(True).lower()
                )
        elif self.direction == "bwd":
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_BWD_SM90.format(
                    ARCH=str(self.sm), DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower()
                )
            else:
                return KERNEL_IMPL_TEMPLATE_BWD_SM8x.format(
                    DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower()
                )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}{f'_{self.head_dim_v}' if self.head_dim_v != self.head_dim else ''}_{self.dtype}{'_paged' if self.paged_kv else ''}{'_split' if self.split else ''}{'_softcap' if self.softcap else ''}{'_packgqa' if self.packgqa else ''}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim, split, paged_kv, softcap, packgqa, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SPLIT, PAGEDKV, SOFTCAP, PACKGQA, SM):
        # We always enable PackGQA for Sm8x or PagedKV or Split
         # so we should just pass in packgqa=False to avoid the `_packgqa` in the filename.
        if packgqa and (sm < 90 or (sm >= 90 and (paged_kv or split))):
            continue
        if sm >= 90 or dtype in DTYPE_MAP_FWD_SM8x:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=head_dim, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa, direction="fwd")
        if sm == 90 and head_dim == 192:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=128, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa, direction="fwd")
        if sm == 90 and head_dim == 64 and dtype in ["bf16", "fp16"]:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=256, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa, direction="fwd")
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=512, split=split, paged_kv=paged_kv, softcap=softcap, packgqa=packgqa, direction="fwd")
    for dtype, head_dim, softcap, sm in itertools.product(DTYPE_MAP_BWD.keys(), HEAD_DIMENSIONS, SOFTCAP, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, head_dim_v=head_dim, split=False, paged_kv=False, softcap=softcap, packgqa=False, direction="bwd")


def batch_hdim(kernels_all) -> List[KERNEL_BATCH]:
    for dtype, split, paged_kv, softcap, packgqa, sm in itertools.product(DTYPE_MAP.keys(), SPLIT, PAGEDKV, SOFTCAP, PACKGQA, SM):
        if sm < 90:
            continue
        # Same hdim and hdimv
        kernels = [k for k in kernels_all if k.direction == "fwd" and k.dtype == dtype and k.split == split and k.paged_kv == paged_kv and k.softcap == softcap and k.packgqa == packgqa and k.sm == sm and k.head_dim == k.head_dim_v]
        if len(kernels) > 0:
            filename = f"flash_fwd_hdimall_{dtype}{'_paged' if paged_kv else ''}{'_split' if split else ''}{'_softcap' if softcap else ''}{'_packgqa' if packgqa else ''}_sm{sm}.cu"
            template = "\n".join([f"#include \"{k.filename}\"" for k in kernels])
            yield KERNEL_BATCH(template, filename)
        # Different hdim and hdimv
        kernels = [k for k in kernels_all if k.direction == "fwd" and k.dtype == dtype and k.split == split and k.paged_kv == paged_kv and k.softcap == softcap and k.packgqa == packgqa and k.sm == sm and k.head_dim != k.head_dim_v]
        if len(kernels) > 0:
            filename = f"flash_fwd_hdimdiff_{dtype}{'_paged' if paged_kv else ''}{'_split' if split else ''}{'_softcap' if softcap else ''}{'_packgqa' if packgqa else ''}_sm{sm}.cu"
            template = "\n".join([f"#include \"{k.filename}\"" for k in kernels])
            yield KERNEL_BATCH(template, filename)


def batch_softcap(kernels_all) -> List[KERNEL_BATCH]:
    for dtype, head_dim, split, paged_kv, packgqa, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SPLIT, PAGEDKV, PACKGQA, SM):
        if sm >= 90:
            continue
        kernels = [k for k in kernels_all if k.direction == "fwd" and k.dtype == dtype and k.head_dim == head_dim and k.split == split and k.paged_kv == paged_kv and k.packgqa == packgqa and k.sm == sm]
        if len(kernels) > 0:
            filename = f"flash_fwd_hdim{head_dim}_{dtype}{'_paged' if paged_kv else ''}{'_split' if split else ''}_softcapall{'_packgqa' if packgqa else ''}_sm{sm}.cu"
            template = "\n".join([f"#include \"{k.filename}\"" for k in kernels])
            yield KERNEL_BATCH(template, filename)

    # Bwd
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        if sm < 90:
            continue
        kernels = [k for k in kernels_all if k.direction == "bwd" and k.dtype == dtype and k.head_dim == head_dim and k.sm == sm]
        if len(kernels) > 0:
            filename = f"flash_bwd_hdim{head_dim}_{dtype}_softcapall_sm{sm}.cu"
            template = "\n".join([f"#include \"{k.filename}\"" for k in kernels])
            yield KERNEL_BATCH(template, filename)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels_all = list(get_all_kernels())
    for kernel in kernels_all:
        write_kernel(kernel, output_dir)
    for kernel in batch_hdim(kernels_all):
        write_kernel(kernel, output_dir)
    for kernel in batch_softcap(kernels_all):
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
