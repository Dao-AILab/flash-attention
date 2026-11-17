"""Standalone kernel that runs the production P quant-dequant codecs on given FP32 inputs.

The forward kernel only feeds the codecs values that came through exp2, so exact ties,
subnormals and saturation cannot be placed there. This kernel takes the codec inputs
directly: each thread loads one 32-value fragment, applies the same helper the softmax
uses (``flash_attn.cute.softmax._P_QUANT_CODECS``) per block, and stores the FP32 result.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32

from flash_attn.cute import softmax
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.p_quant import resolve_p_quant
from flash_attn.cute.testing import is_fake_mode

FRAGMENT = 32  # values per thread, as in SoftmaxSm100.apply_exp2_convert


class PQuantCodecProbe:
    def __init__(self, fmt, num_threads: int = 128):
        self.fmt = fmt
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mOut: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        num_fragments = mIn.shape[0] // FRAGMENT
        self.kernel(mIn, mOut).launch(
            grid=[(num_fragments + self.num_threads - 1) // self.num_threads, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor):
        frag = cute.arch.block_idx()[0] * self.num_threads + cute.arch.thread_idx()[0]
        if frag < mIn.shape[0] // FRAGMENT:
            x = cute.make_rmem_tensor(FRAGMENT, Float32)
            y = cute.make_rmem_tensor(FRAGMENT, Float32)
            for i in cutlass.range_constexpr(FRAGMENT):
                x[i] = mIn[frag * FRAGMENT + i]
            codec = softmax._P_QUANT_CODECS[self.fmt.codec]
            blk_tile = FRAGMENT if self.fmt.block == 1 else self.fmt.block
            x_blk = cute.logical_divide(x, cute.make_layout(blk_tile))
            y_blk = cute.logical_divide(y, cute.make_layout(blk_tile))
            for h in cutlass.range_constexpr(FRAGMENT // blk_tile):
                y_blk[None, h].store(codec(x_blk[None, h].load()))
            for i in cutlass.range_constexpr(FRAGMENT):
                mOut[frag * FRAGMENT + i] = y[i]


def run_codec_probe(x: torch.Tensor, p_quant) -> torch.Tensor:
    """Apply the device codec of ``p_quant`` to FP32 codec inputs ``x`` (1-D, CUDA).

    The length is padded with zeros to a multiple of 32. Under FakeTensorMode the kernel
    is compiled but not launched.
    """
    fmt = resolve_p_quant(p_quant)
    assert fmt is not None and fmt.codec != "identity", (
        "the probe needs a rounding codec"
    )
    assert x.dtype == torch.float32 and x.dim() == 1
    n = x.shape[0]
    padded = torch.zeros(
        -(-n // FRAGMENT) * FRAGMENT, dtype=torch.float32, device=x.device
    )
    if not is_fake_mode():
        padded[:n] = x
    out = torch.empty_like(padded)
    key = (fmt,)
    cache = run_codec_probe.compile_cache
    if key not in cache:
        x_t, out_t = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) for t in (padded, out)
        ]
        cache[key] = cute.compile(
            PQuantCodecProbe(fmt),
            x_t,
            out_t,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        cache[key](padded, out)
    return out[:n]


run_codec_probe.compile_cache = get_jit_cache("p_quant_codec_probe")
