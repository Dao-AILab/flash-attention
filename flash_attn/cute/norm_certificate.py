import math

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass.utils import SmemAllocator

from flash_attn.cute import utils
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


_COMPILED = get_jit_cache("packed_softmax_bounds")


@cute.jit
def _block_max_pair(q, k, scratch, threads: cutlass.Constexpr):
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    warp = tid // 32
    q = utils.warp_reduction_max(q)
    k = utils.warp_reduction_max(k)
    if lane == 0:
        scratch[warp, 0] = q
        scratch[warp, 1] = k
    cute.arch.sync_threads()
    qout = Float32(0.0)
    kout = Float32(0.0)
    if warp == 0:
        if lane < threads // 32:
            qout = scratch[lane, 0]
            kout = scratch[lane, 1]
        qout = utils.warp_reduction_max(qout)
        kout = utils.warp_reduction_max(kout)
    return qout, kout


class _PartialNorms:
    threads = 256
    rows_per_block = 128

    def __init__(self, chunks):
        self.chunks = chunks

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, partial: cute.Tensor,
                 stream: cuda.CUstream = None):
        self.kernel(q, k, v, partial).launch(
            grid=(partial.shape[0], 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, partial: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        lane = tid % 16
        group = tid // 16
        scratch = SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((self.threads // 32, 2)), byte_alignment=16
        )
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        rq = cute.make_rmem_tensor((8,), BFloat16)
        rk = cute.make_rmem_tensor((8,), BFloat16)
        rv = cute.make_rmem_tensor((8,), BFloat16)
        qmax = Float32(0.0)
        kmax = Float32(0.0)
        for i in cutlass.range_constexpr(self.rows_per_block // (self.threads // 16)):
            row = block * self.rows_per_block + group + i * (self.threads // 16)
            qs = Float32(0.0)
            ks = Float32(0.0)
            for part in cutlass.range_constexpr(self.chunks):
                column = lane + part * 16
                rq.fill(0)
                rk.fill(0)
                rv.fill(0)
                if row < q.shape[0] and column * 8 < q.shape[1]:
                    cute.copy(atom, cute.local_tile(q[row, None], (8,), (column,)), rq)
                if row < k.shape[0] and column * 8 < k.shape[1]:
                    cute.copy(atom, cute.local_tile(k[row, None], (8,), (column,)), rk)
                if row < v.shape[0] and column * 8 < v.shape[1]:
                    cute.copy(atom, cute.local_tile(v[row, None], (8,), (column,)), rv)
                qv = rq.load().to(Float32)
                kv = rk.load().to(Float32)
                vv = rv.load().to(Float32)
                qs += (qv * qv).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                ks += (kv * kv).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                vs = (vv * vv).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                qs = qs if vs <= Float32(2.0 ** 32) else Float32.inf
            qs = utils.warp_reduction_sum(qs, threads_in_group=16)
            ks = utils.warp_reduction_sum(ks, threads_in_group=16)
            qs = qs if qs >= Float32(0.0) and qs < Float32.inf else Float32.inf
            ks = ks if ks >= Float32(0.0) and ks < Float32.inf else Float32.inf
            qmax = utils.fmax(qmax, qs)
            kmax = utils.fmax(kmax, ks)
        qmax, kmax = _block_max_pair(qmax, kmax, scratch, self.threads)
        if tid == 0:
            partial[block, 0] = qmax
            partial[block, 1] = kmax


class _FinishNorms:
    threads = 256

    @cute.jit
    def __call__(self, partial: cute.Tensor, flag: cute.Tensor, scale_log2: Float32,
                 stream: cuda.CUstream = None):
        self.kernel(partial, flag, scale_log2).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, partial: cute.Tensor, flag: cute.Tensor, scale_log2: Float32):
        tid = cute.arch.thread_idx()[0]
        scratch = SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((self.threads // 32, 2)), byte_alignment=16
        )
        qmax = Float32(0.0)
        kmax = Float32(0.0)
        index = tid
        while index < partial.shape[0]:
            qmax = utils.fmax(qmax, partial[index, 0])
            kmax = utils.fmax(kmax, partial[index, 1])
            index += self.threads
        qmax, kmax = _block_max_pair(qmax, kmax, scratch, self.threads)
        if tid == 0:
            qs = cute.math.fma(qmax, Float32(1.0001), Float32(2.0 ** -117))
            ks = cute.math.fma(kmax, Float32(1.0001), Float32(2.0 ** -117))
            bound = Float32(1.01) * (cute.math.sqrt(qs) * cute.math.sqrt(ks))
            flag[0] = Int32(
                bound < Float32.inf
                and scale_log2 >= Float32(2.0 ** -16)
                and scale_log2 <= Float32(2.0 ** 16)
                and scale_log2 * bound <= Float32(60.0)
            )


def _launch(kernel, name, *arguments):
    metadata = tuple((arg.dtype, tuple(arg.shape), tuple(arg.stride()))
                     if isinstance(arg, torch.Tensor) else type(arg) for arg in arguments)
    tensor = next(arg for arg in arguments if isinstance(arg, torch.Tensor))
    properties = torch.cuda.get_device_properties(tensor.device)
    key = (name, properties.major, properties.minor, metadata)
    if key not in _COMPILED:
        dtypes = {torch.bfloat16: BFloat16, torch.float32: Float32, torch.int32: Int32}
        fake = [cute.runtime.make_fake_tensor(dtypes[arg.dtype], tuple(arg.shape),
                    tuple(arg.stride()), assumed_align=16)
                if isinstance(arg, torch.Tensor) else Float32(arg) for arg in arguments]
        _COMPILED[key] = cute.compile(kernel, *fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi')
    if not is_fake_mode():
        _COMPILED[key](*arguments)


def attention_norm_certificate(q, k, v, softmax_scale):
    a = float(softmax_scale) * math.log2(math.e)
    flag = torch.empty((1,), device=q.device, dtype=torch.int32)
    eligible = math.isfinite(a) and 2.0 ** -16 <= a <= 2.0 ** 16
    for tensor in (q, k, v):
        eligible = eligible and tensor.ndim == 4 and tensor.dtype == torch.bfloat16 \
            and 0 < tensor.shape[-1] <= 256 and tensor.shape[-1] % 8 == 0 and tensor.numel() <= 2 ** 31 - 1 \
            and (tensor.is_contiguous() or tensor.transpose(1, 2).is_contiguous())
        if not is_fake_mode():
            eligible = eligible and tensor.data_ptr() % 16 == 0
    if not eligible or q.device != k.device or q.device != v.device:
        flag.zero_()
        return flag
    qrows, krows, vrows = [
        tensor.as_strided((tensor.numel() // tensor.shape[-1], tensor.shape[-1]),
                         (tensor.shape[-1], 1)) for tensor in (q, k, v)
    ]
    rows = max(tensor.shape[0] for tensor in (qrows, krows, vrows))
    partial = torch.empty(((rows + 127) // 128, 2), device=q.device, dtype=torch.float32)
    _launch(_PartialNorms((max(q.shape[-1], k.shape[-1], v.shape[-1]) + 127) // 128),
            'partial', qrows, krows, vrows, partial)
    _launch(_FinishNorms(), 'finish', partial, flag, a)
    return flag
