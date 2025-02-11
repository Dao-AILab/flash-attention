import time
import torch
import torch.utils.benchmark as benchmark

from triton.testing import do_bench

if torch.version.cuda:
    backendBLAS = "cuBLAS"
elif torch.version.hip:
    backendBLAS = "hipBLAS"

def benchmark_forward(fn, *inputs, repeats=10, desc='', verbose=True, **kwinputs):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


torch.manual_seed(0)
repeats = 30
dtype = torch.bfloat16
device = 'cuda'
verbose = False
m, n = 8192, 8192

tflops_matmul = {}
tflops_matmul1 = {}
for k in [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]:
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(n, k, device=device, dtype=dtype).transpose(-1, -2)
    nFLOPS_matmul = 2 * m * n * k
    time.sleep(2)  # to reduce power throttling
    timing = benchmark_forward(torch.matmul, a, b, desc=backendBLAS, verbose=verbose, repeats=repeats)[1]
    tflops_matmul[k] = nFLOPS_matmul / timing.mean * 1e-12
    print(f'[torch.utils.benchmark] {backendBLAS}, {m = }, {n = }, {k = }: {timing.mean * 1e3:.3f}ms, {tflops_matmul[k]:.1f} TFLOPS')
    time.sleep(2)  # to reduce power throttling
    ms = do_bench(lambda: torch.matmul(a, b), warmup=10, rep=repeats)
    tflops_matmul1[k] = nFLOPS_matmul / ms * 1e-9
    print(f'[triton.test.do_bench]  {backendBLAS}, {m = }, {n = }, {k = }: {ms:.3f}ms, {tflops_matmul1[k]:.1f} TFLOPS')
