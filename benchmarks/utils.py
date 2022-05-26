# Adapted from https://github.com/HazyResearch/hippo/blob/datasets/benchmark/utils.py
""" Useful functions for writing test code. """

import torch
import torch.utils.benchmark as benchmark


def benchmark_forward(fn, *inputs, min_run_time = 0.2, repeats = 10, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
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


def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Backward pass')
    y = fn(*inputs, **kwinputs)
    if type(y) is tuple:
        y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    t = benchmark.Timer(
            stmt='y.backward(grad, retain_graph=True)',
            globals={'y': y, 'grad': grad},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward + Backward pass')
    # y = fn(*inputs, **kwinputs)
    # if grad is None:
    #     grad = torch.randn_like(y)
    # else:
    #     if grad.shape != y.shape:
    #         raise RuntimeError('Grad shape does not match output shape')
    # del y
    def f(grad, *inputs, **kwinputs):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
        if grad is None:
            grad = torch.randn_like(y)
        else:
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
        y.backward(grad, retain_graph=True)
    t = benchmark.Timer(
            stmt='f(grad, *inputs, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_all(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    return (
        benchmark_forward(fn, *inputs, repeats=repeats, desc=desc, verbose=verbose, **kwinputs),
        benchmark_backward(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           **kwinputs),
        benchmark_combined(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           **kwinputs),
    )


def pytorch_profiler(fn, *inputs, repeats=10):
    """ Wrap benchmark functions in Pytorch profiler to see CUDA information. """
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            ) as p:
            # benchmark_forward(repeats, fn, *inputs)
            fn(*inputs)
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))


def convert_data(*tensors, device='cuda'):
    tensors = tuple(t.to(device) for t in tensors)
    for t in tensors:
        if t.is_leaf: t.requires_grad = True
        t.retain_grad()
    return tensors


def log_backward(output, *inputs):
    """ Perform backward pass of output and print gradients of input tensors. """

    #print(f"{output=}")
    output.sum().backward(retain_graph=True)
    print("Gradients:")
    for t in inputs:
        print(t.grad)
        t.grad.zero_()


def benchmark_memory(fn, *inputs, desc='', verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)
    if verbose:
        print(f'{desc} max memory: ', mem)
    torch.cuda.empty_cache()
    return mem
