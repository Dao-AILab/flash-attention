# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/benchmark.py
import torch

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
    from fvcore.nn import ActivationCountAnalysis
    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    ActivationCountAnalysis = None
    has_fvcore_profiling = False


def profile_deepspeed(model, input_size=(3, 224, 224), input_dtype=torch.float32,
                      batch_size=1, detailed=False):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    flops, macs, params = get_model_profile(
        model=model,
        args=torch.zeros((batch_size,) + input_size, device=device, dtype=input_dtype),
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), input_dtype=torch.float32, max_depth=4,
                   batch_size=1, detailed=False, force_cpu=False):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.zeros((batch_size,) + input_size, device=device, dtype=input_dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        print(flop_count_table(fca, max_depth=max_depth))
    return fca, fca.total(), aca, aca.total()
