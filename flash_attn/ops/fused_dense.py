# Copyright (c) 2022, Tri Dao.
# Inspired by https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# We make it work with pytorch amp and with bfloat16.
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.cuda.amp import custom_bwd, custom_fwd

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda

from flash_attn.ops.gelu_activation import gelu_bwd
from flash_attn.utils.distributed import all_gather_raw, reduce_scatter_raw, reduce_scatter


class FusedDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, return_residual=False, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError('fused_dense only supports matrix dims <= 2M')
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            grad_input, = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            weight, = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]),
                                         grad_output, weight)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                grad_input, handle_grad_input = reduce_scatter_raw(grad_input, process_group,
                                                                   async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None:
                handle_x.wait()
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                total_x.reshape(batch_dim, total_x.shape[-1]), grad_output, ctx.needs_input_grad[2]
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None


def fused_dense_func(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                     return_residual: bool = False, process_group: Optional[ProcessGroup] = None):
    dtype_eligible = (x.dtype in [torch.float16, torch.bfloat16]
                      or (x.dtype == torch.float32 and torch.is_autocast_enabled()))
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(x, weight, bias, return_residual, process_group)
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)


class FusedDense(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 return_residual: bool = False, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.
        """
        return fused_dense_func(x, self.weight, self.bias, return_residual=self.return_residual,
                                process_group=process_group)


class ColumnParallelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, process_group: ProcessGroup,
                 bias: bool = True, device=None, dtype=None) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f'out_features ({out_features}) must be divisible by '
                             f'world_size ({world_size})')
        super().__init__(in_features, out_features // world_size, bias=bias,
                         device=device, dtype=dtype)
        self.process_group = process_group

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do an all_gather of
        x before doing the matmul.
        """
        return fused_dense_func(x, self.weight, self.bias, process_group=self.process_group)


class RowParallelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, process_group: ProcessGroup,
                 bias: bool = True, device=None, dtype=None) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % world_size != 0:
            raise ValueError(f'in_features ({in_features}) must be divisible by '
                             f'world_size ({world_size})')
        # Only rank 0 will have bias
        super().__init__(in_features // world_size, out_features, bias=bias and rank == 0,
                         device=device, dtype=dtype)
        self.process_group = process_group

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func(x, self.weight, self.bias)
        return reduce_scatter(out, self.process_group)


class FusedDenseGeluDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, save_pre_act=True, return_residual=False,
                checkpoint_lvl=0, heuristic=0, process_group=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul.

        checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute gelu_in and gelu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        if not save_pre_act:
            checkpoint_lvl = 2
        assert checkpoint_lvl in [0, 1, 2]
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.heuristic = heuristic

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            weight1, weight2 = [a.to(dtype=dtype) for a in [weight1, weight2]]
            bias1 = bias1.to(dtype=dtype) if bias1 is not None else None
            bias2 = bias2.to(dtype=dtype) if bias2 is not None else None
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous() if bias1 is not None else None
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous() if bias2 is not None else None
        if process_group is not None:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight1.shape, *weight2.shape) > 65535 * 32:
            raise RuntimeError('fused_dense only supports matrix dims <= 2M')
        if heuristic == -1:
            gelu_in = F.linear(total_x, weight1, bias1)
            output1 = F.gelu(gelu_in, approximate='tanh')
            # This is before adding bias1
            # gelu_in = F.linear(total_x.reshape(batch_dim, n), weight1)
            # with torch.jit.fuser('fuser2'):
            #     output1 = bias_gelu(gelu_in, bias1)
        else:
            output1, *rest = fused_dense_cuda.linear_gelu_forward(
                total_x.reshape(batch_dim, n), weight1, bias1, save_pre_act, heuristic
            )
            if save_pre_act:
                gelu_in = rest[0]
        output2 = F.linear(output1, weight2, bias2)
        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, weight2, gelu_in, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, weight2, gelu_in)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        output2 = output2.reshape(*batch_shape, output2.shape[-1])
        return output2 if not return_residual else (output2, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        if ctx.return_residual:
            grad_input, = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        x, weight1, weight2, *rest = ctx.saved_tensors
        if process_group is None:
            total_x = x
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl in [0, 1]:
            if process_group is not None:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            if checkpoint_lvl == 0:
                gelu_in, output1 = rest
            elif checkpoint_lvl == 1:
                gelu_in, = rest
                output1 = F.gelu(gelu_in, approximate='tanh')
        elif checkpoint_lvl == 2:
            bias1, = rest
            if process_group is not None:
                total_x, _ = all_gather_raw(x, process_group)
            if ctx.heuristic == -1:
                gelu_in = F.linear(total_x, weight1, bias1)
                output1 = F.gelu(gelu_in, approximate='tanh')
            else:
                output1, gelu_in = fused_dense_cuda.linear_gelu_forward(
                    total_x.reshape(batch_dim, total_x.shape[-1]), weight1, bias1, True,
                    ctx.heuristic
                )

        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        output1 = output1.reshape(batch_dim, output1.shape[-1])
        gelu_in = gelu_in.reshape(batch_dim, gelu_in.shape[-1])
        if ctx.needs_input_grad[3]:
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(
                output1, grad_output, ctx.needs_input_grad[4]
            )
        else:
            grad_weight2 = None
            grad_bias2 = grad_output if ctx.needs_input_grad[4] else None
        if ctx.heuristic == -1:
            # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
            grad_output1 = F.linear(grad_output, weight2.t())
            with torch.jit.fuser('fuser2'):
                grad_gelu = gelu_bwd(grad_output1, gelu_in)
        else:
            # The cublasLt epilogue has to compute both gelu grad and bias grad, we can't
            # just compute gelu grad
            grad_gelu, grad_bias1 = fused_dense_cuda.bias_gelu_linear_dgrad_bgrad(
                weight2, grad_output, gelu_in, ctx.heuristic
            )
            if not ctx.needs_input_grad[2]:
                grad_bias1 = None
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_gelu, weight1.t())
            else:
                grad_input = torch.addmm(grad_input.reshape(batch_dim, grad_input.shape[-1]),
                                         grad_gelu, weight1)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                grad_input, handle_grad_input = reduce_scatter_raw(grad_input, process_group,
                                                                   async_op=True)
        else:
            grad_input = None
        if ctx.heuristic == -1:
            if ctx.needs_input_grad[1]:
                if process_group is not None:
                    handle_x.wait()
                grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_wgrad(
                    total_x.reshape(batch_dim, total_x.shape[-1]), grad_gelu,
                    ctx.needs_input_grad[2]
                )
            else:
                grad_weight1 = None
                grad_bias1 = grad_gelu if ctx.needs_input_grad[2] else None
        else:
            if ctx.needs_input_grad[1]:
                if process_group is not None:
                    handle_x.wait()
                grad_weight1 = F.linear(grad_gelu.t(),
                                        total_x.reshape(batch_dim, total_x.shape[-1]).t())
            else:
                grad_weight1 = None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return (grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2,
                None, None, None, None, None)


def fused_dense_gelu_dense_func(
    x: Tensor, weight1: Tensor, weight2: Tensor, bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    save_pre_act: bool = True, return_residual: bool = False,
    checkpoint_lvl: int = 0, heuristic: int = 0,
    process_group: Optional[ProcessGroup] = None
):
    dtype_eligible = (x.dtype in [torch.float16, torch.bfloat16]
                      or (x.dtype == torch.float32 and torch.is_autocast_enabled()))
    if (x.is_cuda and weight1.is_cuda and weight2.is_cuda and (bias1 is None or bias1.is_cuda)
        and (bias2 is None or bias2.is_cuda) and dtype_eligible):
        return FusedDenseGeluDenseFunc.apply(
            x, weight1, bias1, weight2, bias2,
            save_pre_act, return_residual, checkpoint_lvl, heuristic, process_group
        )
    else:
        assert process_group is None
        gelu_in = F.linear(x, weight1, bias1)
        output1 = F.gelu(gelu_in, approximate='tanh')
        output2 = F.linear(output1, weight2, bias2)
        return output2 if not return_residual else (output2, x)


class FusedDenseGeluDense(nn.Module):

    def __init__(self, in_features, hidden_features, out_features=None, bias1=True,
                 bias2=True, return_residual=False, checkpoint_lvl=0, heuristic=0,
                 device=None, dtype=None):
        """
        If process_group is not None, we're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            For CUDA >= 11.8, you'd want heuristic=0 for both fp16 and bf16 for best perf.
            For CUDA <= 11.7, you'd want heuristic=1 for fp16 and heuristic=-1 for bf16.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.return_residual = return_residual
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x, process_group=None):
        out = fused_dense_gelu_dense_func(
            x, self.fc1.weight, self.fc2.weight, self.fc1.bias, self.fc2.bias,
            save_pre_act=self.training, return_residual=self.return_residual,
            checkpoint_lvl=self.checkpoint_lvl, heuristic=self.heuristic,
            process_group=process_group
        )
        if self.return_residual:
            out, x = out
        if process_group is not None:
            out = reduce_scatter(out, process_group)
        return out if not self.return_residual else (out, x)


class ParallelFusedDenseGeluDense(nn.Module):

    def __init__(self, in_features, hidden_features, out_features=None,
                 process_group: ProcessGroup = None, bias1=True, bias2=True,
                 checkpoint_lvl=0, heuristic=0, device=None, dtype=None):
        """
        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            For CUDA >= 11.8, you'd want heuristic=0 for both fp16 and bf16 for best perf.
            For CUDA <= 11.7, you'd want heuristic=1 for fp16 and heuristic=-1 for bf16.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert process_group is not None
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.process_group = process_group
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, process_group,
                                        bias=bias1, **factory_kwargs)
        self.fc2 = RowParallelLinear(hidden_features, out_features, process_group,
                                     bias=bias2, **factory_kwargs)

    def forward(self, x):
        out = fused_dense_gelu_dense_func(
            x, self.fc1.weight, self.fc2.weight, self.fc1.bias, self.fc2.bias,
            save_pre_act=self.training, checkpoint_lvl=self.checkpoint_lvl,
            heuristic=self.heuristic, process_group=self.process_group
        )
        return reduce_scatter(out, self.process_group)
