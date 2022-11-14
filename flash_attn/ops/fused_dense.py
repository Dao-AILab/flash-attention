# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# We make it work with pytorch amp and with bfloat16.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

# import fused_dense_cuda  # from apex
import fused_dense_lib as fused_dense_cuda
from flash_attn.ops.gelu_activation import gelu_bwd


# implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedDenseFuncTD(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias):
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight, bias = [a.to(dtype=dtype) for a in [x, weight, bias]]
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        assert batch_dim <= 64 * 1024, 'fused_dense only supports dimension at most 64k'
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        if ctx.needs_input_grad[0]:
            grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_backward(
                x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1])
            )
            grad_input = grad_input.reshape_as(x)
        else:
            grad_weight, grad_bias = fused_dense_cuda.linear_bias_wgrad(
                x.reshape(batch_dim, n), grad_output.reshape(batch_dim, grad_output.shape[-1])
            )
            grad_input = None
        # print((grad_bias - grad_output.view(-1, grad_output.shape[-1]).sum(dim=0)).abs().max())
        return grad_input, grad_weight, grad_bias
        # grad_input, grad_weight = None, None
        # grad_output_reshaped = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # if ctx.needs_input_grad[0]:
        #     grad_input = (grad_output_reshaped @ weight.conj()).reshape(*batch_shape, n)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output_reshaped.t() @ x.conj().reshape(batch_dim, n)
        # # We don't need to compute grad_bias explicitly, when we return grad_out Pytorch
        # # will sum over the batch dimension to get grad_bias.
        # return grad_input, grad_weight, grad_output


fused_dense_function_td = FusedDenseFuncTD.apply


class FusedDenseTD(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_function_td(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class FusedDenseResidualFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias):
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight, bias = [a.to(dtype=dtype) for a in [x, weight, bias]]
        x = x.contiguous()
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        assert batch_dim <= 64 * 1024, 'fused_dense only supports dimension at most 64k'
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1]), x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_input):
        grad_output = grad_output.contiguous()
        grad_input = grad_input.contiguous()
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_residual_backward(
            x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1]),
            grad_input.reshape(batch_dim, n)
        )
        return grad_input.reshape_as(x), grad_weight, grad_bias


fused_dense_residual_function = FusedDenseResidualFunc.apply


class FusedDenseResidual(nn.Linear):
    """Similar to FusedDense, but we return both the output and the input.
    This is so that in the backward pass, we can combine the input gradient from the residual branch
    with the input gradient from the matrix multiply, without having to do a separate addition.
    """

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_residual_function(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias), x


class FusedDenseGeluDenseFuncTD(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_lvl=0, heuristic=0):
        """checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute gelu_in and gelu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight1, bias1, weight2, bias2 = [a.to(dtype=dtype)
                                                 for a in [x, weight1, bias1, weight2, bias2]]
        assert checkpoint_lvl in [0, 1, 2]
        x = x.contiguous()
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous()
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        assert batch_dim <= 64 * 1024, 'fused_dense only supports dimension at most 64k'
        # output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(
        #     x.reshape(batch_dim, n), weight1, bias1, weight2, bias2
        # )
        if heuristic == -1:
            gelu_in = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight1, bias1)
            output1 = F.gelu(gelu_in, approximate='tanh')
            # gelu_in = F.linear(x.reshape(batch_dim, n), weight1)  # This is before adding bias1
            # with torch.jit.fuser('fuser2'):
            #     output1 = bias_gelu(gelu_in, bias1)
        else:
            save_gelu_in = checkpoint_lvl != 2
            output1, *rest = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n), weight1,
                                                                  bias1, save_gelu_in, heuristic)
            if save_gelu_in:
                gelu_in = rest[0]
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.heuristic = heuristic
        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, bias1, weight2, gelu_in, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, bias1, weight2, gelu_in)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, bias1, weight2)
        return output2.reshape(*batch_shape, output2.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        x, weight1, bias1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl == 0:
            gelu_in, output1 = rest
        elif checkpoint_lvl == 1:
            gelu_in, = rest
            output1 = F.gelu(gelu_in, approximate='tanh')
        elif checkpoint_lvl == 2:
            # bias1, = rest
            if ctx.heuristic == -1:
                gelu_in = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight1, bias1)
                output1 = F.gelu(gelu_in, approximate='tanh')
            else:
                output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n),
                                                                        weight1, bias1, True, ctx.heuristic)

        if ctx.heuristic == -1:
            grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
            # grad_output1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_backward(output1, weight2, grad_output)
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
            # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
            grad_output1 = grad_output @ weight2
            with torch.jit.fuser('fuser2'):
                grad_gelu = gelu_bwd(grad_output1, gelu_in)
            grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
                x.reshape(batch_dim, n), weight1, grad_gelu
            )
            # with torch.jit.fuser('fuser2'):
            #     grad_gelu, grad_bias1 = bias_gelu_back(grad_output1, gelu_in, bias1)
            # grad_input = grad_gelu @ weight1
            # grad_weight1 = grad_gelu.reshape(batch_dim, -1).T @ x.reshape(batch_dim, n)
            # grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
            #     x.reshape(batch_dim, n), weight1, grad_gelu
            # )
        else:
            grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_gelu_linear_backward(
                x.reshape(batch_dim, n), gelu_in, output1, weight1, weight2,
                grad_output.reshape(batch_dim, grad_output.shape[-1]),
                ctx.heuristic
            )
        # grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # # grad_output1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_backward(output1, weight2, grad_output)
        # grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
        # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
        # grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_backward(
        #     x.reshape(batch_dim, n), weight1, grad_gelu
        # )
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None


fused_dense_gelu_dense_function_td = FusedDenseGeluDenseFuncTD.apply


class FusedDenseGeluDenseTD(nn.Module):

    def __init__(self, in_features, intermediate_features, out_features=None, bias=True,
                 checkpoint_lvl=0, heuristic=0, device=None, dtype=None):
        """
        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if out_features is None:
            out_features = in_features
        assert bias == True, "DenseGeluDense module without bias is currently not supported"
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic
        self.fc1 = nn.Linear(in_features, intermediate_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(intermediate_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        return fused_dense_gelu_dense_function_td(x, self.fc1.weight, self.fc1.bias,
                                                  self.fc2.weight, self.fc2.bias,
                                                  self.checkpoint_lvl, self.heuristic)


class FusedDenseResGeluDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_lvl=0, heuristic=0):
        """checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute gelu_in and gelu_out in the bwd
        """
        assert -1 <= heuristic <= 4
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight1, bias1, weight2, bias2 = [a.to(dtype=dtype)
                                                 for a in [x, weight1, bias1, weight2, bias2]]
        assert checkpoint_lvl in [0, 1, 2]
        x = x.contiguous()
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous()
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        assert batch_dim <= 64 * 1024, 'fused_dense only supports dimension at most 64k'
        # output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(
        #     x.reshape(batch_dim, n), weight1, bias1, weight2, bias2
        # )
        # gelu_in = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight1, bias1)
        # output1 = F.gelu(gelu_in, approximate='tanh')
        save_gelu_in = checkpoint_lvl != 2
        output1, *rest = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n), weight1,
                                                              bias1, save_gelu_in, heuristic)
        if save_gelu_in:
            gelu_in = rest[0]
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.heuristic = heuristic
        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, weight2, gelu_in, output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, weight2, gelu_in)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, weight2, bias1)
        return output2.reshape(*batch_shape, output2.shape[-1]), x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_input):
        grad_output = grad_output.contiguous()
        grad_input = grad_input.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        x, weight1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        if checkpoint_lvl == 0:
            gelu_in, output1 = rest
        elif checkpoint_lvl == 1:
            gelu_in, = rest
            output1 = F.gelu(gelu_in, approximate='tanh')
        elif checkpoint_lvl == 2:
            bias1, = rest
            output1, gelu_in = fused_dense_cuda.linear_gelu_forward(x.reshape(batch_dim, n),
                                                                    weight1, bias1, True, ctx.heuristic)
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_residual_gelu_linear_backward(
            x.reshape(batch_dim, n), gelu_in, output1, weight1, weight2,
            grad_output.reshape(batch_dim, grad_output.shape[-1]),
            grad_input.reshape(batch_dim, n),
            ctx.heuristic
        )
        # grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        # # grad_output1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_backward(output1, weight2, grad_output)
        # grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1, grad_output)
        # grad_gelu = matmul_dgelu(grad_output, weight2, gelu_in)
        # grad_input, grad_weight1, grad_bias1 = fused_dense_cuda.linear_bias_residual_backward(
        #     x.reshape(batch_dim, n), weight1, grad_gelu,
        #     grad_input.reshape(batch_dim, n)
        # )
        return grad_input.reshape_as(x), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None, None


fused_dense_res_gelu_dense_function_td = FusedDenseResGeluDenseFunc.apply


class FusedDenseResGeluDense(FusedDenseGeluDenseTD):

    def forward(self, x):
        return fused_dense_res_gelu_dense_function_td(x, self.fc1.weight, self.fc1.bias,
                                                      self.fc2.weight, self.fc2.bias,
                                                      self.checkpoint_lvl, False, self.heuristic)
