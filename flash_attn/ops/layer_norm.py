# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py
import torch
from torch.nn import init

# from apex._autocast_utils import _cast_if_autocast_enabled
import dropout_layer_norm


def _dropout_add_layer_norm_forward(x0, x1, gamma, beta, rowscale, dropout_p, epsilon,
                                    residual_in_fp32):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size)) if x1 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat, x1mat, gamma, beta, rowscale, dropout_p, epsilon, None, residual_in_fp32
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and x1 is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(dz, x, dmask, mu, rsigma, gamma, rowscale, dropout_p,
                                     has_residual):
    """ Assume that arguments are contiguous
    """
    # dmask is None if dropout_p == 0.0
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    rowscale = rowscale.view(-1) if rowscale is not None else None
    dx0mat, dx1mat, dgamma, dbeta, _, _ = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat, xmat, dmask, mu, rsigma, gamma, rowscale, dropout_p, has_residual
    )
    # dx1mat is None if not has_residual
    return dx0mat, dx1mat, dgamma, dbeta


def _dropout_add_layer_norm_prenorm_backward(dz, dx, x, dmask, mu, rsigma, gamma, rowscale,
                                             dropout_p, has_residual):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape)
    rowscale = rowscale.view(-1) if rowscale is not None else None
    dx0mat, dx1mat, dgamma, dbeta, _, _ = dropout_layer_norm.dropout_add_ln_prenorm_bwd(
        dzmat, dxmat, xmat, dmask, mu, rsigma, gamma, rowscale, dropout_p, has_residual
    )
    return dx0mat, dx1mat, dgamma, dbeta


class DropoutAddLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, rowscale, dropout_p, epsilon, residual_in_fp32,
                return_dmask=False):
        x0 = x0.contiguous()
        x1 = x1.contiguous() if x1 is not None else None
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        rowscale = rowscale.contiguous() if rowscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0, x1, gamma, beta, rowscale, dropout_p, epsilon, residual_in_fp32
        )
        ctx.save_for_backward(xmat.view(x0.shape), dmask, gamma, mu, rsigma, rowscale)
        ctx.dropout_p = dropout_p
        ctx.has_residual = x1 is not None
        if not return_dmask:
            return zmat.view(x0.shape)
        else:
            dmask = (dmask.view(x0.shape) if dropout_p > 0.
                     else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device))
            ctx.mark_non_differentiable(dmask)
            return zmat.view(x0.shape), dmask

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        x, dmask, gamma, mu, rsigma, rowscale = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dx1mat, dgamma, dbeta = _dropout_add_layer_norm_backward(
            dz, x, dmask, mu, rsigma, gamma, rowscale, dropout_p, has_residual
        )
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        return dx0, dx1, dgamma, dbeta, None, None, None, None, None


class DropoutAddLayerNormPrenormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, rowscale, dropout_p, epsilon, residual_in_fp32,
                return_dmask=False):
        x0 = x0.contiguous()
        x1 = x1.contiguous() if x1 is not None else None
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        rowscale = rowscale.contiguous() if rowscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0, x1, gamma, beta, rowscale, dropout_p, epsilon, residual_in_fp32
        )
        ctx.save_for_backward(xmat.view(x0.shape), dmask, gamma, mu, rsigma, rowscale)
        ctx.dropout_p = dropout_p
        ctx.has_residual = x1 is not None
        if not return_dmask:
            return zmat.view(x0.shape), xmat.view(x0.shape)
        else:
            dmask = (dmask.view(x0.shape) if dropout_p > 0.
                     else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device))
            ctx.mark_non_differentiable(dmask)
            return zmat.view(x0.shape), xmat.view(x0.shape), dmask

    @staticmethod
    def backward(ctx, dz, dx, *args):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        dx = dx.contiguous()  # this happens!
        x, dmask, gamma, mu, rsigma, rowscale = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dx1mat, dgamma, dbeta = _dropout_add_layer_norm_prenorm_backward(
            dz, dx, x, dmask, mu, rsigma, gamma, rowscale, dropout_p, has_residual
        )
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        return dx0, dx1, dgamma, dbeta, None, None, None, None, None


def dropout_add_layer_norm(x0, x1, weight, bias, dropout_p, epsilon, rowscale=None,
                           prenorm=False, residual_in_fp32=False,
                           return_dropout_mask=False):
    """residual_in_fp32 only has an effect if x1 is None.
    Otherwise residual dtype is x1.dtype.
    """
    args = (x0, x1, weight, bias, rowscale, dropout_p, epsilon, residual_in_fp32,
            return_dropout_mask)
    if not prenorm:
        return DropoutAddLayerNormFN.apply(*args)
    else:
        return DropoutAddLayerNormPrenormFN.apply(*args)


class DropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, prenorm=False, p=0.5, eps=1e-5, residual_in_fp32=False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.epsilon = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, x1=None):
        return dropout_add_layer_norm(x0, x1, self.weight, self.bias,
                                      self.p if self.training else 0.0, self.epsilon,
                                      prenorm=self.prenorm, residual_in_fp32=self.residual_in_fp32)
