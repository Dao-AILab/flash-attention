# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py

import torch
from torch.nn import init

import dropout_layer_norm


def _dropout_add_layer_norm_forward(x0, x1, gamma, beta, rowscale, colscale, dropout_p, epsilon,
                                    residual_in_fp32):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size)) if x1 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat, x1mat, gamma, beta, rowscale, colscale, None, None, dropout_p, epsilon,
        1.0, 0, None, residual_in_fp32
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and x1 is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale,
                                     dropout_p, has_residual):
    """ Assume that arguments are contiguous
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + x1 was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, 'x0 is required to compute the gradient of colscale'
    dx0mat, dx1mat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat, dxmat, xmat, x0mat, dmask, mu, rsigma, gamma, rowscale, colscale, None, None,
        dropout_p, 1.0, 0, has_residual
    )
    # dx1mat is None if not has_residual
    if colscale is None:
        return dx0mat, dx1mat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dx1mat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_subset_forward(x0, x1, gamma, beta, colscale, x0_subset, out_subset,
                                           dropout_p, epsilon, rowscale_const, out_numrows,
                                           residual_in_fp32):
    """ Assume that arguments are contiguous
    """
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size)) if x1 is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat, x1mat, gamma, beta, None, colscale, x0_subset, out_subset, dropout_p, epsilon,
        rowscale_const, out_numrows, None, residual_in_fp32
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and x1 is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_subset_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, colscale,
                                            x0_subset, out_subset, dropout_p, rowscale_const,
                                            x0_numrows, has_residual):
    """ Assume that arguments are contiguous
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + x1 was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(-1, hidden_size)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    if colscale is not None:
        assert x0 is not None, 'x0 is required to compute the gradient of colscale'
    dx0mat, dx1mat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat, dxmat, xmat, x0mat, dmask, mu, rsigma, gamma, None, colscale, x0_subset, out_subset,
        dropout_p, rowscale_const, x0_numrows, has_residual
    )
    # dx1mat is None if not has_residual
    if colscale is None:
        return dx0mat, dx1mat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dx1mat, dgamma, dbeta, dcolscale


class DropoutAddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32,
                prenorm=False, return_dmask=False):
        x0 = x0.contiguous()
        x1 = x1.contiguous() if x1 is not None else None
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        rowscale = rowscale.contiguous() if rowscale is not None else None
        colscale = colscale.contiguous() if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0, x1, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32
        )
        # Only need to save x0 if we need to compute gradient wrt colscale
        x0_saved = x0 if colscale is not None else None
        ctx.save_for_backward(xmat.view(x0.shape), x0, dmask, gamma, mu, rsigma, rowscale, colscale)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_residual = x1 is not None
        if not return_dmask:
            return (zmat.view(x0.shape) if not prenorm
                    else (zmat.view(x0.shape), xmat.view(x0.shape)))
        else:
            dmask = (dmask.view(x0.shape) if dropout_p > 0.
                     else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device))
            ctx.mark_non_differentiable(dmask)
            return ((zmat.view(x0.shape), dmask) if not prenorm
                    else (zmat.view(x0.shape), xmat.view(x0.shape), dmask))

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        dx = args[0].contiguous() if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
        # x0 is None if colscale is None
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dx1mat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(
            dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale, dropout_p, has_residual
        )
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return dx0, dx1, dgamma, dbeta, None, dcolscale, None, None, None, None, None


class DropoutAddLayerNormSubsetFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, gamma, beta, colscale, x0_subset, out_subset, dropout_p, epsilon,
                rowscale_const, out_numrows, residual_in_fp32, prenorm=False, return_dmask=False):
        x0 = x0.contiguous()
        x1 = x1.contiguous() if x1 is not None else None
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        colscale = colscale.contiguous() if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_subset_forward(
            x0, x1, gamma, beta, colscale, x0_subset, out_subset, dropout_p, epsilon,
            rowscale_const, out_numrows, residual_in_fp32
        )
        # Only need to save x0 if we need to compute gradient wrt colscale
        x0_saved = x0 if colscale is not None else None
        x_shape = (-1, *x0.shape[1:])
        ctx.save_for_backward(xmat.view(x_shape), x0, dmask, gamma, mu, rsigma, colscale,
                              x0_subset, out_subset)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.rowscale_const = rowscale_const
        ctx.x0_numrows = x0.shape[:-1].numel()
        ctx.has_residual = x1 is not None
        z_shape = (-1, *x0.shape[1:])
        if not return_dmask:
            return (zmat.view(z_shape) if not prenorm
                    else (zmat.view(z_shape), xmat.view(x0.shape)))
        else:
            z = zmat.view(z_shape)
            dmask = (dmask.view(x0.shape) if dropout_p > 0.
                     else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device))
            ctx.mark_non_differentiable(dmask)
            return ((z, dmask) if not prenorm else (z, xmat.view(x_shape), dmask))

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = dz.contiguous()  # this happens!
        dx = args[0].contiguous() if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, colscale, x0_subset, out_subset = ctx.saved_tensors
        # x0 is None if colscale is None
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dx1mat, dgamma, dbeta, *rest = _dropout_add_layer_norm_subset_backward(
            dz, dx, x, x0, dmask, mu, rsigma, gamma, colscale, x0_subset, out_subset, dropout_p,
            ctx.rowscale_const, ctx.x0_numrows, has_residual
        )
        dx0 = dx0mat.view(-1, *x.shape[1:])
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (dx0, dx1, dgamma, dbeta, dcolscale, None, None, None, None, None, None, None,
                None, None)


def layer_norm(x, weight, bias, epsilon):
    return DropoutAddLayerNormFn.apply(x, None, weight, bias, None, None, 0.0, epsilon, False)


def dropout_add_layer_norm(x0, x1, weight, bias, dropout_p, epsilon, rowscale=None, layerscale=None,
                           prenorm=False, residual_in_fp32=False,
                           return_dropout_mask=False):
    """residual_in_fp32 only has an effect if x1 is None.
    Otherwise residual dtype is x1.dtype.
    """
    return DropoutAddLayerNormFn.apply(
        x0, x1, weight, bias, rowscale, layerscale, dropout_p, epsilon, residual_in_fp32, prenorm,
        return_dropout_mask
    )


def dropout_add_layer_norm_subset(x0, x1, weight, bias, dropout_p, epsilon, layerscale=None,
                                  x0_subset=None, out_subset=None, rowscale_const=1.0,
                                  out_numrows=0, prenorm=False, residual_in_fp32=False,
                                  return_dropout_mask=False):
    """residual_in_fp32 only has an effect if x1 is None.
    Otherwise residual dtype is x1.dtype.
    """
    return DropoutAddLayerNormSubsetFn.apply(
        x0, x1, weight, bias, layerscale, x0_subset, out_subset, dropout_p, epsilon,
        rowscale_const, out_numrows, residual_in_fp32, prenorm, return_dropout_mask
    )


class DropoutAddLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, prenorm=False, p=0.0, eps=1e-5, residual_in_fp32=False,
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
