# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/modules/test_block_parallel.py

import math
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from einops import rearrange
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import FusedMLP, ParallelFusedMLP
from flash_attn.utils.distributed import allreduce_sequence_parallel_grad

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("dtype", [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize("sequence_parallel", [True, False])
# @pytest.mark.parametrize('sequence_parallel', [True])
@pytest.mark.parametrize("dim", [1024])
def test_block_parallel(dim, sequence_parallel, world_size, dtype):
    head_dim = 64
    assert dim % head_dim == 0
    num_heads = dim // head_dim
    assert num_heads % world_size == 0
    rtol, atol = (3e-3, 5e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    x_pt = torch.randn(batch_size * seqlen, dim, device=device, dtype=dtype, requires_grad=True)
    residual_pt = torch.randn(batch_size * seqlen, dim, device=device, requires_grad=True)
    # We need to generate g here so that all processes get the same gradient,
    # as rank 0 will have an extra bias that changes the RNG.
    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(x_pt) / 32
    if sequence_parallel:
        x = (
            tensor_parallel.scatter_to_sequence_parallel_region(x_pt)
            .detach()
            .clone()
            .requires_grad_()
        )
        residual = (
            tensor_parallel.scatter_to_sequence_parallel_region(residual_pt)
            .detach()
            .clone()
            .requires_grad_()
        )
    else:
        x = x_pt.detach().clone().requires_grad_()
        residual = residual_pt.detach().clone().requires_grad_()

    mixer_cls_pt = partial(
        MHA,
        num_heads=num_heads,
        rotary_emb_dim=int(head_dim // 2),
        use_flash_attn=True,
        device=device,
        dtype=dtype,
    )
    mlp_cls_pt = partial(FusedMLP, hidden_features=4 * dim, device=device, dtype=dtype)
    norm_cls = partial(nn.LayerNorm, device=device, dtype=dtype)
    model_pt = Block(dim, mixer_cls_pt, mlp_cls_pt, norm_cls, fused_dropout_add_ln=True)
    with torch.no_grad():
        nn.init.normal_(model_pt.norm1.weight)
        nn.init.normal_(model_pt.norm1.bias)
        nn.init.normal_(model_pt.norm2.weight)
        nn.init.normal_(model_pt.norm2.bias)

    mixer_cls = partial(
        ParallelMHA,
        num_heads=num_heads,
        process_group=parallel_state.get_tensor_model_parallel_group(),
        rotary_emb_dim=int(head_dim // 2),
        use_flash_attn=True,
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )
    mlp_cls = partial(
        ParallelFusedMLP,
        hidden_features=4 * dim,
        process_group=parallel_state.get_tensor_model_parallel_group(),
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )
    model = Block(
        dim,
        mixer_cls,
        mlp_cls,
        norm_cls,
        fused_dropout_add_ln=True,
        sequence_parallel=sequence_parallel,
        mark_shared_params=True,
    )

    partition_dim = dim // world_size
    partition_hidden_dim = 4 * dim // world_size
    with torch.no_grad():
        model.mixer.Wqkv.weight.copy_(
            rearrange(
                rearrange(model_pt.mixer.Wqkv.weight, "(three o) i -> three o i", three=3)[
                    :, rank * partition_dim : (rank + 1) * partition_dim
                ],
                "three o i -> (three o) i",
            )
        )
        model.mixer.Wqkv.bias.copy_(
            rearrange(
                rearrange(model_pt.mixer.Wqkv.bias, "(three o) -> three o", three=3)[
                    :, rank * partition_dim : (rank + 1) * partition_dim
                ],
                "three o -> (three o)",
            )
        )
        model.mixer.out_proj.weight.copy_(
            model_pt.mixer.out_proj.weight[:, rank * partition_dim : (rank + 1) * partition_dim]
        )
        if rank == 0:
            model.mixer.out_proj.bias.copy_(model_pt.mixer.out_proj.bias)
        model.mlp.fc1.weight.copy_(
            model_pt.mlp.fc1.weight[rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim]
        )
        model.mlp.fc1.bias.copy_(
            model_pt.mlp.fc1.bias[rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim]
        )
        model.mlp.fc2.weight.copy_(
            model_pt.mlp.fc2.weight[
                :, rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim
            ]
        )
        if rank == 0:
            model.mlp.fc2.bias.copy_(model_pt.mlp.fc2.bias)
        model.norm1.weight.copy_(model_pt.norm1.weight)
        model.norm1.bias.copy_(model_pt.norm1.bias)
        model.norm2.weight.copy_(model_pt.norm2.weight)
        model.norm2.bias.copy_(model_pt.norm2.bias)

    mixer_kwargs = {"seqlen": seqlen}
    out, out_residual = model(x, residual, mixer_kwargs=mixer_kwargs)
    out_pt, out_residual_pt = model_pt(
        rearrange(x_pt, "(b s) d -> b s d", s=seqlen),
        rearrange(residual_pt, "(b s) d -> b s d", s=seqlen),
    )
    out_pt, out_residual_pt = [rearrange(x, "b s d -> (b s) d") for x in [out_pt, out_residual_pt]]
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out,
        out_pt[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else out_pt,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        out_residual,
        out_residual_pt[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else out_residual_pt,
        rtol=rtol,
        atol=atol,
    )

    (out_pt + 2 * out_residual_pt).backward(g)
    (out + 2 * out_residual).backward(
        g[rank * partition_batch_dim : (rank + 1) * partition_batch_dim] if sequence_parallel else g
    )
    allreduce_sequence_parallel_grad(model, parallel_state.get_tensor_model_parallel_group())
    parallel_state.destroy_model_parallel()

    assert torch.allclose(
        x.grad,
        x_pt.grad[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else x_pt.grad,
        rtol=rtol,
        atol=atol / 10,  # magnitude of x.grad is quite small
    )
    assert torch.allclose(
        residual.grad,
        residual_pt.grad[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else residual_pt.grad,
        rtol=rtol,
        atol=atol,
    )
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(
        model.mixer.Wqkv.weight.grad,
        rearrange(
            rearrange(model_pt.mixer.Wqkv.weight.grad, "(three o) i -> three o i", three=3)[
                :, rank * partition_dim : (rank + 1) * partition_dim
            ],
            "three o i -> (three o) i",
        ),
        rtol=rtol,
        atol=atol * 10,
    )
    assert torch.allclose(
        model.mixer.Wqkv.bias.grad,
        rearrange(
            rearrange(model_pt.mixer.Wqkv.bias.grad, "(three o) -> three o", three=3)[
                :, rank * partition_dim : (rank + 1) * partition_dim
            ],
            "three o -> (three o)",
        ),
        rtol=rtol,
        atol=atol * 5,
    )
    assert torch.allclose(
        model.mixer.out_proj.weight.grad,
        model_pt.mixer.out_proj.weight.grad[:, rank * partition_dim : (rank + 1) * partition_dim],
        rtol=rtol,
        atol=atol * 10,
    )
    if rank == 0:
        assert torch.allclose(
            model.mixer.out_proj.bias.grad,
            model_pt.mixer.out_proj.bias.grad,
            rtol=rtol,
            atol=atol * 5,
        )
    assert torch.allclose(
        model.mlp.fc1.weight.grad,
        model_pt.mlp.fc1.weight.grad[
            rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim
        ],
        rtol=rtol,
        atol=atol * 10,
    )
    assert torch.allclose(
        model.mlp.fc1.bias.grad,
        model_pt.mlp.fc1.bias.grad[rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim],
        rtol=rtol,
        atol=atol * 5,
    )
    assert torch.allclose(
        model.mlp.fc2.weight.grad,
        model_pt.mlp.fc2.weight.grad[
            :, rank * partition_hidden_dim : (rank + 1) * partition_hidden_dim
        ],
        rtol=rtol,
        atol=atol * 10,
    )
    if rank == 0:
        assert torch.allclose(
            model.mlp.fc2.bias.grad, model_pt.mlp.fc2.bias.grad, rtol=rtol, atol=atol * 5
        )

    assert torch.allclose(
        model.norm1.weight.grad, model_pt.norm1.weight.grad, rtol=rtol, atol=atol * 5
    )
    assert torch.allclose(model.norm1.bias.grad, model_pt.norm1.bias.grad, rtol=rtol, atol=atol * 5)
    assert torch.allclose(
        model.norm2.weight.grad, model_pt.norm2.weight.grad, rtol=rtol, atol=atol * 5
    )
    assert torch.allclose(model.norm2.bias.grad, model_pt.norm2.bias.grad, rtol=rtol, atol=atol * 5)
