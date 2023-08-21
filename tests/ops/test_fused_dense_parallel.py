# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/ops/test_fused_dense_parallel.py

import math

import pytest
import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, FusedMLP, ParallelFusedMLP

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("dtype", [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize("sequence_parallel", [True, False])
# @pytest.mark.parametrize('sequence_parallel', [False])
@pytest.mark.parametrize("has_bias", [True, False])
# @pytest.mark.parametrize('has_bias', [False])
@pytest.mark.parametrize("out_features", [1024])
@pytest.mark.parametrize("in_features", [4096])
def test_fused_linear_bias(
    in_features, out_features, has_bias, sequence_parallel, world_size, dtype
):
    assert out_features % world_size == 0
    rtol, atol = (3e-3, 3e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    seqlen = 512
    assert batch_size * seqlen % world_size == 0
    x_pt = torch.randn(
        batch_size * seqlen, in_features, device=device, dtype=dtype, requires_grad=True
    )
    if sequence_parallel:
        x = (
            tensor_parallel.scatter_to_sequence_parallel_region(x_pt)
            .detach()
            .clone()
            .requires_grad_()
        )
    else:
        x = x_pt.detach().clone().requires_grad_()

    model_pt = torch.nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype)
    partition_out_features = out_features // world_size
    model = ColumnParallelLinear(
        in_features,
        out_features,
        parallel_state.get_tensor_model_parallel_group(),
        bias=has_bias,
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        model.weight.copy_(
            model_pt.weight[rank * partition_out_features : (rank + 1) * partition_out_features]
        )
        if has_bias:
            model.bias.copy_(
                model_pt.bias[rank * partition_out_features : (rank + 1) * partition_out_features]
            )

    out = model(x)
    out_pt = model_pt(x_pt)
    assert torch.allclose(
        out,
        out_pt[:, rank * partition_out_features : (rank + 1) * partition_out_features],
        rtol=rtol,
        atol=atol,
    )

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out_pt) / 32
    out_pt.backward(g)
    out.backward(g[:, rank * partition_out_features : (rank + 1) * partition_out_features])
    parallel_state.destroy_model_parallel()

    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        x.grad,
        x_pt.grad[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else x_pt.grad,
        rtol=rtol,
        atol=atol,
    )
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(
        model.weight.grad,
        model_pt.weight.grad[rank * partition_out_features : (rank + 1) * partition_out_features],
        rtol=rtol,
        atol=atol * 10,
    )
    if has_bias:
        assert torch.allclose(
            model.bias.grad,
            model_pt.bias.grad[rank * partition_out_features : (rank + 1) * partition_out_features],
            rtol=rtol,
            atol=atol * 5,
        )


@pytest.mark.parametrize("dtype", [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize("sequence_parallel", [True, False])
# @pytest.mark.parametrize('sequence_parallel', [False])
@pytest.mark.parametrize("has_bias2", [True, False])
# @pytest.mark.parametrize('has_bias2', [True])
@pytest.mark.parametrize("out_features", [4096])
@pytest.mark.parametrize("in_features", [1024])
def test_fused_mlp(in_features, out_features, has_bias2, sequence_parallel, world_size, dtype):
    assert out_features % world_size == 0
    rtol, atol = (3e-3, 3e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    seqlen = 512
    assert batch_size * seqlen % world_size == 0
    x_pt = torch.randn(
        batch_size * seqlen, in_features, device=device, dtype=dtype, requires_grad=True
    )
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
    else:
        x = x_pt.detach().clone().requires_grad_()

    model_pt_fc1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    model_pt_fc2 = torch.nn.Linear(
        out_features, in_features, bias=has_bias2, device=device, dtype=dtype
    )
    partition_out_features = out_features // world_size
    partition_in_features = in_features // world_size
    model = ParallelFusedMLP(
        in_features,
        out_features,
        in_features,
        process_group=parallel_state.get_tensor_model_parallel_group(),
        bias2=has_bias2 and rank == 0,
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        model.fc1.weight.copy_(
            model_pt_fc1.weight[rank * partition_out_features : (rank + 1) * partition_out_features]
        )
        model.fc1.bias.copy_(
            model_pt_fc1.bias[rank * partition_out_features : (rank + 1) * partition_out_features]
        )
        model.fc2.weight.copy_(
            model_pt_fc2.weight[
                :, rank * partition_out_features : (rank + 1) * partition_out_features
            ]
        )
        if has_bias2 and rank == 0:
            model.fc2.bias.copy_(model_pt_fc2.bias)

    out = model(x)
    out_pt = model_pt_fc2(F.gelu(model_pt_fc1(x_pt), approximate="tanh"))
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out,
        out_pt[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else out_pt,
        rtol=rtol,
        atol=atol,
    )

    out_pt.backward(g)
    out.backward(
        g[rank * partition_batch_dim : (rank + 1) * partition_batch_dim] if sequence_parallel else g
    )
    parallel_state.destroy_model_parallel()

    assert torch.allclose(
        x.grad,
        x_pt.grad[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else x_pt.grad,
        rtol=rtol,
        atol=atol,
    )
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(
        model.fc1.weight.grad,
        model_pt_fc1.weight.grad[
            rank * partition_out_features : (rank + 1) * partition_out_features
        ],
        rtol=rtol,
        atol=atol * 10,
    )
    assert torch.allclose(
        model.fc1.bias.grad,
        model_pt_fc1.bias.grad[rank * partition_out_features : (rank + 1) * partition_out_features],
        rtol=rtol,
        atol=atol * 5,
    )
    assert torch.allclose(
        model.fc2.weight.grad,
        model_pt_fc2.weight.grad[
            :, rank * partition_out_features : (rank + 1) * partition_out_features
        ],
        rtol=rtol,
        atol=atol * 10,
    )
    if has_bias2 and rank == 0:
        assert torch.allclose(model.fc2.bias.grad, model_pt_fc2.bias.grad, rtol=rtol, atol=atol * 5)
