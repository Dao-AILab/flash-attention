# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/modules/test_mlp_parallel.py

import pytest
import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from einops import rearrange
from flash_attn.modules.mlp import GatedMlp, ParallelGatedMlp

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("dtype", [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize("sequence_parallel", [True, False])
# @pytest.mark.parametrize('sequence_parallel', [False])
@pytest.mark.parametrize("activation", [F.silu, F.sigmoid])
# @pytest.mark.parametrize('activation', [F.silu])
@pytest.mark.parametrize("dim", [1024, 4096])
# @pytest.mark.parametrize('dim', [1024])
def test_mlp_parallel(dim, activation, sequence_parallel, world_size, dtype):
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
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    x_pt = torch.randn(batch_size * seqlen, dim, device=device, dtype=dtype, requires_grad=True)
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

    model_pt = GatedMlp(dim, activation=activation, device=device, dtype=dtype)
    partition_dim = model_pt.fc1.weight.shape[0] // 2 // world_size
    model = ParallelGatedMlp(
        dim,
        parallel_state.get_tensor_model_parallel_group(),
        activation=activation,
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        model.fc1.weight.copy_(
            rearrange(
                rearrange(model_pt.fc1.weight, "(two o) i -> two o i", two=2)[
                    :, rank * partition_dim : (rank + 1) * partition_dim
                ],
                "two o i -> (two o) i",
            )
        )
        model.fc1.bias.copy_(
            rearrange(
                rearrange(model_pt.fc1.bias, "(two o) -> two o", two=2)[
                    :, rank * partition_dim : (rank + 1) * partition_dim
                ],
                "two o -> (two o)",
            )
        )
        model.fc2.weight.copy_(
            model_pt.fc2.weight[:, rank * partition_dim : (rank + 1) * partition_dim]
        )
        if rank == 0:
            model.fc2.bias.copy_(model_pt.fc2.bias)

    out = model(x)
    out_pt = model_pt(x_pt)
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

    assert torch.allclose(
        model.fc1.weight.grad,
        rearrange(
            rearrange(model_pt.fc1.weight.grad, "(two o) i -> two o i", two=2)[
                :, rank * partition_dim : (rank + 1) * partition_dim
            ],
            "two o i -> (two o) i",
        ),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        model.fc1.bias.grad,
        rearrange(
            rearrange(model_pt.fc1.bias.grad, "(two o) -> two o", two=2)[
                :, rank * partition_dim : (rank + 1) * partition_dim
            ],
            "two o -> (two o)",
        ),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        model.fc2.weight.grad,
        model_pt.fc2.weight.grad[:, rank * partition_dim : (rank + 1) * partition_dim],
        rtol=rtol,
        atol=atol,
    )
    if rank == 0:
        assert torch.allclose(model.fc2.bias.grad, model_pt.fc2.bias.grad, rtol=rtol, atol=atol)
