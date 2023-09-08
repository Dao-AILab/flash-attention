# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/losses/test_cross_entropy_parallel.py

import math

import pytest
import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from flash_attn.losses.cross_entropy import CrossEntropyLoss

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32] + ([torch.bfloat16] if is_sm8x else [])
)
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("inplace_backward", [False, True])
# @pytest.mark.parametrize('inplace_backward', [False])
@pytest.mark.parametrize("smoothing", [0.0, 0.9])
# @pytest.mark.parametrize('smoothing', [0.9])
@pytest.mark.parametrize("vocab_size", [50264])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
def test_cross_entropy_loss_parallel(vocab_size, world_size, smoothing, inplace_backward, dtype):
    assert vocab_size % world_size == 0
    rtol, atol = (
        (1e-5, 1e-6)
        if dtype == torch.float32
        else ((1e-3, 1e-4) if dtype == torch.float16 else (1e-2, 3e-3))
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    partition_vocab_size = vocab_size // world_size
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 128
    x_pt = (
        torch.randn(batch_size * seqlen, vocab_size, device=device, dtype=dtype) * 10
    ).requires_grad_()
    x = (
        tensor_parallel.scatter_to_tensor_model_parallel_region(x_pt)
        .detach()
        .clone()
        .requires_grad_()
    )
    y = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    y[torch.randperm(batch_size * seqlen)[:10]] = -100
    model_pt = torch.nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")
    model = CrossEntropyLoss(
        label_smoothing=smoothing,
        reduction="none",
        inplace_backward=inplace_backward,
        process_group=parallel_state.get_tensor_model_parallel_group(),
    )
    out = model(x, y)
    out_pt = model_pt(x_pt.float(), y)
    assert torch.allclose(out, out_pt, rtol=1e-5, atol=1e-6)

    g = torch.randn_like(out)
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(
        x.grad,
        x_pt.grad[:, (rank * partition_vocab_size) : (rank + 1) * partition_vocab_size],
        rtol=rtol,
        atol=atol,
    )

    parallel_state.destroy_model_parallel()
