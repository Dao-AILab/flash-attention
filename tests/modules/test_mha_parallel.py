# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/modules/test_mha_parallel.py

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel

from flash_attn.modules.mha import MHA, ParallelMHA

is_sm8x = torch.cuda.get_device_capability('cuda')[0] >= 8


@pytest.mark.parametrize('dtype', [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('world_size', [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize('head_dim', [64, 128])
# @pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('embed_dim', [1024, 4096])
# @pytest.mark.parametrize('embed_dim', [1024])
def test_mha_parallel(embed_dim, head_dim, world_size, dtype):
    assert embed_dim % head_dim == 0
    num_heads = embed_dim // head_dim
    assert num_heads % world_size == 0
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = f'cuda:{torch.distributed.get_rank()}'
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    x_pt = torch.randn(batch_size * seqlen, embed_dim, device=device, dtype=dtype,
                       requires_grad=True)
    # We need to generate g here so that all processes get the same gradient,
    # as rank 0 will have an extra bias that changes the RNG.
    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(x_pt) / 32
    x = tensor_parallel.scatter_to_sequence_parallel_region(x_pt).detach().clone().requires_grad_()

    model_pt = MHA(embed_dim, num_heads, rotary_emb_dim=int(head_dim // 2),
                   use_flash_attn=True, device=device, dtype=dtype)
    partition_dim = embed_dim // world_size
    model = ParallelMHA(embed_dim, num_heads, parallel_state.get_tensor_model_parallel_group(),
                        rotary_emb_dim=int(head_dim // 2), use_flash_attn=True,
                        device=device, dtype=dtype)

    with torch.no_grad():
        model.Wqkv.weight.copy_(
            rearrange(rearrange(model_pt.Wqkv.weight, '(three o) i -> three o i', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                      'three o i -> (three o) i')
        )
        model.Wqkv.bias.copy_(
            rearrange(rearrange(model_pt.Wqkv.bias, '(three o) -> three o', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                      'three o -> (three o)')
        )
        model.out_proj.weight.copy_(
            model_pt.out_proj.weight[:, rank * partition_dim:(rank + 1) * partition_dim]
        )
        if rank == 0:
            model.out_proj.bias.copy_(model_pt.out_proj.bias)

    out = model(x, seqlen=seqlen)
    out_pt = rearrange(model_pt(rearrange(x_pt, '(b s) d -> b s d', s=seqlen)), 'b s d -> (b s) d')
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out, out_pt[rank * partition_batch_dim:(rank + 1) * partition_batch_dim],
        rtol=rtol, atol=atol
    )

    out_pt.backward(g)
    out.backward(g[rank * partition_batch_dim:(rank + 1) * partition_batch_dim])
    parallel_state.destroy_model_parallel()

    assert torch.allclose(
        x.grad, x_pt.grad[rank * partition_batch_dim:(rank + 1) * partition_batch_dim],
        rtol=rtol, atol=atol
    )
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(
        model.Wqkv.weight.grad,
        rearrange(rearrange(model_pt.Wqkv.weight.grad, '(three o) i -> three o i', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                  'three o i -> (three o) i'),
        rtol=rtol, atol=atol * 10
    )
    assert torch.allclose(
        model.Wqkv.bias.grad,
        rearrange(rearrange(model_pt.Wqkv.bias.grad, '(three o) -> three o', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                  'three o -> (three o)'),
        rtol=rtol, atol=atol * 5
    )
    assert torch.allclose(
        model.out_proj.weight.grad,
        model_pt.out_proj.weight.grad[:, rank * partition_dim:(rank + 1) * partition_dim],
        rtol=rtol, atol=atol * 10
    )
    if rank == 0:
        assert torch.allclose(model.out_proj.bias.grad, model_pt.out_proj.bias.grad, rtol=rtol, atol=atol * 5)
