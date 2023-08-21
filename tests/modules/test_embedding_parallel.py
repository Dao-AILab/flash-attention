# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/modules/test_embedding_parallel.py

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.transformer import parallel_state
from einops import rearrange
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("dtype", [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize("sequence_parallel", [True, False])
# @pytest.mark.parametrize('sequence_parallel', [False])
@pytest.mark.parametrize("has_pos_emb", [True, False])
# @pytest.mark.parametrize('has_pos_emb', [True])
@pytest.mark.parametrize("dim", [1024])
def test_embedding_parallel(dim, has_pos_emb, sequence_parallel, world_size, dtype):
    vocab_size = 50264
    seqlen = 2048
    assert vocab_size % world_size == 0
    assert dim % world_size == 0
    rtol, atol = (3e-3, 5e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    input_ids_pt = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
    input_ids = input_ids_pt.detach().clone()

    model_pt = GPT2Embeddings(
        dim, vocab_size, seqlen if has_pos_emb else 0, device=device, dtype=dtype
    )
    model = ParallelGPT2Embeddings(
        dim,
        vocab_size,
        seqlen if has_pos_emb else 0,
        parallel_state.get_tensor_model_parallel_group(),
        sequence_parallel=sequence_parallel,
        device=device,
        dtype=dtype,
    )
    partition_vocab_size = vocab_size // world_size
    partition_dim = dim // world_size
    with torch.no_grad():
        model.word_embeddings.weight.copy_(
            model_pt.word_embeddings.weight[
                rank * partition_vocab_size : (rank + 1) * partition_vocab_size
            ]
        )
        if has_pos_emb:
            model.position_embeddings.weight.copy_(
                model_pt.position_embeddings.weight[
                    :, rank * partition_dim : (rank + 1) * partition_dim
                ]
            )

    out = model(input_ids, combine_batch_seqlen_dim=True)
    out_pt = rearrange(model_pt(input_ids), "b s d -> (b s) d")
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out,
        out_pt[rank * partition_batch_dim : (rank + 1) * partition_batch_dim]
        if sequence_parallel
        else out_pt,
        rtol=rtol,
        atol=atol,
    )

    g = torch.randn_like(out_pt)
    out_pt.backward(g)
    out.backward(
        g[rank * partition_batch_dim : (rank + 1) * partition_batch_dim] if sequence_parallel else g
    )
    parallel_state.destroy_model_parallel()

    assert torch.allclose(
        model.word_embeddings.weight.grad,
        model_pt.word_embeddings.weight.grad[
            rank * partition_vocab_size : (rank + 1) * partition_vocab_size
        ],
        rtol=rtol,
        atol=atol,
    )
    if has_pos_emb:
        assert torch.allclose(
            model.position_embeddings.weight.grad,
            model_pt.position_embeddings.weight.grad[
                :, rank * partition_dim : (rank + 1) * partition_dim
            ],
            rtol=rtol,
            atol=atol,
        )
