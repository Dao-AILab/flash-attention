# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/models/test_gpt_parallel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from einops import rearrange

from transformers import GPT2Config

from apex.transformer import parallel_state

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.utils.distributed import allreduce_sequence_parallel_grad

is_sm8x = torch.cuda.get_device_capability('cuda')[0] >= 8


@pytest.mark.parametrize('dtype', [torch.float16] + ([torch.bfloat16] if is_sm8x else []))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('world_size', [1, 2, 4, 8])
# @pytest.mark.parametrize('world_size', [1])
@pytest.mark.parametrize('has_pos_emb', [True, False])
# @pytest.mark.parametrize('has_pos_emb', [True])
@pytest.mark.parametrize('dim', [1024])
def test_block_parallel(dim, has_pos_emb, world_size, dtype):
    head_dim = 64
    assert dim % head_dim == 0
    num_heads = dim // head_dim
    assert num_heads % world_size == 0
    vocab_size = 50264
    assert vocab_size % world_size == 0
    num_layers = 2
    rtol, atol = (3e-3, 1e-1) if dtype == torch.bfloat16 else (3e-3, 1e-2)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = f'cuda:{torch.distributed.get_rank()}'
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 1024
    assert (batch_size * seqlen) % world_size == 0
    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen + 1), device=device)

    # We need to generate g here so that all processes get the same gradient,
    # as rank 0 will have an extra bias that changes the RNG.
    g = torch.randn(batch_size * seqlen, device=device)

    config = GPT2Config(n_embd=dim, n_head=num_heads, n_layer=num_layers,
                        n_positions=seqlen if has_pos_emb else 0,
                        vocab_size=50257, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                        scale_attn_by_inverse_layer_idx=True, use_flash_attn=True,
                        fused_dense_gelu_dense=True, fused_bias_fc=True, fused_dropout_add_ln=True,
                        rotary_emb_fraction=0.0 if has_pos_emb else 0.5,
                        pad_vocab_size_multiple=8 * world_size)
    model_pt = GPTLMHeadModel(config, device=device)

    def init_layer_norm(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight)
            nn.init.normal_(module.bias)
    model_pt.apply(init_layer_norm)

    model = GPTLMHeadModel(config, process_group=process_group, device=device)
    total_nparams = sum(p.numel() for p in model_pt.parameters())
    sharded_nparams = sum(p.numel() for p in model.parameters())
    sharded_nparams_all = torch.empty(world_size, dtype=torch.long, device=device)
    torch.distributed.all_gather_into_tensor(
        sharded_nparams_all, torch.tensor([sharded_nparams], device=device), group=process_group
    )
    sequence_parallel_nparams = sum(p.numel() for p in model.parameters()
                                    if getattr(p, '_sequence_parallel', False))
    sequence_parallel_nparams_all = torch.empty(world_size, dtype=torch.long, device=device)
    torch.distributed.all_gather_into_tensor(
        sequence_parallel_nparams_all, torch.tensor([sequence_parallel_nparams], device=device),
        group=process_group
    )
    assert torch.all(sequence_parallel_nparams_all == sequence_parallel_nparams)
    assert total_nparams == ((sharded_nparams_all - sequence_parallel_nparams_all).sum().item()
                             + sequence_parallel_nparams)

    # vocab_size has been rounded up here
    partition_vocab_size = config.vocab_size // world_size
    partition_dim = dim // world_size
    partition_hidden_dim = 4 * dim // world_size
    with torch.no_grad():
        model.transformer.embeddings.word_embeddings.weight.copy_(
            model_pt.transformer.embeddings.word_embeddings.weight[rank * partition_vocab_size:(rank + 1) * partition_vocab_size]
        )
        if has_pos_emb:
            model.transformer.embeddings.position_embeddings.weight.copy_(
                model_pt.transformer.embeddings.position_embeddings.weight[:, rank * partition_dim:(rank + 1) * partition_dim]
            )
        model.transformer.ln_0.weight.copy_(model_pt.transformer.ln_0.weight)
        model.transformer.ln_0.bias.copy_(model_pt.transformer.ln_0.bias)
        for i in range(num_layers):
            model.transformer.layers[i].mixer.Wqkv.weight.copy_(
                rearrange(rearrange(model_pt.transformer.layers[i].mixer.Wqkv.weight, '(three o) i -> three o i', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                        'three o i -> (three o) i')
            )
            model.transformer.layers[i].mixer.Wqkv.bias.copy_(
                rearrange(rearrange(model_pt.transformer.layers[i].mixer.Wqkv.bias, '(three o) -> three o', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                        'three o -> (three o)')
            )
            model.transformer.layers[i].mixer.out_proj.weight.copy_(
                model_pt.transformer.layers[i].mixer.out_proj.weight[:, rank * partition_dim:(rank + 1) * partition_dim]
            )
            if rank == 0:
                model.transformer.layers[i].mixer.out_proj.bias.copy_(model_pt.transformer.layers[i].mixer.out_proj.bias)
            model.transformer.layers[i].mlp.fc1.weight.copy_(
                model_pt.transformer.layers[i].mlp.fc1.weight[rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim]
            )
            model.transformer.layers[i].mlp.fc1.bias.copy_(
                model_pt.transformer.layers[i].mlp.fc1.bias[rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim]
            )
            model.transformer.layers[i].mlp.fc2.weight.copy_(
                model_pt.transformer.layers[i].mlp.fc2.weight[:, rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim]
            )
            if rank == 0:
                model.transformer.layers[i].mlp.fc2.bias.copy_(model_pt.transformer.layers[i].mlp.fc2.bias)
            model.transformer.layers[i].norm1.weight.copy_(model_pt.transformer.layers[i].norm1.weight)
            model.transformer.layers[i].norm1.bias.copy_(model_pt.transformer.layers[i].norm1.bias)
            model.transformer.layers[i].norm2.weight.copy_(model_pt.transformer.layers[i].norm2.weight)
            model.transformer.layers[i].norm2.bias.copy_(model_pt.transformer.layers[i].norm2.bias)
        # Don't need to copy the lm_head weight since it's tied to the word embedding weight

    with torch.autocast(device_type='cuda', dtype=dtype):
        out = model(input_ids[:, :-1]).logits
        out_pt = rearrange(model_pt(input_ids[:, :-1]).logits, 'b s d -> (b s) d')
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out, out_pt[:, rank * partition_vocab_size:(rank + 1) * partition_vocab_size],
        rtol=rtol, atol=atol
    )
    loss_fn = CrossEntropyLoss(inplace_backward=True, reduction='none', process_group=process_group)
    loss_fn_pt = CrossEntropyLoss(inplace_backward=True, reduction='none')
    loss = loss_fn(out, input_ids[:, 1:].flatten())
    loss_pt = loss_fn_pt(out_pt, input_ids[:, 1:].flatten())
    assert torch.allclose(loss, loss_pt, rtol=rtol, atol=atol)

    loss_pt.backward(g)
    loss.backward(g)
    allreduce_sequence_parallel_grad(model, process_group)
    parallel_state.destroy_model_parallel()

    assert torch.allclose(
        model.transformer.embeddings.word_embeddings.weight.grad,
        model_pt.transformer.embeddings.word_embeddings.weight.grad[rank * partition_vocab_size:(rank + 1) * partition_vocab_size],
        rtol=rtol, atol=atol * 5
    )
    if has_pos_emb:
        assert torch.allclose(
            model.transformer.embeddings.position_embeddings.weight.grad,
            model_pt.transformer.embeddings.position_embeddings.weight.grad[:, rank * partition_dim:(rank + 1) * partition_dim],
            rtol=rtol, atol=atol
        )
    assert torch.allclose(model.transformer.ln_0.weight.grad, model_pt.transformer.ln_0.weight.grad,
                          rtol=rtol, atol=atol)
    assert torch.allclose(model.transformer.ln_0.bias.grad, model_pt.transformer.ln_0.bias.grad,
                          rtol=rtol, atol=atol)
    for i in range(num_layers):
        # if rank == 0: breakpoint()
        # torch.distributed.barrier()
        assert torch.allclose(
            model.transformer.layers[i].mixer.Wqkv.weight.grad,
            rearrange(rearrange(model_pt.transformer.layers[i].mixer.Wqkv.weight.grad, '(three o) i -> three o i', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim], 'three o i -> (three o) i'),
            rtol=rtol, atol=atol * 10
        )
        assert torch.allclose(
            model.transformer.layers[i].mixer.Wqkv.bias.grad,
            rearrange(rearrange(model_pt.transformer.layers[i].mixer.Wqkv.bias.grad, '(three o) -> three o', three=3)[:, rank * partition_dim:(rank + 1) * partition_dim],
                    'three o -> (three o)'),
            rtol=rtol, atol=atol * 10
        )
        assert torch.allclose(
            model.transformer.layers[i].mixer.out_proj.weight.grad,
            model_pt.transformer.layers[i].mixer.out_proj.weight.grad[:, rank * partition_dim:(rank + 1) * partition_dim],
            rtol=rtol, atol=atol * 10
        )
        if rank == 0:
            assert torch.allclose(model.transformer.layers[i].mixer.out_proj.bias.grad, model_pt.transformer.layers[i].mixer.out_proj.bias.grad, rtol=rtol, atol=atol * 5)
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc1.weight.grad,
            model_pt.transformer.layers[i].mlp.fc1.weight.grad[rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim],
            rtol=rtol, atol=atol * 10
        )
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc1.bias.grad,
            model_pt.transformer.layers[i].mlp.fc1.bias.grad[rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim],
            rtol=rtol, atol=atol * 10
        )
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc2.weight.grad,
            model_pt.transformer.layers[i].mlp.fc2.weight.grad[:, rank * partition_hidden_dim:(rank + 1) * partition_hidden_dim],
            rtol=rtol, atol=atol * 10
        )
        if rank == 0:
            assert torch.allclose(model.transformer.layers[i].mlp.fc2.bias.grad, model_pt.transformer.layers[i].mlp.fc2.bias.grad,
                                rtol=rtol, atol=atol * 5)

        assert torch.allclose(model.transformer.layers[i].norm1.weight.grad, model_pt.transformer.layers[i].norm1.weight.grad, rtol=rtol, atol=atol)
        assert torch.allclose(model.transformer.layers[i].norm1.bias.grad, model_pt.transformer.layers[i].norm1.bias.grad, rtol=rtol, atol=atol)
        assert torch.allclose(model.transformer.layers[i].norm2.weight.grad, model_pt.transformer.layers[i].norm2.weight.grad, rtol=rtol, atol=atol)
        assert torch.allclose(model.transformer.layers[i].norm2.bias.grad, model_pt.transformer.layers[i].norm2.bias.grad, rtol=rtol, atol=atol)
