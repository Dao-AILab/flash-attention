# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/models/test_gpt_parallel.py

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.transformer import parallel_state
from einops import rearrange
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.models.gpt import GPTLMHeadModel, shard_state_dict_tp
from flash_attn.utils.distributed import allreduce_sequence_parallel_grad
from transformers import GPT2Config

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
def test_gpt_parallel(dim, has_pos_emb, sequence_parallel, world_size, dtype):
    head_dim = 64
    assert dim % head_dim == 0
    num_heads = dim // head_dim
    assert num_heads % world_size == 0
    vocab_size = 50264
    assert vocab_size % world_size == 0
    num_layers = 2
    rtol, atol = (3e-3, 1e-1) if dtype == torch.bfloat16 else (3e-3, 1e-2)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
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

    config = GPT2Config(
        n_embd=dim,
        n_head=num_heads,
        n_layer=num_layers,
        n_positions=seqlen if has_pos_emb else 0,
        vocab_size=50257,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        scale_attn_by_inverse_layer_idx=True,
        use_flash_attn=True,
        fused_mlp=True,
        fused_bias_fc=True,
        fused_dropout_add_ln=True,
        residual_in_fp32=True,
        rotary_emb_fraction=0.0 if has_pos_emb else 0.5,
        pad_vocab_size_multiple=8 * world_size,
        sequence_parallel=sequence_parallel,
    )
    config.vocab_size = math.ceil(config.vocab_size / (8 * world_size)) * (8 * world_size)
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
    shared_nparams = sum(
        p.numel() for p in model.parameters() if getattr(p, "_shared_params", False)
    )
    shared_nparams_all = torch.empty(world_size, dtype=torch.long, device=device)
    torch.distributed.all_gather_into_tensor(
        shared_nparams_all, torch.tensor([shared_nparams], device=device), group=process_group
    )
    assert torch.all(shared_nparams_all == shared_nparams)
    assert total_nparams == (
        (sharded_nparams_all - shared_nparams_all).sum().item() + shared_nparams
    )

    # vocab_size has been rounded up here
    partition_vocab_size = config.vocab_size // world_size
    partition_dim = dim // world_size
    partition_hidden_dim = 4 * dim // world_size
    with torch.no_grad():
        model.load_state_dict(shard_state_dict_tp(model_pt.state_dict(), config, world_size, rank))
        model.tie_weights()

    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(input_ids[:, :-1]).logits
        if not sequence_parallel:
            out = rearrange(out, "b s d -> (b s) d")
        out_pt = rearrange(model_pt(input_ids[:, :-1]).logits, "b s d -> (b s) d")
    partition_batch_dim = batch_size * seqlen // world_size
    assert torch.allclose(
        out,
        out_pt[:, rank * partition_vocab_size : (rank + 1) * partition_vocab_size],
        rtol=rtol,
        atol=atol,
    )
    loss_fn = CrossEntropyLoss(inplace_backward=True, reduction="none", process_group=process_group)
    loss_fn_pt = CrossEntropyLoss(inplace_backward=True, reduction="none")
    loss = loss_fn(out, input_ids[:, 1:].flatten())
    loss_pt = loss_fn_pt(out_pt, input_ids[:, 1:].flatten())
    assert torch.allclose(loss, loss_pt, rtol=rtol, atol=atol)

    loss_pt.backward(g)
    loss.backward(g)
    allreduce_sequence_parallel_grad(model, process_group)
    parallel_state.destroy_model_parallel()

    grad_dict = shard_state_dict_tp(
        {k: v.grad for k, v in model_pt.named_parameters()}, config, world_size, rank
    )

    assert torch.allclose(
        model.transformer.embeddings.word_embeddings.weight.grad,
        grad_dict["transformer.embeddings.word_embeddings.weight"],
        rtol=rtol,
        atol=atol * 5,
    )
    if has_pos_emb:
        assert torch.allclose(
            model.transformer.embeddings.position_embeddings.weight.grad,
            grad_dict["transformer.embeddings.position_embeddings.weight"],
            rtol=rtol,
            atol=atol,
        )
    assert torch.allclose(
        model.transformer.ln_f.weight.grad,
        grad_dict["transformer.ln_f.weight"],
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        model.transformer.ln_f.bias.grad, grad_dict["transformer.ln_f.bias"], rtol=rtol, atol=atol
    )
    for i in range(num_layers):
        assert torch.allclose(
            model.transformer.layers[i].mixer.Wqkv.weight.grad,
            grad_dict[f"transformer.layers.{i}.mixer.Wqkv.weight"],
            rtol=rtol,
            atol=atol * 10,
        )
        assert torch.allclose(
            model.transformer.layers[i].mixer.Wqkv.bias.grad,
            grad_dict[f"transformer.layers.{i}.mixer.Wqkv.bias"],
            rtol=rtol,
            atol=atol * 10,
        )
        assert torch.allclose(
            model.transformer.layers[i].mixer.out_proj.weight.grad,
            grad_dict[f"transformer.layers.{i}.mixer.out_proj.weight"],
            rtol=rtol,
            atol=atol * 10,
        )
        if rank == 0:
            assert torch.allclose(
                model.transformer.layers[i].mixer.out_proj.bias.grad,
                grad_dict[f"transformer.layers.{i}.mixer.out_proj.bias"],
                rtol=rtol,
                atol=atol * 5,
            )
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc1.weight.grad,
            grad_dict[f"transformer.layers.{i}.mlp.fc1.weight"],
            rtol=rtol,
            atol=atol * 10,
        )
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc1.bias.grad,
            grad_dict[f"transformer.layers.{i}.mlp.fc1.bias"],
            rtol=rtol,
            atol=atol * 10,
        )
        assert torch.allclose(
            model.transformer.layers[i].mlp.fc2.weight.grad,
            grad_dict[f"transformer.layers.{i}.mlp.fc2.weight"],
            rtol=rtol,
            atol=atol * 10,
        )
        if rank == 0:
            assert torch.allclose(
                model.transformer.layers[i].mlp.fc2.bias.grad,
                grad_dict[f"transformer.layers.{i}.mlp.fc2.bias"],
                rtol=rtol,
                atol=atol * 5,
            )

        assert torch.allclose(
            model.transformer.layers[i].norm1.weight.grad,
            grad_dict[f"transformer.layers.{i}.norm1.weight"],
            rtol=rtol,
            atol=atol,
        )
        assert torch.allclose(
            model.transformer.layers[i].norm1.bias.grad,
            grad_dict[f"transformer.layers.{i}.norm1.bias"],
            rtol=rtol,
            atol=atol,
        )
        assert torch.allclose(
            model.transformer.layers[i].norm2.weight.grad,
            grad_dict[f"transformer.layers.{i}.norm2.weight"],
            rtol=rtol,
            atol=atol,
        )
        assert torch.allclose(
            model.transformer.layers[i].norm2.bias.grad,
            grad_dict[f"transformer.layers.{i}.norm2.bias"],
            rtol=rtol,
            atol=atol,
        )
