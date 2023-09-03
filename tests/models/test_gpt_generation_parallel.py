# Run test with:
# torchrun --no_python --nproc_per_node=8 pytest -q -s tests/models/test_gpt_generation_parallel.py -k "parallel"
import os
import re

import pytest
import torch
from einops import rearrange
from flash_attn.models.gpt import GPTLMHeadModel, remap_state_dict_hf_gpt2
from flash_attn.utils.distributed import all_gather_raw
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import GPT2Config, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2LMHeadModelHF


# @pytest.mark.parametrize('world_size', [1, 2, 4, 8])
@pytest.mark.parametrize("world_size", [2])
# @pytest.mark.parametrize('fused_ft_kernel', [False, True])
@pytest.mark.parametrize("fused_ft_kernel", [True])
# @pytest.mark.parametrize('rotary', [False, True])
@pytest.mark.parametrize("rotary", [False])
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_tensor_parallel(model_name, rotary, fused_ft_kernel, world_size):
    """Check that our implementation of GPT2 generation matches the HF implementation:
    the scores in fp16 should be around the same as the HF scores in fp16, when compared to
    the HF scores in fp32.
    """
    dtype = torch.float16
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    if rotary:
        config.n_positions = 0
        config.rotary_emb_dim = 64
    config.residual_in_fp32 = True
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    config.pad_vocab_size_multiple = 8 * world_size
    config.sequence_parallel = False  # Need to set this to False for generation

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    # Need this, otherwise when we capture the graph the process for GPU 1 would run on both
    # GPU0 and GPU1 and things would hang
    torch.cuda.set_device(device)

    from apex.transformer import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()

    # if not rotary, we load the weight from HF but ignore the position embeddings.
    # The model would be nonsense but it doesn't matter for the test.
    model = GPTLMHeadModel.from_pretrained(
        model_name,
        config,
        strict=not rotary,
        device=device,
        dtype=dtype,
        process_group=process_group,
        world_size=world_size,
        rank=rank,
    )
    model.eval()

    if not rotary:
        model_ref = GPT2LMHeadModelHF.from_pretrained(model_name).to(device=device)
        model_hf = GPT2LMHeadModelHF.from_pretrained(model_name).to(device=device, dtype=dtype)
        model_ref.eval()
        model_hf.eval()

    torch.manual_seed(0)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Hello, my dog is cute and ", return_tensors="pt").input_ids.to(
        device=device
    )
    max_length = 30
    # input_ids = torch.randint(0, 100, (1, 10), dtype=torch.long, device='cuda')
    # max_length = input_ids.shape[1] + 40

    # Slow generation for reference
    sequences = []
    scores = []
    cur_input_ids = input_ids
    with torch.inference_mode():
        logits, _ = all_gather_raw(model(cur_input_ids).logits[:, -1], process_group)
        logits = rearrange(logits, "(n b) d -> b (n d)", b=input_ids.shape[0])[
            ..., : config.vocab_size
        ]
        scores.append(logits)
        sequences.append(scores[-1].argmax(dim=-1))
        for _ in range(input_ids.shape[1] + 1, max_length):
            cur_input_ids = torch.cat([cur_input_ids, rearrange(sequences[-1], "b -> b 1")], dim=-1)
            logits, _ = all_gather_raw(model(cur_input_ids).logits[:, -1], process_group)
            logits = rearrange(logits, "(n b) d -> b (n d)", b=input_ids.shape[0])[
                ..., : config.vocab_size
            ]
            scores.append(logits)
            sequences.append(scores[-1].argmax(dim=-1))
    sequences = torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1)
    scores = tuple(scores)
    print(sequences)

    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        tensor_parallel=world_size,
        vocab_size=config.vocab_size,
        fused_ft_kernel=fused_ft_kernel,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
    )
    print(out.sequences)
    if fused_ft_kernel:
        out_cg = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            tensor_parallel=world_size,
            vocab_size=config.vocab_size,
            fused_ft_kernel=fused_ft_kernel,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            timing=True,
        )
        print(out_cg.sequences)

    parallel_state.destroy_model_parallel()

    if not rotary:
        out_hf = model_hf.generate(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
        out_ref = model_ref.generate(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )

        print(
            f"Scores max diff: {(torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item()}"
        )
        print(
            f"Scores mean diff: {(torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().mean().item()}"
        )
        print(
            f"HF fp16 max diff: {(torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item()}"
        )
        print(
            f"HF fp16 mean diff: {(torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)).abs().mean().item()}"
        )

    assert torch.all(out.sequences == sequences)
    assert torch.allclose(
        torch.stack(out.scores, dim=1), torch.stack(scores, dim=1), rtol=rtol, atol=atol
    )
    if not rotary:
        assert torch.all(out.sequences == out_ref.sequences)
        assert torch.all(out.sequences == out_hf.sequences)

        assert (
            torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)
        ).abs().max().item() < 3 * (
            torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)
        ).abs().max().item()
