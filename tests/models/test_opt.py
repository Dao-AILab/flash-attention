import re
import time

import pytest
import torch
from einops import rearrange
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.opt import opt_config_to_gpt2_config, remap_state_dict_hf_opt
from flash_attn.utils.generation import update_graph_cache
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import AutoTokenizer, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM


@pytest.mark.parametrize(
    "model_name", ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]
)
# @pytest.mark.parametrize('model_name', ["facebook/opt-350m"])
def test_opt_state_dict(model_name):
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    pretrained_state_dict = remap_state_dict_hf_opt(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config)
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize(
    "model_name", ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]
)
# @pytest.mark.parametrize('model_name', ["facebook/opt-350m"])
def test_opt_optimized(model_name):
    """Check that our implementation of OPT (without all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.pad_vocab_size_multiple = 8

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)

    model_ref = OPTForCausalLM.from_pretrained(model_name).to(device=device)
    model_hf = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device=device)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 2
    max_seqlen = 256
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    if model_name != "facebook/opt-350m":  # The OPT-350m projects the embeddings to dimension 512
        out = model.transformer(input_ids)
        out_hf = model_hf.model(input_ids).last_hidden_state
        out_ref = model_ref.model(input_ids).last_hidden_state

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(f"HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}")
        print(f"HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}")
        assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    logits = model(input_ids).logits
    logits_hf = model_hf(input_ids).logits
    logits_ref = model_ref(input_ids).logits

    print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
    print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}")
    assert (logits - logits_ref).abs().max().item() < 3 * (
        logits_hf - logits_ref
    ).abs().max().item()


@pytest.mark.parametrize(
    "model_name",
    [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
    ],
)
# @pytest.mark.parametrize('model_name', ["facebook/opt-125m"])
def test_opt_generation(model_name):
    """Check that our implementation of OPT generation matches the HF implementation:
    the scores in fp16 should be around the same as the HF scores in fp16, when compared to
    the HF scores in fp32.
    """
    print(f"\nMODEL: {model_name}")
    verbose = False
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    fused_ft_kernel = True
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, "prenorm", True)
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    # OPT tokenizer requires use_fast=False
    # https://huggingface.co/docs/transformers/model_doc/opt
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    eos_token_id = tokenizer.eos_token_id

    input_ids = tokenizer("Hello, my dog is cute and he", return_tensors="pt").input_ids.to(
        device=device
    )
    max_length = 25
    # input_ids = torch.randint(0, 100, (2, 10), dtype=torch.long, device='cuda')
    # max_length = input_ids.shape[1] + 40

    # Slow generation for reference
    sequences = []
    scores = []
    cur_input_ids = input_ids
    with torch.inference_mode():
        scores.append(model(cur_input_ids).logits[:, -1])
        sequences.append(scores[-1].argmax(dim=-1))
        for _ in range(input_ids.shape[1] + 1, max_length):
            cur_input_ids = torch.cat([cur_input_ids, rearrange(sequences[-1], "b -> b 1")], dim=-1)
            scores.append(model(cur_input_ids).logits[:, -1])
            sequences.append(scores[-1].argmax(dim=-1))
            if eos_token_id is not None and (sequences[-1] == eos_token_id).all():
                break
    sequences = torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1)
    scores = tuple(scores)

    print("Without CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        fused_ft_kernel=fused_ft_kernel,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    if verbose:
        print(out.sequences)
    print(tokenizer.batch_decode(out.sequences.tolist()))
    if fused_ft_kernel:
        # Capture graph outside the timing loop
        batch_size, seqlen_og = input_ids.shape
        model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
        print("With CUDA graph")
        torch.cuda.synchronize()
        start = time.time()
        out_cg = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            fused_ft_kernel=fused_ft_kernel,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            timing=True,
        )
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
        if verbose:
            print(out_cg.sequences)
        print(tokenizer.batch_decode(out_cg.sequences.tolist()))

    del model

    model_hf = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device=device)
    model_hf.eval()
    print("HF fp16")
    torch.cuda.synchronize()
    start = time.time()
    out_hf = model_hf.generate(
        input_ids=input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    del model_hf

    model_ref = OPTForCausalLM.from_pretrained(model_name).to(device=device)
    model_ref.eval()
    print("HF fp32")
    torch.cuda.synchronize()
    start = time.time()
    out_ref = model_ref.generate(
        input_ids=input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    del model_ref
    print(tokenizer.batch_decode(out_ref.sequences.tolist()))

    if verbose:
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
    assert torch.all(out.sequences == out_ref.sequences)
    assert torch.all(out.sequences == out_hf.sequences)

    assert (torch.stack(out.scores, 1) - torch.stack(out_ref.scores, 1)).abs().max().item() < 3 * (
        torch.stack(out_hf.scores, 1) - torch.stack(out_ref.scores, 1)
    ).abs().max().item()
