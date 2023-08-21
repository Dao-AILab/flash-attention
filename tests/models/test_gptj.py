# Copyright (c) 2023, Tri Dao.

import time

import pytest
import torch
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.gptj import gptj_config_to_gpt2_config, remap_state_dict_hf_gptj
from flash_attn.utils.generation import update_graph_cache
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import AutoTokenizer, GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM


@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-j-6B"])
def test_gptj_state_dict(model_name):
    config = gptj_config_to_gpt2_config(GPTJConfig.from_pretrained(model_name))
    pretrained_state_dict = remap_state_dict_hf_gptj(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config, device="meta")  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-j-6B"])
def test_gptj_optimized(model_name):
    """Check that our implementation of GPT-J (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = gptj_config_to_gpt2_config(GPTJConfig.from_pretrained(model_name))
    config.use_flash_attn = True  # FlashAttention-2 supports headdim 256
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    batch_size = 2
    max_seqlen = 256
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = model.transformer(input_ids)
        logits = model(input_ids).logits
    del model

    # Without device_map, the model is loaded on the CPU, which is very slow
    model_ref = GPTJForCausalLM.from_pretrained(model_name, device_map={"": device})
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.transformer(input_ids).last_hidden_state
        logits_ref = model_ref(input_ids).logits
    del model_ref

    model_hf = GPTJForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}
    )
    model_hf.eval()
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    logits_hf = model_hf(input_ids).logits
    del model_hf

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}")
    assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
    print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}")
    assert (logits - logits_ref).abs().max().item() < 3 * (
        logits_hf - logits_ref
    ).abs().max().item()


@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-j-6B"])
def test_gptj_generation(model_name):
    """Check that our implementation of GPT-J (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = gptj_config_to_gpt2_config(GPTJConfig.from_pretrained(model_name))
    config.use_flash_attn = True  # FlashAttention-2 supports headdim 256
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = tokenizer.eos_token_id

    torch.manual_seed(0)
    batch_size = 1
    seqlen = 100
    max_length = 150
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )

    model_hf = GPTJForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}
    )
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

    model_ref = GPTJForCausalLM.from_pretrained(model_name, device_map={"": device})
    model_ref.eval()
    with torch.no_grad():
        logits_ref = model_ref(out_hf.sequences).logits[:, (seqlen - 1) : -1]
    del model_ref

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    print("Without CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        fused_ft_kernel=True,
        # eos_token_id=eos_token_id, fused_ft_kernel=False,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
        teacher_outputs=out_hf.sequences,
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")

    # Capture graph outside the timing loop
    batch_size, seqlen_og = input_ids.shape
    model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
    print("With CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out_cg = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        fused_ft_kernel=True,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
        teacher_outputs=out_hf.sequences,
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")

    with torch.no_grad():
        logits_parallel = model(out_hf.sequences).logits[:, (seqlen - 1) : -1]
    logits_hf = torch.stack(out_hf.scores, dim=1)
    logits = torch.stack(out.scores, dim=1)
    logits_cg = torch.stack(out_cg.scores, dim=1)

    del model

    hf_error = (logits_hf - logits_ref).abs().max().item()
    assert (logits_parallel - logits_ref).abs().max().item() < 2 * hf_error

    print(f"HF fp16 logits max diff: {hf_error}")
    print(f"Logits max diff: {(logits - logits_ref).abs().max().item() }")
    assert (logits - logits_ref).abs().max().item() < 2 * hf_error
    print(f"Logits CG max diff: {(logits_cg - logits_ref).abs().max().item() }")
    assert torch.equal(logits_cg, logits)
