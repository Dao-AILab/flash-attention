# Copyright (c) 2023, Tri Dao.

import time

import pytest
import torch
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.gpt_neox import gpt_neox_config_to_gpt2_config, remap_state_dict_hf_gpt_neox
from flash_attn.utils.generation import update_graph_cache
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import AutoTokenizer, GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM


@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-neox-20b"])
def test_gptj_state_dict(model_name):
    config = gpt_neox_config_to_gpt2_config(GPTNeoXConfig.from_pretrained(model_name))
    pretrained_state_dict = remap_state_dict_hf_gpt_neox(
        state_dict_from_pretrained(model_name), config
    )
    model = GPTLMHeadModel(config, device="meta")  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-neox-20b"])
def test_gpt_neox_optimized(model_name):
    """Check that our implementation of GPT-NeoX (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = gpt_neox_config_to_gpt2_config(GPTNeoXConfig.from_pretrained(model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True  # GPT-NeoX-20B uses "gelu_fast"
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    batch_size = 2
    max_seqlen = 256
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device=device)
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = model.transformer(input_ids)
        logits = model(input_ids).logits
    del model

    # Need at least 2 GPUs, otherwise we'll OOM
    # Without device_map, the model is loaded on the CPU, which is very slow
    model_ref = GPTNeoXForCausalLM.from_pretrained(model_name, device_map="auto")
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.gpt_neox(input_ids).last_hidden_state.to(device=device)
        logits_ref = model_ref(input_ids).logits.to(device=device)
    del model_ref

    model_hf = GPTNeoXForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}
    )
    model_hf.eval()
    with torch.no_grad():
        out_hf = model_hf.gpt_neox(input_ids).last_hidden_state
        logits_hf = model_hf(input_ids).logits
    del model_hf

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}")
    assert (out - out_ref).abs().max().item() < 2 * (out_hf - out_ref).abs().max().item()
    assert (out - out_ref).abs().mean().item() < 2 * (out_hf - out_ref).abs().mean().item()

    print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
    print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}")
    assert (logits - logits_ref).abs().max().item() < 2 * (
        logits_hf - logits_ref
    ).abs().max().item()
    assert (logits - logits_ref).abs().mean().item() < 2 * (
        logits_hf - logits_ref
    ).abs().mean().item()
