import re

import torch
import pytest

from transformers import GPTJConfig, AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.gptj import remap_state_dict_hf_gptj, gptj_config_to_gpt2_config
from flash_attn.utils.pretrained import state_dict_from_pretrained


@pytest.mark.parametrize('model_name', ["EleutherAI/gpt-j-6B"])
def test_gptj_state_dict(model_name):
    config = gptj_config_to_gpt2_config(GPTJConfig.from_pretrained(model_name))
    pretrained_state_dict = remap_state_dict_hf_gptj(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config, device='meta')  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    rotary_inv_freq_keys = {f'transformer.layers.{l}.mixer.rotary_emb.inv_freq'
                            for l in range(config.n_layer)}
    assert state_dict.keys() == pretrained_state_dict.keys() | rotary_inv_freq_keys
    for k in state_dict.keys() - rotary_inv_freq_keys:
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize('model_name', ["EleutherAI/gpt-j-6B"])
def test_gptj_optimized(model_name):
    """Check that our implementation of GPT-J (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = 'cuda'
    config = gptj_config_to_gpt2_config(GPTJConfig.from_pretrained(model_name))
    config.use_flash_attn = False  # FlashAttention doesn't support hdim 256 yet
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    batch_size = 2
    max_seqlen = 256
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device=device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long,
                              device=device)
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

    model_hf = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                               device_map={"": device})
    model_hf.eval()
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    logits_hf = model_hf(input_ids).logits
    del model_hf

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}')
    assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    print(f'Logits max diff: {(logits - logits_ref).abs().max().item()}')
    print(f'Logits mean diff: {(logits - logits_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}')
    assert (logits - logits_ref).abs().max().item() < 3 * (logits_hf - logits_ref).abs().max().item()
