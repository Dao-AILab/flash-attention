import re

import torch
import pytest

from transformers import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.opt import remap_state_dict_hf_opt, opt_config_to_gpt2_config
from flash_attn.utils.pretrained import state_dict_from_pretrained


@pytest.mark.parametrize('model_name', ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"])
# @pytest.mark.parametrize('model_name', ["facebook/opt-350m"])
def test_opt_state_dict(model_name):
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    pretrained_state_dict = remap_state_dict_hf_opt(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config)
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize('model_name', ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"])
# @pytest.mark.parametrize('model_name', ["facebook/opt-350m"])
def test_opt_optimized(model_name):
    """Check that our implementation of OPT (without all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = 'cuda'
    config = opt_config_to_gpt2_config(OPTConfig.from_pretrained(model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    # Only prenorm supports residual_in_fp32
    config.residual_in_fp32 = getattr(config, 'prenorm', True)
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
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device='cuda')
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long,
                              device='cuda')
    if model_name != 'facebook/opt-350m':  # The OPT-350m projects the embeddings to dimension 512
        out = model.transformer(input_ids)
        out_hf = model_hf.model(input_ids).last_hidden_state
        out_ref = model_ref.model(input_ids).last_hidden_state

        print(f'Output max diff: {(out - out_ref).abs().max().item()}')
        print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
        print(f'HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}')
        print(f'HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}')
        assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    logits = model(input_ids).logits
    logits_hf = model_hf(input_ids).logits
    logits_ref = model_ref(input_ids).logits

    print(f'Logits max diff: {(logits - logits_ref).abs().max().item()}')
    print(f'Logits mean diff: {(logits - logits_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}')
    assert (logits - logits_ref).abs().max().item() < 3 * (logits_hf - logits_ref).abs().max().item()
