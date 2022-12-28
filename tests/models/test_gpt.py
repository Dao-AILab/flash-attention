import re

import torch
import pytest

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2LMHeadModelHF

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.gpt import remap_state_dict_gpt2
from flash_attn.utils.pretrained import state_dict_from_pretrained


@pytest.mark.parametrize('model_name', ["gpt2", "gpt2-medium"])
# @pytest.mark.parametrize('model_name', ["gpt2"])
def test_gpt2_state_dict(model_name):
    config = GPT2Config.from_pretrained(model_name)
    pretrained_state_dict = remap_state_dict_gpt2(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config)
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize('model_name', ["gpt2", "gpt2-medium"])
# @pytest.mark.parametrize('model_name', ["gpt2"])
def test_gpt2_non_optimized(model_name):
    """Check that our implementation of GPT2 (without any optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    config = GPT2Config.from_pretrained(model_name)

    model = GPTLMHeadModel.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = GPT2LMHeadModelHF.from_pretrained(model_name).cuda()
    model_hf = GPT2LMHeadModelHF.from_pretrained(model_name).cuda().to(dtype=dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device='cuda')
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long,
                              device='cuda')
    out = model.transformer(input_ids)
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    out_ref = model_ref.transformer(input_ids).last_hidden_state

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


@pytest.mark.parametrize('model_name', ["gpt2", "gpt2-medium"])
# @pytest.mark.parametrize('model_name', ["gpt2"])
def test_gpt2_optimized(model_name):
    """Check that our implementation of GPT2 (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    config = GPT2Config.from_pretrained(model_name)
    vocab_size_og = config.vocab_size
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_dense_gelu_dense = True
    config.fused_dropout_add_ln = True
    config.pad_vocab_size_multiple = 8

    model = GPTLMHeadModel.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = GPT2LMHeadModelHF.from_pretrained(model_name).cuda()
    model_hf = GPT2LMHeadModelHF.from_pretrained(model_name).cuda().to(dtype=dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device='cuda')
    input_ids = torch.randint(0, vocab_size_og, (batch_size, max_seqlen), dtype=torch.long,
                              device='cuda')
    out = model.transformer(input_ids)
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    out_ref = model_ref.transformer(input_ids).last_hidden_state

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}')
    assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    logits = model(input_ids).logits[..., :vocab_size_og]
    logits_hf = model_hf(input_ids).logits
    logits_ref = model_ref(input_ids).logits

    print(f'Logits max diff: {(logits - logits_ref).abs().max().item()}')
    print(f'Logits mean diff: {(logits - logits_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}')
    assert (logits - logits_ref).abs().max().item() < 3 * (logits_hf - logits_ref).abs().max().item()
