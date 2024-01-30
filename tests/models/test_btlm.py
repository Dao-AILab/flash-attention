# Copyright (c) 2023, Tri Dao.
import time

import torch
import pytest

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.btlm import btlm_config_to_gpt2_config, remap_state_dict_hf_btlm
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache


@pytest.mark.parametrize("model_name", ["cerebras/btlm-3b-8k-base"])
def test_btlm_state_dict(model_name):
    config = btlm_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    pretrained_state_dict = remap_state_dict_hf_btlm(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config, device="meta")  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    assert len(state_dict.keys()) == len(pretrained_state_dict.keys())
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize("model_name", ["cerebras/btlm-3b-8k-base"])
def test_btlm_optimized(model_name):
    """Check that our implementation of Btlm (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = btlm_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.fused_bias_fc = True
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    pretrained_state_dict = remap_state_dict_hf_btlm(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(pretrained_state_dict)
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

    # Without device_map, the model is loaded on the CPU, which is very slow
    # Need auto here since the 13B fp32 model doesn't fit in memory on a A100 40GB
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.transformer(input_ids).last_hidden_state.to(device=device)
        logits_ref = model_ref(input_ids).logits.to(device=device)
    del model_ref

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model_hf.eval()
    with torch.no_grad():
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


@pytest.mark.parametrize("model_name", ["cerebras/btlm-3b-8k-base"])
def test_btlm_generation(model_name):
    dtype = torch.float16
    device = "cuda"
    config = btlm_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.fused_bias_fc = True
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    torch.manual_seed(0)
    batch_size = 1
    seqlen = 2048
    max_length = 2048 + 150
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    model_hf.eval()
    print("HF fp16")
    torch.cuda.synchronize()
    start = time.time()
    out_hf = model_hf.generate(
        input_ids=input_ids,
        max_length=max_length,
        return_dict_in_generate=True,
        output_scores=True,
    )
    torch.cuda.synchronize()
    print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
    del model_hf

    # Need auto here since the 13B fp32 model doesn't fit in memory on a A100 40GB
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model_ref.eval()
    with torch.no_grad():
        logits_ref = model_ref(out_hf.sequences).logits[:, (seqlen - 1) : -1].to(device=device)
    del model_ref

    pretrained_state_dict = remap_state_dict_hf_btlm(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    model(input_ids)  # Warm up
    print("Without CUDA graph")
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=True,
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
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=True,
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

    print(f"HF fp16 logits max diff: {hf_error}")
    print(f"Logits max diff: {(logits - logits_ref).abs().max().item() }")
    print(f"Logits CG max diff: {(logits_cg - logits_ref).abs().max().item() }")

    assert (logits_parallel - logits_ref).abs().max().item() < 2 * hf_error
    assert (logits - logits_ref).abs().max().item() < 2 * hf_error
    assert torch.equal(logits_cg, logits)


@pytest.mark.parametrize("model_name", ["cerebras/btlm-3b-8k-base"])
def test_btlm_init(model_name):
    dtype = torch.float32
    device = "cuda"
    btlm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = btlm_config_to_gpt2_config(btlm_config)
    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model_ref = AutoModelForCausalLM.from_config(btlm_config, trust_remote_code=True).to(device)

    assert model.transformer.embeddings.word_embeddings.weight.mean().abs() < 1e-4
    assert (
        model.transformer.embeddings.word_embeddings.weight.std()
        - model_ref.transformer.wte.weight.std()
    ).abs() < 1e-4
    assert model.lm_head.weight.mean().abs() < 1e-4
    assert (model.lm_head.weight.std() - model_ref.lm_head.weight.std()).abs() < 1e-4
    for l in range(config.n_layer):
        assert model.transformer.layers[l].mixer.Wqkv.weight.mean().abs() < 1e-4
        assert (
            model.transformer.layers[l].mixer.Wqkv.weight.std()
            - model_ref.transformer.h[l].attn.c_attn.weight.std()
        ).abs() < 1e-4
        assert model.transformer.layers[l].mixer.Wqkv.bias.abs().max() == 0.0
        assert model.transformer.layers[l].mixer.out_proj.weight.mean().abs() < 1e-4
        assert (
            model.transformer.layers[l].mixer.out_proj.weight.std()
            - model_ref.transformer.h[l].attn.c_proj.weight.std()
        ).abs() < 1e-4
        assert model.transformer.layers[l].mixer.out_proj.bias.abs().max() == 0.0
        assert model.transformer.layers[l].mlp.fc1.weight.mean().abs() < 1e-4
        assert (
            model.transformer.layers[l].mlp.fc1.weight.std()
            - model_ref.transformer.h[l].mlp.c_fc.weight.std()
        ).abs() < 1e-4
        assert model.transformer.layers[l].mlp.fc1.bias.abs().max() == 0.0
        assert model.transformer.layers[l].mlp.fc2.weight.mean().abs() < 1e-4
        assert (
            model.transformer.layers[l].mlp.fc2.weight.std()
            - model_ref.transformer.h[l].mlp.c_proj.weight.std()
        ).abs() < 1e-4
        assert model.transformer.layers[l].mlp.fc2.bias.abs().max() == 0.0
