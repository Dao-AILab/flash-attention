import re

import pytest
import torch
from einops import rearrange
from flash_attn.models.gpt import GPTLMHeadModel, remap_state_dict_hf_gpt2, shard_state_dict_tp, combine_state_dicts_tp
from flash_attn.utils.generation import InferenceParams
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import GPT2Config, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2LMHeadModelHF


@pytest.mark.parametrize("model_name", ["gpt2", "gpt2-medium"])
# @pytest.mark.parametrize('model_name', ["gpt2"])
def test_gpt2_state_dict(model_name):
    config = GPT2Config.from_pretrained(model_name)
    pretrained_state_dict = remap_state_dict_hf_gpt2(state_dict_from_pretrained(model_name), config)
    model = GPTLMHeadModel(config)
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize("model_name", ["gpt2", "gpt2-medium"])
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
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    out = model.transformer(input_ids)
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    out_ref = model_ref.transformer(input_ids).last_hidden_state

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


@pytest.mark.parametrize("model_name", ["gpt2", "gpt2-medium"])
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
    config.fused_mlp = True
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True
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
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    input_ids = torch.randint(
        0, vocab_size_og, (batch_size, max_seqlen), dtype=torch.long, device="cuda"
    )
    out = model.transformer(input_ids)
    out_hf = model_hf.transformer(input_ids).last_hidden_state
    out_ref = model_ref.transformer(input_ids).last_hidden_state

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}")
    assert (out - out_ref).abs().max().item() < 3 * (out_hf - out_ref).abs().max().item()

    logits = model(input_ids).logits[..., :vocab_size_og]
    logits_hf = model_hf(input_ids).logits
    logits_ref = model_ref(input_ids).logits

    print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
    print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}")
    assert (logits - logits_ref).abs().max().item() < 3 * (
        logits_hf - logits_ref
    ).abs().max().item()


@pytest.mark.parametrize("fused_ft_kernel", [False, True])
# @pytest.mark.parametrize('fused_ft_kernel', [True])
@pytest.mark.parametrize("optimized", [False, True])
# @pytest.mark.parametrize('optimized', [False])
@pytest.mark.parametrize("rotary", [False, True])
# @pytest.mark.parametrize('rotary', [False])
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_gpt2_generation(model_name, rotary, optimized, fused_ft_kernel):
    """Check that our implementation of GPT2 generation matches the HF implementation:
    the scores in fp16 should be around the same as the HF scores in fp16, when compared to
    the HF scores in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    if rotary:
        config.n_positions = 0
        config.rotary_emb_fraction = 0.5
        config.rotary_emb_base = 24000
    config.residual_in_fp32 = True
    if optimized:
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = True
        config.fused_dropout_add_ln = True

    # if not rotary, we load the weight from HF but ignore the position embeddings.
    # The model would be nonsense but it doesn't matter for the test.
    model = GPTLMHeadModel.from_pretrained(
        model_name, config, strict=not rotary, device=device, dtype=dtype
    )
    model.eval()

    if not rotary:
        model_ref = GPT2LMHeadModelHF.from_pretrained(model_name).to(device=device)
        model_hf = GPT2LMHeadModelHF.from_pretrained(model_name, torch_dtype=dtype).to(
            device=device
        )
        model_ref.eval()
        model_hf.eval()

    torch.manual_seed(0)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    sequences = torch.cat([input_ids, torch.stack(sequences, dim=1)], dim=1)
    scores = tuple(scores)

    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        fused_ft_kernel=fused_ft_kernel,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
    )
    print(out.sequences)
    print(tokenizer.batch_decode(out.sequences.tolist()))
    if fused_ft_kernel:
        out_cg = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            fused_ft_kernel=fused_ft_kernel,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            timing=True,
        )
        print(out_cg.sequences)

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
        print(tokenizer.batch_decode(out_ref.sequences.tolist()))

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


def get_logits(model, input_ids, max_length, teacher_outputs=None, **kwargs):
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        fused_ft_kernel=True,
        teacher_outputs=teacher_outputs,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
        **kwargs,
    )
    return torch.stack(out.scores, dim=1)


@pytest.mark.parametrize("seqlen,maxlen", [(10, 20), (30, 150), (3000, 3400), (14000, 15000)])
# @pytest.mark.parametrize('seqlen,maxlen', [(10, 20)])
@pytest.mark.parametrize("rotary", [None, "interleaved", "block"])
# @pytest.mark.parametrize('rotary', [None])
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_gpt2_generation_cg(model_name, rotary, seqlen, maxlen):
    """Check that decoding with CUDA graph is the same as decoding without CUDA graph."""
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    config.n_positions = 16 * 1024
    assert seqlen <= maxlen <= config.n_positions
    if rotary is not None:
        config.n_positions = 0
        config.rotary_emb_dim = 32
        config.rotary_emb_interleaved = rotary == "interleaved"
    config.residual_in_fp32 = True
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = True
    config.fused_dropout_add_ln = True

    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    batch_size = 1
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )
    teacher_outputs = torch.randint(
        0, config.vocab_size, (batch_size, maxlen), dtype=torch.long, device=device
    )

    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)

    # Try increasing batch size and seqlen, then decrease them to see if it's still correct
    batch_size = 3
    maxlen += 30
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )
    teacher_outputs = torch.randint(
        0, config.vocab_size, (batch_size, maxlen), dtype=torch.long, device=device
    )
    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)

    batch_size = 2
    maxlen -= 35
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )
    teacher_outputs = torch.randint(
        0, config.vocab_size, (batch_size, maxlen), dtype=torch.long, device=device
    )
    logits = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs)
    logits_cg = get_logits(model, input_ids, maxlen, teacher_outputs=teacher_outputs, cg=True)
    assert torch.equal(logits, logits_cg)


@pytest.mark.parametrize("optimized", [False, True])
# @pytest.mark.parametrize("optimized", [False])
@pytest.mark.parametrize("model_name", ["gpt2"])
def test_gpt2_multiple_token_generation(model_name, optimized):
    """Generation when we pass in multiple tokens at a time, not just one."""
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    config.residual_in_fp32 = True
    if optimized:
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = True
        config.fused_dropout_add_ln = True
    # fused_ft_kernel currently doesn't work with multiple tokens at a time

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, 20), dtype=torch.long, device=device)
    # Reference logits
    logits_ref = model(input_ids).logits

    # Run 10 tokens, then pass in another 4, then another 6, to see if we get the same logits
    inference_params = InferenceParams(max_sequence_len=20, max_batch_size=1)
    logits_10 = model(input_ids[:, :10], inference_params=inference_params).logits
    inference_params.sequence_len_offset += 10
    position_ids = torch.arange(10, 14, dtype=torch.long, device=device)
    logits_1014 = model(
        input_ids[:, 10:14], position_ids=position_ids, inference_params=inference_params
    ).logits
    inference_params.sequence_len_offset += 4
    position_ids = torch.arange(14, 20, dtype=torch.long, device=device)
    logits_1420 = model(
        input_ids[:, 14:20], position_ids=position_ids, inference_params=inference_params
    ).logits
    logits = torch.cat([logits_10, logits_1014, logits_1420], dim=1)
    print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
    print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
    assert torch.allclose(logits, logits_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("fused_ft_kernel, cg", [(False, False), (True, False), (True, True)])
# @pytest.mark.parametrize("fused_ft_kernel, cg", [(True, True)])
# @pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("optimized", [True])
# @pytest.mark.parametrize("model_name", ["gpt2-medium"])
@pytest.mark.parametrize("model_name", ["gpt2-xl"])
def test_gpt2_speculative_decoding(model_name, optimized, fused_ft_kernel, cg):
    dtype = torch.float16
    device = "cuda"
    rtol, atol = 3e-3, 3e-1
    config = GPT2Config.from_pretrained(model_name)
    config.residual_in_fp32 = True
    if optimized:
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = True
        config.fused_dropout_add_ln = True
    config_draft = GPT2Config.from_pretrained("gpt2")
    config_draft.residual_in_fp32 = True
    if optimized:
        config_draft.use_flash_attn = True
        config_draft.fused_bias_fc = True
        config_draft.fused_mlp = True
        config_draft.fused_dropout_add_ln = True

    model = GPTLMHeadModel.from_pretrained(model_name, config, device=device, dtype=dtype)
    model.eval()
    model_draft = GPTLMHeadModel.from_pretrained("gpt2", config_draft, device=device, dtype=dtype)
    model_draft.eval()

    torch.manual_seed(0)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Hello, my dog is cute and he", return_tensors="pt").input_ids.to(
        device=device
    )
    max_length = 100

    from flash_attn.utils.generation import decode_speculative

    torch.manual_seed(42)
    out = decode_speculative(
        input_ids,
        model,
        model_draft,
        max_length=max_length,
        top_k=5,
        fused_ft_kernel=fused_ft_kernel,
        cg=cg,
        speculative_lookahead=4,
        timing=True,
    )
    print(tokenizer.batch_decode(out.sequences))
    out_og = model.generate(
        input_ids,
        max_length=max_length,
        top_k=5,
        fused_ft_kernel=fused_ft_kernel,
        cg=False,
        timing=True,
        return_dict_in_generate=True,
    )
    print(tokenizer.batch_decode(out_og.sequences))


@pytest.mark.parametrize("n_heads_q_kv", [
    (8, 8), # Regular attention
    (8, 4), # GQA
    (8, 2), # MQA
])
def test_gpt2_shard_unshard(n_heads_q_kv):
    world_size = 2

    config = GPT2Config.from_pretrained("gpt2")
    config.vocab_size = 1024
    config.n_head, config.n_head_kv = n_heads_q_kv
    model = GPTLMHeadModel(config, device="cuda", dtype=torch.float16)
    state_dict = model.state_dict()
    shards = [
        # NOTE: Shallow copy as `state_dict` is modified in-place
        shard_state_dict_tp(dict(state_dict), config, world_size, rank)
        for rank in range(world_size)
    ]
    state_dict2 = combine_state_dicts_tp(shards, config)
    assert state_dict2.keys() == state_dict.keys()
    for k in state_dict.keys():
        ref = state_dict[k]
        new = state_dict[k]
        assert torch.allclose(ref, new, atol=0.0, rtol=0.0)
