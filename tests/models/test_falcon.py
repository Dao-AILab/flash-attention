# Copyright (c) 2023, Tri Dao.

import os
import time
from pathlib import Path

current_dir = Path(__file__).parent.absolute()

import pytest
import torch
from einops import rearrange
from flash_attn.models.falcon import falcon_config_to_gpt2_config, remap_state_dict_hf_falcon
from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp, shard_state_dict_tp
from flash_attn.utils.distributed import all_gather_raw
from flash_attn.utils.generation import update_graph_cache
from flash_attn.utils.pretrained import state_dict_from_pretrained
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b", "tiiuae/falcon-40b"])
def test_falcon_state_dict(model_name):
    config = falcon_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    pretrained_state_dict = remap_state_dict_hf_falcon(
        state_dict_from_pretrained(model_name), config
    )
    model = GPTLMHeadModel(config, device="meta")  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    assert state_dict.keys() == pretrained_state_dict.keys()
    for k in state_dict.keys():
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b"])
def test_falcon_optimized(model_name):
    """Check that our implementation (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = falcon_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused MLP for "gelu" activation
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
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, trust_remote_code=True
    )
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.transformer(input_ids).last_hidden_state.to(device=device)
        logits_ref = model_ref(input_ids).logits.to(device=device)
    del model_ref

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
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


# torchrun --no_python --nproc_per_node=4 pytest -q -s tests/models/test_falcon.py -k "falcon_parallel_forward"
# We want to run this on a machine with 4 x A100 80GB or 8 x A100 40GB so we have enough
# memory to run the model in fp32.
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("model_name", ["tiiuae/falcon-40b"])
def test_falcon_parallel_forward(model_name, world_size):
    from apex.transformer import parallel_state

    dtype = torch.float16
    config = falcon_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.use_flash_attn = False
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused MLP for "gelu" activation
    config.fused_dropout_add_ln = False
    config.residual_in_fp32 = True

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()

    pretrained_state_dict = remap_state_dict_hf_falcon(
        state_dict_from_pretrained(model_name), config
    )

    model = GPTLMHeadModel(config, process_group=process_group, device=device, dtype=dtype)
    model.load_state_dict(shard_state_dict_tp(pretrained_state_dict, config, world_size, rank))
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
        out, _ = all_gather_raw(out, process_group=process_group)
        out = rearrange(out, "(b s) d -> b s d", b=batch_size)
        logits = model(input_ids).logits
        logits = rearrange(logits, "(b s) d -> b s d", b=batch_size)
        logits, _ = all_gather_raw(logits, process_group)
        logits = rearrange(logits, "(n b) ... d -> b ... (n d)", b=batch_size)
    del model

    if rank == 0:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model_hf.eval()
        out_hf = model_hf.transformer(input_ids).last_hidden_state.to(device=device)
        logits_hf = model_hf(input_ids).logits.to(device=device)
        del model_hf

        # Without device_map, the model is loaded on the CPU, which is very slow
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        model_ref.eval()
        with torch.no_grad():
            out_ref = model_ref.transformer(input_ids).last_hidden_state.to(device=device)
            logits_ref = model_ref(input_ids).logits.to(device=device)
        del model_ref

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(f"HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}")
        print(f"HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}")
        assert (out - out_ref).abs().max().item() < 2 * (out_hf - out_ref).abs().max().item()

        print(f"Logits max diff: {(logits - logits_ref).abs().max().item()}")
        print(f"Logits mean diff: {(logits - logits_ref).abs().mean().item()}")
        print(f"HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}")
        print(f"HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}")
        assert (logits - logits_ref).abs().max().item() < 2 * (
            logits_hf - logits_ref
        ).abs().max().item()


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b"])
def test_falcon_generation(model_name):
    """Check that our implementation (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"
    config = falcon_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused MLP for "gelu" activation
    config.fused_dropout_add_ln = True
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

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
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

    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, trust_remote_code=True
    )
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


# torchrun --no_python --nproc_per_node=4 pytest -q -s tests/models/test_falcon.py -k "falcon_parallel_generation"
# We want to run this on a machine with 4 x A100 80GB or 8 x A100 40GB so we have enough
# memory to run the model in fp32.
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("model_name", ["tiiuae/falcon-40b"])
def test_falcon_parallel_generation(model_name, world_size):
    """Check that our implementation matches the HF implementation:
    the scores in fp16 should be around the same as the HF scores in fp16, when compared to
    the HF scores in fp32.
    """
    from apex.transformer import parallel_state

    dtype = torch.float16
    config = falcon_config_to_gpt2_config(
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    )
    config.use_flash_attn = False
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused MLP for "gelu" activation
    config.fused_dropout_add_ln = False
    config.residual_in_fp32 = True
    config.pad_vocab_size_multiple = 8 * world_size
    config.sequence_parallel = False  # Need to set this to False for generation

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{torch.distributed.get_rank()}"
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()

    torch.manual_seed(0)
    batch_size = 1
    seqlen = 100
    max_length = 150
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seqlen), dtype=torch.long, device=device
    )

    # Need this, otherwise when we capture the graph the process for GPU 1 would run on both
    # GPU0 and GPU1 and things would hang
    torch.cuda.set_device(device)

    pretrained_state_dict = remap_state_dict_hf_falcon(
        state_dict_from_pretrained(model_name), config
    )

    model = GPTLMHeadModel(config, process_group=process_group, device=device, dtype=dtype)
    model.load_state_dict(shard_state_dict_tp(pretrained_state_dict, config, world_size, rank))
    model.eval()

    print("Without CUDA graph")
    out = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        tensor_parallel=world_size,
        vocab_size=config.vocab_size,
        fused_ft_kernel=True,
        # teacher_outputs=out_hf.sequences,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
    )

    # Capture graph outside the timing loop
    batch_size, seqlen_og = input_ids.shape
    model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
    print("With CUDA graph")
    out_cg = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        tensor_parallel=world_size,
        vocab_size=config.vocab_size,
        fused_ft_kernel=True,
        cg=True,
        # teacher_outputs=out_hf.sequences,
        return_dict_in_generate=True,
        output_scores=True,
        timing=True,
    )
    del model
    parallel_state.destroy_model_parallel()

    if rank == 0:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model_hf.eval()
        print("HF fp16")
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            out_hf = model_hf.generate(
                input_ids=input_ids,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")
        del model_hf

        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        model_ref.eval()
        with torch.inference_mode():
            logits_ref = model_ref(out_hf.sequences).logits[:, (seqlen - 1) : -1]
        del model_ref
        logits_hf = torch.stack(out_hf.scores, dim=1)

        logits = torch.stack(out.scores, dim=1)
        logits_cg = torch.stack(out_cg.scores, dim=1)

        hf_error = (logits_hf - logits_ref).abs().max().item()
        print(f"HF fp16 logits max diff: {hf_error}")
        print(f"Logits max diff: {(logits - logits_ref).abs().max().item() }")
        assert (logits - logits_ref).abs().max().item() < 2 * hf_error
        print(f"Logits CG max diff: {(logits_cg - logits_ref).abs().max().item() }")
        assert torch.equal(logits_cg, logits)
