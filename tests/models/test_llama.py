# Copyright (c) 2023, Tri Dao.

# To run the huggingface implementation, we first need to convert the weights:
# https://github.com/huggingface/transformers/pull/21955
# python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir $CHECKPOINT_DIR/llama --model_size 7B --output_dir $CHECKPOINT_DIR$/llama/7B-hf
# and repeat for 13B, 30B, 65B

import os
import time
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import torch
import pytest

from transformers import LlamaConfig, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.generation import update_graph_cache


@pytest.mark.parametrize('model_name', ["7B"])
def test_llama_state_dict(model_name):
    checkpoint_path = Path(os.environ.get('CHECKPOINT_DIR',
                                          current_dir.parent.parent / 'checkpoints')) / 'llama'
    config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
    ckpt_state_dicts = state_dicts_from_checkpoint(checkpoint_path, model_name)
    pretrained_state_dict = remap_state_dict_meta_llama(ckpt_state_dicts[0], config)
    model = GPTLMHeadModel(config, device='meta')  # Without device='meta' init is very slow
    state_dict = model.state_dict()
    rotary_inv_freq_keys = {f'transformer.layers.{l}.mixer.rotary_emb.inv_freq'
                            for l in range(config.n_layer)}
    assert state_dict.keys() == pretrained_state_dict.keys() | rotary_inv_freq_keys
    for k in state_dict.keys() - rotary_inv_freq_keys:
        assert state_dict[k].shape == pretrained_state_dict[k].shape


@pytest.mark.parametrize('model_name', ["7B", "13B"])
def test_llama_optimized(model_name):
    """Check that our implementation of LLaMa (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    checkpoint_path = Path(os.environ.get('CHECKPOINT_DIR',
                                          current_dir.parent.parent / 'checkpoints')) / 'llama'

    dtype = torch.float16
    device = 'cuda'
    config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    ckpt_state_dicts = state_dicts_from_checkpoint(checkpoint_path, model_name)
    pretrained_state_dicts = [remap_state_dict_meta_llama(s, config) for s in ckpt_state_dicts]
    pretrained_state_dict = combine_state_dicts_tp(pretrained_state_dicts, config)
    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(pretrained_state_dict, strict=False)
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
    # Need auto here since the 13B fp32 model doesn't fit in memory on a A100 40GB
    model_ref = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                 device_map='auto')
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.model(input_ids).last_hidden_state.to(device=device)
        logits_ref = model_ref(input_ids).logits.to(device=device)
    del model_ref

    model_hf = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                torch_dtype=dtype, device_map={"": device})
    model_hf.eval()
    out_hf = model_hf.model(input_ids).last_hidden_state
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


# torchrun --no_python --nproc_per_node=2 pytest -q -s tests/models/test_llama.py -k "parallel"
@pytest.mark.skip(reason="Tensor Parallel is not implemented for GatedMLP yet")
@pytest.mark.parametrize('world_size', [2])
@pytest.mark.parametrize('model_name', ["13B"])
def test_llama_parallel(model_name, world_size):
    """Check that our implementation of LLaMa (with all optimizations enabled) matches the
    HF implementation: the output of our forward pass in fp16 should be around the same as the HF
    forward pass in fp16, when compared to the HF forward pass in fp32.
    """
    from apex.transformer import parallel_state

    checkpoint_path = Path(os.environ.get('CHECKPOINT_DIR',
                                          current_dir.parent.parent / 'checkpoints')) / 'llama'

    dtype = torch.float16
    device = 'cuda'
    config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = f'cuda:{torch.distributed.get_rank()}'
    assert world_size <= torch.distributed.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
    rank = parallel_state.get_tensor_model_parallel_rank()
    process_group = parallel_state.get_tensor_model_parallel_group()

    ckpt_state_dicts = state_dicts_from_checkpoint(checkpoint_path, model_name)
    pretrained_state_dicts = [remap_state_dict_meta_llama(s, config) for s in ckpt_state_dicts]
    pretrained_state_dict = combine_state_dicts_tp(pretrained_state_dicts, config)

    model = GPTLMHeadModel(config, process_group=process_group, device=device, dtype=dtype)
    model.load_state_dict(shard_state_dict_tp(pretrained_state_dict, config, world_size, rank),
                          strict=False)
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
    model_ref = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                 device_map='auto')
    model_ref.eval()
    with torch.no_grad():
        out_ref = model_ref.model(input_ids).last_hidden_state.to(device=device)
        logits_ref = model_ref(input_ids).logits.to(device=device)
    del model_ref

    model_hf = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                torch_dtype=dtype, device_map="auto")
    model_hf.eval()
    out_hf = model_hf.model(input_ids).last_hidden_state.to(device=device)
    logits_hf = model_hf(input_ids).logits.to(device=device)
    del model_hf

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(out_hf - out_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(out_hf - out_ref).abs().mean().item()}')
    assert (out - out_ref).abs().max().item() < 2 * (out_hf - out_ref).abs().max().item()

    print(f'Logits max diff: {(logits - logits_ref).abs().max().item()}')
    print(f'Logits mean diff: {(logits - logits_ref).abs().mean().item()}')
    print(f'HF fp16 max diff: {(logits_hf - logits_ref).abs().max().item()}')
    print(f'HF fp16 mean diff: {(logits_hf - logits_ref).abs().mean().item()}')
    assert (logits - logits_ref).abs().max().item() < 2 * (logits_hf - logits_ref).abs().max().item()


@pytest.mark.parametrize('model_name', ["7B"])
def test_llama_generation(model_name):
    checkpoint_path = Path(os.environ.get('CHECKPOINT_DIR',
                                          current_dir.parent.parent / 'checkpoints')) / 'llama'

    dtype = torch.float16
    device = 'cuda'
    config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoint_path, model_name))
    config.use_flash_attn = True
    config.fused_bias_fc = True
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = True
    config.residual_in_fp32 = True

    tokenizer = LlamaTokenizer.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf')
    eos_token_id = tokenizer.eos_token_id

    torch.manual_seed(0)
    batch_size = 1
    seqlen = 100
    max_length = 150
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), dtype=torch.long,
                              device=device)

    model_hf = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                torch_dtype=dtype, device_map={"": device})
    model_hf.eval()
    print("HF fp16")
    torch.cuda.synchronize()
    start = time.time()
    out_hf = model_hf.generate(input_ids=input_ids, max_length=max_length,
                               return_dict_in_generate=True, output_scores=True)
    torch.cuda.synchronize()
    print(f'Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms')
    del model_hf

    model_ref = LlamaForCausalLM.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf',
                                                 device_map={"": device})
    model_ref.eval()
    with torch.no_grad():
        logits_ref = model_ref(out_hf.sequences).logits[:, (seqlen - 1):-1]
    del model_ref

    ckpt_state_dicts = state_dicts_from_checkpoint(checkpoint_path, model_name)
    pretrained_state_dicts = [remap_state_dict_meta_llama(s, config) for s in ckpt_state_dicts]
    pretrained_state_dict = combine_state_dicts_tp(pretrained_state_dicts, config)
    model = GPTLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(pretrained_state_dict, strict=False)
    model.eval()

    print('Without CUDA graph')
    torch.cuda.synchronize()
    start = time.time()
    out = model.generate(input_ids=input_ids, max_length=max_length,
                         eos_token_id=eos_token_id, fused_ft_kernel=True,
                         return_dict_in_generate=True, output_scores=True, timing=True,
                         teacher_outputs=out_hf.sequences)
    torch.cuda.synchronize()
    print(f'Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms')

    # Capture graph outside the timing loop
    batch_size, seqlen_og = input_ids.shape
    model._decoding_cache = update_graph_cache(model, None, batch_size, seqlen_og, max_length)
    print('With CUDA graph')
    torch.cuda.synchronize()
    start = time.time()
    out_cg = model.generate(input_ids=input_ids, max_length=max_length,
                            fused_ft_kernel=True, cg=True,
                            return_dict_in_generate=True, output_scores=True, timing=True,
                            teacher_outputs=out_hf.sequences)
    torch.cuda.synchronize()
    print(f'Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms')

    with torch.no_grad():
        logits_parallel = model(out_hf.sequences).logits[:, (seqlen - 1):-1]
    logits_hf = torch.stack(out_hf.scores, dim=1)
    logits = torch.stack(out.scores, dim=1)
    logits_cg = torch.stack(out_cg.scores, dim=1)

    del model

    hf_error = (logits_hf - logits_ref).abs().max().item()

    print(f'HF fp16 logits max diff: {hf_error}')
    print(f'Logits max diff: {(logits - logits_ref).abs().max().item() }')
    print(f'Logits CG max diff: {(logits_cg - logits_ref).abs().max().item() }')

    assert (logits_parallel - logits_ref).abs().max().item() < 2 * hf_error
    assert (logits - logits_ref).abs().max().item() < 2 * hf_error
    assert torch.equal(logits_cg, logits)
