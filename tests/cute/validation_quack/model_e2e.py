"""Full Qwen3 inference on one immutable FA4 source tree per process."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

real_version = metadata.version
quack_stamp = os.getenv("FA_TEST_QUACK_STAMP")
if quack_stamp is not None:

    def version(name):
        return (
            real_version(name) + "+test." + quack_stamp
            if name == "quack-kernels"
            else real_version(name)
        )

    metadata.version = version

# A metadata identity probe must precede FA4's module-level cache creation.
from flash_attn.cute import cache_utils, flash_attn_func, softmax  # noqa: E402

calls = 0


def fa4_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **kwargs
):
    global calls
    assert not module.training and dropout == 0.0 and sliding_window is None
    calls += 1
    out, _ = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=scaling,
        causal=True,
        pack_gqa=True,
        num_splits=1,
    )
    return out, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    AttentionInterface.register("fa4", fa4_attention)
    AttentionMaskInterface.register("fa4", sdpa_mask)
    reference = {} if args.write_reference else json.loads(args.reference.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="fa4",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    storages = tuple(p.data_ptr() for p in model.parameters())
    prefix = tokenizer(
        "Explain why the sky looks blue and give a short example. ", return_tensors="pt"
    ).input_ids.cuda()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hash = hashlib.sha256(Path(softmax.__file__).read_bytes()).hexdigest()
    for length in (128, 512):
        ids = prefix.repeat(1, (length + prefix.shape[1] - 1) // prefix.shape[1])[:, :length]
        mask = torch.ones_like(ids)
        generation = {
            "max_new_tokens": 32,
            "min_new_tokens": 32,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            torch.cuda.synchronize()
            prefill_started = time.perf_counter()
            actual = model(ids, attention_mask=mask, use_cache=False).logits
            torch.cuda.synchronize()
            first_prefill_s = time.perf_counter() - prefill_started
            assert torch.isfinite(actual).all()
            generation_started = time.perf_counter()
            actual_tokens = model.generate(ids, attention_mask=mask, **generation)
            torch.cuda.synchronize()
            first_generate_s = time.perf_counter() - generation_started
            assert actual_tokens.shape[1] - length == 32
            identity = {
                "input_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                "logits_sha256": hashlib.sha256(actual.cpu().float().numpy().tobytes()).hexdigest(),
                "token_sha256": hashlib.sha256(actual_tokens.cpu().numpy().tobytes()).hexdigest(),
            }
            if args.write_reference:
                reference[str(length)] = identity
                args.reference.write_text(json.dumps(reference, indent=2))
            else:
                assert identity == reference[str(length)], (identity, reference[str(length)])
            for _ in range(2):
                model.generate(ids, attention_mask=mask, **generation)
            torch.cuda.synchronize()
            samples = []
            initial_calls = calls
            for _ in range(args.rounds):
                start = time.perf_counter()
                tokens = model.generate(ids, attention_mask=mask, **generation)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
                torch.testing.assert_close(tokens, actual_tokens, rtol=0, atol=0)
            observed_calls = calls - initial_calls
            expected_calls = model.config.num_hidden_layers * 32 * args.rounds
            assert observed_calls == expected_calls
            assert tuple(p.data_ptr() for p in model.parameters()) == storages
        print(
            json.dumps(
                {
                    "label": args.label,
                    "sha": sha,
                    "softmax_sha256": source_hash,
                    "pid": os.getpid(),
                    "gpu": torch.cuda.get_device_name(),
                    "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "model": "Qwen/Qwen3-0.6B",
                    **identity,
                    "quack": real_version("quack-kernels"),
                    "simulated_quack_stamp": quack_stamp,
                    "persistent_cache": cache_utils.CUTE_DSL_CACHE_ENABLED,
                    "cache_fingerprint": (
                        cache_utils._compute_source_fingerprint()
                        if cache_utils.CUTE_DSL_CACHE_ENABLED
                        else None
                    ),
                    "load_s": load_s,
                    "first_fa4_prefill_s": first_prefill_s,
                    "first_fa4_generate_s": first_generate_s,
                    "mode": "transformers_eager_fa4",
                    "prefill_tokens": length,
                    "output_tokens": 32,
                    "median_s": statistics.median(samples),
                    "samples_s": samples,
                    "observed_fa4_calls": observed_calls,
                    "expected_fa4_calls": expected_calls,
                    "native_fa4_reference_match": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
