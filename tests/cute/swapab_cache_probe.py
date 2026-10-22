"""Real-SM100 fresh-process persistent-cache gate (no production patches).

Run these in TWO processes with an initially empty, dedicated directory::

    export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$PWD/swapab_resume_cache"
    timeout 240s python tests/cute/swapab_cache_probe.py --phase populate
    timeout 240s python tests/cute/swapab_cache_probe.py --phase reuse

Both passes force native routing and real tensors, enable the existing persistent
cache before importing FA4, validate O/LSE numerically, and verify on-disk objects.
The reuse pass starts with empty process-local caches and temporarily replaces
cute.compile with a failing guard. It cannot silently recompile on a disk miss.
The populate pass requires at least one compile; use a fresh directory to rerun.
Coverage: BF16 dense GQA, Dqk96/Dv64, used lengths [0,513], splits1/3 and generic
combine. Not a paging/semantic-modifier/performance/race-freedom validation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=("populate", "reuse"), required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"):
        parser.error(
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must be set to a dedicated cache directory"
        )
    os.environ.update(
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_SM100_BACKEND="swapab",
        FLASH_ATTENTION_FAKE_TENSOR="0",
    )

    # Direct script invocation must not import the unrelated FA2 extension.
    # Only supply the local package namespace when it has not been loaded yet;
    # all FA4 submodules, routing, caches and launchers remain real production code.
    root = Path(__file__).resolve().parents[2]
    if "flash_attn" not in sys.modules:
        import importlib.machinery
        import types

        package = types.ModuleType("flash_attn")
        package.__path__ = [str(root / "flash_attn")]
        package.__file__ = str(root / "flash_attn" / "__init__.py")
        package.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None, is_package=True
        )
        sys.modules["flash_attn"] = package
    import torch
    import cutlass.cute as cute
    from flash_attn.cute.cache_utils import JITPersistentCache
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_fwd_combine

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("cache probe requires a real SM100 GPU")
    caches = {
        "fwd": _flash_attn_fwd.compile_cache,
        "combine": _flash_attn_fwd_combine.compile_cache,
    }
    for name, cache in caches.items():
        assert isinstance(cache, JITPersistentCache), (
            f"{name}: persistent cache disabled"
        )
        assert not cache.cache, f"{name}: process-local cache must start empty"
    print(f"START cache_probe phase={args.phase} pid={os.getpid()}", flush=True)
    torch.manual_seed(982)
    q = torch.randn(2, 1, 6, 96, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 777, 2, 96, device="cuda", dtype=q.dtype)
    v = torch.randn(2, 777, 2, 64, device="cuda", dtype=q.dtype)
    used = torch.tensor([0, 513], device="cuda", dtype=torch.int32)
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q.float(), k.repeat_interleave(3, 2).float()
    ) * (96**-0.5)
    positions = torch.arange(777, device="cuda")
    scores.masked_fill_(
        positions[None, None, None, :] >= used[:, None, None, None], -torch.inf
    )
    expected_lse = scores.logsumexp(-1)
    expected = torch.einsum(
        "bhqk,bkhd->bqhd",
        scores.softmax(-1).nan_to_num(),
        v.repeat_interleave(3, 2).float(),
    )
    compiles = 0
    original_compile = cute.compile

    def checked_compile(*compile_args, **kwargs):
        nonlocal compiles
        compiles += 1
        if args.phase == "reuse":
            raise AssertionError(
                "fresh-process persistent cache miss: cute.compile called during reuse"
            )
        return original_compile(*compile_args, **kwargs)

    # Test-scoped instrumentation only: never replace a cache, backend or launcher.
    with patch.object(cute, "compile", checked_compile):
        for ns in (1, 3):
            out, lse, *_ = _flash_attn_fwd(
                q, k, v, seqused_k=used, num_splits=ns, return_lse=True
            )
            torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.025)
            torch.testing.assert_close(lse, expected_lse, atol=0.004, rtol=0.003)
            assert torch.count_nonzero(out[0]) == 0
            assert torch.isneginf(lse[0]).all()
        torch.cuda.synchronize()
    assert len(caches["fwd"].cache) == 2, "expected distinct split1/split3 forward keys"
    assert caches["combine"].cache, "split3 must exercise the shared combine cache"
    assert all(key[0] == ("sm100_swapab", 1) for key in caches["fwd"].cache), (
        "native route not selected"
    )
    if args.phase == "populate":
        assert compiles > 0, "populate requires a fresh dedicated cache directory"
    else:
        assert compiles == 0
    object_files = []
    for name, cache in caches.items():
        for key in cache.cache:
            path = cache.cache_path / f"{cache._key_to_hash(key)}.o"
            assert path.is_file() and path.stat().st_size > 0, (
                f"{name}: missing exported object {path}"
            )
            object_files.append(str(path.resolve()))
    report = dict(
        phase=args.phase,
        pid=os.getpid(),
        root=str(root),
        compiles=compiles,
        splits=[1, 3],
        forward_keys=sorted(repr(key) for key in caches["fwd"].cache),
        object_files=sorted(object_files),
    )
    print("CACHE_PROBE " + json.dumps(report, sort_keys=True), flush=True)
    print(f"PASS cache_probe phase={args.phase}", flush=True)


if __name__ == "__main__":
    main()
