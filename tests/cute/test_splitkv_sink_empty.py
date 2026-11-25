import math
import os

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1"


@pytest.mark.parametrize(
    "backend,hq,hkv,sq,packed",
    [
        ("dense", 4, 2, 1024, False),
        ("dense", 6, 2, 1024, True),
        ("mla", 16, 1, 320, True),
        ("mla", 64, 1, 320, True),
        ("mla", 24, 1, 1, True),
    ],
    ids=[
        "dense-tma",
        "dense-pack-gqa",
        "mla-kb128",
        "mla-kb64",
        "mla-kb64-padded-varlen",
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitkv_sink_on_empty_tile(backend, hq, hkv, sq, packed, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 regression")
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")
    device, dtype = "cuda", torch.bfloat16
    dv = 64 if backend == "dense" else 512
    sk, num_splits = 256, 3
    varlen = sq == 1
    b = 3 if varlen else 2
    q_prefix = (b * sq,) if varlen else (b, sq)
    q = torch.full((*q_prefix, hq, 64), 0.5, device=device, dtype=dtype)
    k = torch.full((b, sk, hkv, 64), -0.5, device=device, dtype=dtype)
    v = torch.ones((b, sk, hkv, dv), device=device, dtype=dtype)
    qv = (
        torch.zeros((*q_prefix, hq, dv), device=device, dtype=dtype)
        if backend == "mla"
        else None
    )
    sink = (torch.arange(hq, device=device, dtype=torch.float32) % 11 - 5) * 0.5
    layout = {}
    if varlen:
        layout = {
            "cu_seqlens_q": torch.arange(b + 1, device=device, dtype=torch.int32),
            "seqused_k": torch.tensor([sk, 0, sk], device=device, dtype=torch.int32),
            "max_seqlen_q": sq,
        }
    # Poison the O partials so an LSE-only fix can't pass by reading a zeroed allocation.
    empty = torch.empty
    poisoned = False

    def empty_with_poison(*shape, **kwargs):
        nonlocal poisoned
        result = empty(*shape, **kwargs)
        if (
            shape == (num_splits, *q_prefix, hq, dv)
            and kwargs.get("dtype") == torch.float32
        ):
            poisoned = True
            result.fill_(float("nan"))
        return result

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", empty_with_poison)
        out, lse, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            causal=True,
            pack_gqa=packed,
            num_splits=num_splits,
            learnable_sink=sink,
            softmax_scale=0.125,
            return_lse=True,
            **layout,
        )
    if is_fake_mode():
        return
    assert poisoned, "the call did not allocate the expected SplitKV partials"
    # Every visible score is 64 * 0.5 * (-0.5) * 0.125 = -2; each V is the vector of ones.
    count = (torch.arange(sq, device=device) + 1 + sk - sq).clamp_min(0)[None]
    if varlen:
        count = torch.tensor([[sk], [0], [sk]], device=device)
    else:
        count = count.expand(b, -1)
    kv_mass = count[..., None].double() * math.exp(-2.0)
    total_mass = kv_mass + sink.double().exp()
    if varlen:
        out, lse = out[:, None], lse[:, None]
    elif backend == "dense":
        lse = lse.transpose(-1, -2)
    assert (out[count == 0] == 0).all()
    torch.testing.assert_close(
        lse[count == 0], sink.expand_as(lse[count == 0]), atol=1e-6, rtol=0
    )
    expected = (kv_mass / total_mass)[..., None].expand_as(out)
    torch.testing.assert_close(out, expected.to(dtype), atol=4e-3, rtol=0)
    torch.testing.assert_close(lse, total_mass.log().float(), atol=2e-3, rtol=0)
