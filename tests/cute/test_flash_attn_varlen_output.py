# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yong Li, Wei Lin.

import pytest
import torch

from test_flash_attn import _run_flash_attn_varlen_output


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_learnable_sink", [False])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize(
    "causal,local_enum",
    [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0)],
)
@pytest.mark.parametrize("add_unused_qkv", [False])
@pytest.mark.parametrize("d", [64, 128, 192, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        # SM100 hd256 2CTA test cases
        (64, 1),
        (255, 256),
        (4096, 4096),
        (4224, 4224),
    ],
)
@pytest.mark.parametrize("varlen_mode", ["random", "third", "full"])
@pytest.mark.parametrize(
    "zero_lengths_q, zero_lengths_k",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "unpad_q, unpad_kv",
    [
        (True, True),
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_flash_attn_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    add_unused_qkv,
    causal,
    local_enum,
    softcap,
    deterministic,
    has_qv,
    has_learnable_sink,
    mha_type,
    dtype,
    varlen_mode,
    zero_lengths_q,
    zero_lengths_k,
    unpad_q,
    unpad_kv,
):
    return _run_flash_attn_varlen_output(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        d=d,
        add_unused_qkv=add_unused_qkv,
        causal=causal,
        local_enum=local_enum,
        softcap=softcap,
        deterministic=deterministic,
        has_qv=has_qv,
        has_learnable_sink=has_learnable_sink,
        mha_type=mha_type,
        dtype=dtype,
        varlen_mode=varlen_mode,
        zero_lengths_q=zero_lengths_q,
        zero_lengths_k=zero_lengths_k,
        unpad_q=unpad_q,
        unpad_kv=unpad_kv,
    )
