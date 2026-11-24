# torchrun --standalone --nproc_per_node=2 -m pytest -q tests/modules/test_mha_parallel_gqa.py
import os

import pytest
import torch
import torch.distributed as dist
from transformers import GPT2Config

from flash_attn.models.gpt import combine_state_dicts_tp, shard_state_dict_tp
from flash_attn.modules.mha import MHA, ParallelMHA


@pytest.fixture(scope="module")
def tp_group():
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("Run with torchrun --standalone --nproc_per_node=2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl")
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


@pytest.mark.parametrize("use_alibi", [False, True], ids=["plain", "alibi"])
@torch.no_grad()
def test_mha_parallel_gqa(tp_group, use_alibi):
    torch.manual_seed(0)
    rank = dist.get_rank(tp_group)
    head_dim = 64
    embed_dim = 6 * head_dim
    kwargs = dict(
        num_heads_kv=3,
        use_flash_attn=True,
        causal=True,
        use_alibi=use_alibi,
        device="cuda",
        dtype=torch.float16,
    )
    # Generate shared data before the rank-dependent ParallelMHA initialization.
    x = torch.randn(2, 32, embed_dim, device="cuda", dtype=torch.float16)
    dense = MHA(embed_dim, 6, **kwargs).eval()
    # One nonzero per output row isolates head placement from fp16 reduction order.
    identity = torch.eye(embed_dim, device="cuda", dtype=torch.float16)
    dense.Wqkv.weight.copy_(identity.repeat(2, 1))
    dense.out_proj.weight.copy_(identity)
    model = ParallelMHA(embed_dim, 6, tp_group, sequence_parallel=False, **kwargs).eval()

    config = GPT2Config(n_embd=embed_dim, n_head=6, n_head_kv=3, n_layer=1, vocab_size=4)
    prefix = "transformer.layers.0.mixer."
    shard = shard_state_dict_tp(
        {prefix + key: value for key, value in dense.state_dict().items()}, config, 2, rank
    )
    model.load_state_dict({key[len(prefix) :]: value for key, value in shard.items()}, strict=True)
    local_q, local_kv = ((4, 2), (2, 1))[rank]
    assert (model.num_heads_per_rank, model.num_heads_kv_per_rank) == (local_q, local_kv)
    assert model.Wqkv.out_features == (local_q + 2 * local_kv) * head_dim
    assert model.out_proj.in_features == local_q * head_dim
    if use_alibi:
        q_slice = (slice(0, 4), slice(4, 6))[rank]
        torch.testing.assert_close(
            model.inner_cross_attn.alibi_slopes,
            dense.inner_cross_attn.alibi_slopes[q_slice],
            rtol=0,
            atol=0,
        )
    actual = model(x.flatten(0, 1), seqlen=x.shape[1])
    expected = dense(x).flatten(0, 1)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-3)


def test_combine_legacy_uneven_gqa_shards():
    # Six Q heads share three KV heads across two ranks.
    config = GPT2Config(n_embd=12, n_head=6, n_head_kv=3, n_layer=1, vocab_size=4)
    prefix = "transformer.layers.0.mixer."
    embedding = "transformer.embeddings.word_embeddings.weight"
    state = {
        embedding: torch.arange(48).reshape(4, 12),
        prefix + "Wqkv.weight": torch.arange(288).reshape(24, 12),
        prefix + "Wqkv.bias": torch.arange(24),
        prefix + "out_proj.weight": torch.arange(144).reshape(12, 12),
        prefix + "out_proj.bias": torch.arange(12),
    }
    current = [shard_state_dict_tp(dict(state), config, 2, rank) for rank in range(2)]
    assert [s[prefix + "Wqkv.weight"].shape[0] for s in current] == [16, 8]
    assert [s[prefix + "out_proj.weight"].shape[1] for s in current] == [8, 4]
    torch.testing.assert_close(combine_state_dicts_tp(current, config), state, rtol=0, atol=0)

    # Older checkpoints split the Q heads evenly even when the KV head counts differed.
    legacy = []
    for rank, kv_slice in enumerate((slice(0, 4), slice(4, 6))):
        q_slice = slice(rank * 6, (rank + 1) * 6)
        shard = {
            embedding: state[embedding][rank * 2 : (rank + 1) * 2],
            prefix + "out_proj.weight": state[prefix + "out_proj.weight"][:, q_slice],
        }
        for suffix in ("weight", "bias"):
            key = prefix + "Wqkv." + suffix
            q, k, v = state[key].split((12, 6, 6), dim=0)
            shard[key] = torch.cat((q[q_slice], k[kv_slice], v[kv_slice]))
        if rank == 0:
            shard[prefix + "out_proj.bias"] = state[prefix + "out_proj.bias"]
        legacy.append(shard)
    assert [s[prefix + "Wqkv.weight"].shape[0] for s in legacy] == [14, 10]
    torch.testing.assert_close(combine_state_dicts_tp(legacy, config), state, rtol=0, atol=0)
