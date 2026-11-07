"""Host checks for the SM100 backward work-tile and head maps (NOTE [SM100 backward coordinate
spaces] in flash_bwd_sm100.py, NOTE [SM100 backward work coordinates] in tile_scheduler.py)."""

import itertools

import pytest

from flash_attn.cute.flash_bwd_sm100 import bwd_kv_group_pos, bwd_kv_head, work_kv_block


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@pytest.mark.parametrize("cta_group_size", [1, 2])
@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize(
    "seqlen_k", [1, 63, 64, 65, 127, 128, 129, 255, 256, 511, 1000, 4224]
)
def test_work_tile_rows_partition_kv(cta_group_size, tile_n, seqlen_k):
    """Every CTA of a work tile gets its own tile_n rows; the work tiles partition the KV rows."""
    work_rows = tile_n * cta_group_size
    num_work_n = ceil_div(seqlen_k, work_rows)
    # The scheduler count equals the old per-CTA count rounded up to whole clusters.
    assert num_work_n * cta_group_size == ceil_div(
        ceil_div(seqlen_k, tile_n), cta_group_size
    ) * (cta_group_size)
    owner = {}
    for work_n, cta_rank in itertools.product(range(num_work_n), range(cta_group_size)):
        kv_block = work_kv_block(work_n, cta_rank, cta_group_size=cta_group_size)
        # Rank 0 is the work tile's origin; the rank is added once.
        assert (
            kv_block
            == work_kv_block(work_n, 0, cta_group_size=cta_group_size) + cta_rank
        )
        for row_in_cta in range(tile_n):
            kv_row = tile_n * kv_block + row_in_cta
            # Ordered layout (row_in_cta, cta_rank, work_n) : (1, N, N*C).
            assert kv_row == row_in_cta + tile_n * cta_rank + work_rows * work_n
            assert kv_row not in owner
            owner[kv_row] = (work_n, cta_rank)
            # Both CTAs of a work tile stay inside its row range, so the mask and the m_block
            # range can be computed from work_n alone.
            assert work_n * work_rows <= kv_row < (work_n + 1) * work_rows
            # The per-CTA row predicate of the dK/dV stores: row_in_cta < seqlen_k - N * kv_block.
            assert (row_in_cta < seqlen_k - tile_n * kv_block) == (kv_row < seqlen_k)
    assert sorted(owner) == list(range(num_work_n * work_rows))
    assert sorted(kv_row for kv_row in owner if kv_row < seqlen_k) == list(
        range(seqlen_k)
    )
    # Only the last work tile holds tail rows.
    assert all(
        owner[kv_row][0] == num_work_n - 1 for kv_row in owner if kv_row >= seqlen_k
    )


@pytest.mark.parametrize("qhead_per_kvhead", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("nheads_kv", [1, 2, 5])
def test_scheduler_heads_cover_query_heads(qhead_per_kvhead, nheads_kv):
    """Without pack_gqa a scheduler head is a query head and its KV head is the group it
    belongs to; the group positions enumerate the group in scheduler order."""
    nheads = qhead_per_kvhead * nheads_kv
    groups = {}
    for head_idx in range(nheads):
        kv_head = bwd_kv_head(
            head_idx, qhead_per_kvhead=qhead_per_kvhead, pack_gqa=False
        )
        pos = bwd_kv_group_pos(head_idx, qhead_per_kvhead=qhead_per_kvhead)
        assert 0 <= kv_head < nheads_kv and 0 <= pos < qhead_per_kvhead
        # The query head of a (KV head, position) pair, as the GQA layouts place it.
        assert head_idx == kv_head * qhead_per_kvhead + pos
        groups.setdefault(kv_head, []).append(pos)
    # Each KV head accumulates its query heads in order 0 .. qh - 1, each exactly once (the
    # deterministic dK/dV semaphore counts that order).
    assert groups == {
        kv_head: list(range(qhead_per_kvhead)) for kv_head in range(nheads_kv)
    }


@pytest.mark.parametrize("qhead_per_kvhead", [2, 4, 8])
@pytest.mark.parametrize("nheads_kv", [1, 3])
def test_packed_scheduler_heads_are_kv_heads(qhead_per_kvhead, nheads_kv):
    """With pack_gqa a scheduler head is a KV head: the packed rows of a tile hold its query
    heads (packed row r = token r // pf, query head kv_head * pf + r % pf; NOTE [bwd pack_gqa])."""
    pack_factor = qhead_per_kvhead
    for head_idx in range(nheads_kv):
        kv_head = bwd_kv_head(
            head_idx, qhead_per_kvhead=qhead_per_kvhead, pack_gqa=True
        )
        assert kv_head == head_idx
        seen = set()
        for packed_row in range(pack_factor * 7):
            token, offset = divmod(packed_row, pack_factor)
            query_head = kv_head * pack_factor + offset
            # The packed row's query head lies in this KV head's group and comes back to it.
            assert bwd_kv_head(
                query_head, qhead_per_kvhead=qhead_per_kvhead, pack_gqa=False
            ) == (kv_head)
            assert (
                bwd_kv_group_pos(query_head, qhead_per_kvhead=qhead_per_kvhead)
                == offset
            )
            seen.add((token, query_head))
        assert len(seen) == pack_factor * 7
