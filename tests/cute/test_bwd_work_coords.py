"""Host checks for the SM100 backward work-tile and head maps (NOTE [SM100 backward coordinate
spaces] in flash_bwd_sm100.py, NOTE [SM100 backward work coordinates] in tile_scheduler.py)."""

import itertools

import pytest

from flash_attn.cute.flash_bwd_sm100 import (
    bwd_fold_trip,
    bwd_kv_group_pos,
    bwd_kv_head,
    bwd_scratch_head,
    work_kv_block,
)


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


@pytest.mark.parametrize("pack_factor", [1, 2, 4])
@pytest.mark.parametrize("qhead_per_kvhead", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("nheads_kv", [1, 2, 5])
def test_scheduler_heads_cover_query_heads(pack_factor, qhead_per_kvhead, nheads_kv):
    """Without the head loop a scheduler head is a query head (pack_factor 1) or a packed
    group of pack_factor query heads; its KV head is the group it belongs to and the group
    positions enumerate the KV head's kv_group tiles in scheduler order."""
    if qhead_per_kvhead % pack_factor:
        pytest.skip("pack_factor must divide qhead_per_kvhead")
    kv_group = qhead_per_kvhead // pack_factor
    nheads_sched = kv_group * nheads_kv
    groups = {}
    for head_idx in range(nheads_sched):
        kv_head = bwd_kv_head(head_idx, kv_group=kv_group, head_loop=False)
        pos = bwd_kv_group_pos(head_idx, kv_group=kv_group)
        assert 0 <= kv_head < nheads_kv and 0 <= pos < kv_group
        # The scheduler head of a (KV head, position) pair, as the GQA layouts place it.
        assert head_idx == kv_head * kv_group + pos
        # Its scratch head is itself; packed row r of it holds query head
        # head_idx * pack_factor + r % pack_factor, which lies in this KV head's group.
        assert (
            bwd_scratch_head(head_idx, 0, kv_group=kv_group, head_loop=False)
            == head_idx
        )
        for packed_row in range(pack_factor * 3):
            query_head = head_idx * pack_factor + packed_row % pack_factor
            assert query_head // qhead_per_kvhead == kv_head
        groups.setdefault(kv_head, []).append(pos)
    # Each KV head accumulates its tiles in order 0 .. kv_group - 1, each exactly once (the
    # deterministic dK/dV semaphore counts that order).
    assert groups == {kv_head: list(range(kv_group)) for kv_head in range(nheads_kv)}


@pytest.mark.parametrize("pack_factor", [1, 2, 4])
@pytest.mark.parametrize("qhead_per_kvhead", [2, 4, 8, 16])
@pytest.mark.parametrize("nheads_kv", [1, 3])
@pytest.mark.parametrize("n_m_blocks", [1, 3, 7])
def test_head_loop_walks_every_sub_group(
    pack_factor, qhead_per_kvhead, nheads_kv, n_m_blocks
):
    """With the head loop a scheduler head is a KV head whose tile walks the KV head's
    kv_group sub-groups (scratch heads) in order, n_m_blocks trips each; together the
    sub-groups' packed rows cover every query head of the KV head exactly once."""
    if qhead_per_kvhead % pack_factor:
        pytest.skip("pack_factor must divide qhead_per_kvhead")
    kv_group = qhead_per_kvhead // pack_factor
    if kv_group == 1:
        pytest.skip("head_loop needs more than one sub-group")
    for head_idx in range(nheads_kv):
        assert bwd_kv_head(head_idx, kv_group=kv_group, head_loop=True) == head_idx
        trips = [
            bwd_fold_trip(i, n_m_blocks=n_m_blocks, head_loop=True)
            for i in range(kv_group * n_m_blocks)
        ]
        assert trips == [(g, m) for g in range(kv_group) for m in range(n_m_blocks)]
        query_heads = []
        for g in range(kv_group):
            scratch = bwd_scratch_head(head_idx, g, kv_group=kv_group, head_loop=True)
            # The scratch head is the KV head's g-th packed group in the GQA layout...
            assert bwd_kv_head(scratch, kv_group=kv_group, head_loop=False) == head_idx
            assert bwd_kv_group_pos(scratch, kv_group=kv_group) == g
            # ... and its packed rows hold pack_factor consecutive query heads.
            query_heads += [scratch * pack_factor + r for r in range(pack_factor)]
        assert query_heads == list(
            range(head_idx * qhead_per_kvhead, (head_idx + 1) * qhead_per_kvhead)
        )
    # Without the head loop a trip is just its m_block.
    assert [
        bwd_fold_trip(i, n_m_blocks=n_m_blocks, head_loop=False) for i in range(3)
    ] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]


@pytest.mark.parametrize("qhead_per_kvhead", [2, 4, 8])
@pytest.mark.parametrize("nheads_kv", [1, 3])
def test_packed_scheduler_heads_are_kv_heads(qhead_per_kvhead, nheads_kv):
    """With a full pack (pack_factor = qhead_per_kvhead, kv_group 1) a scheduler head is a KV
    head: the packed rows of a tile hold its query heads (packed row r = token r // pf, query
    head kv_head * pf + r % pf; NOTE [bwd pack_gqa])."""
    pack_factor = qhead_per_kvhead
    for head_idx in range(nheads_kv):
        kv_head = bwd_kv_head(head_idx, kv_group=1, head_loop=False)
        assert kv_head == head_idx
        seen = set()
        for packed_row in range(pack_factor * 7):
            token, offset = divmod(packed_row, pack_factor)
            query_head = kv_head * pack_factor + offset
            # The packed row's query head lies in this KV head's group and comes back to it.
            assert (
                bwd_kv_head(query_head, kv_group=qhead_per_kvhead, head_loop=False)
                == kv_head
            )
            assert bwd_kv_group_pos(query_head, kv_group=qhead_per_kvhead) == offset
            seen.add((token, query_head))
        assert len(seen) == pack_factor * 7
