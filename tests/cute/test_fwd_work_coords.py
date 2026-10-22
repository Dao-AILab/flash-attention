"""Host checks for the SM100 forward work-tile row map (NOTE [SM100 forward work coordinates])."""

import itertools

import pytest

from flash_attn.cute.flash_fwd_sm100 import work_q_block


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@pytest.mark.parametrize("cta_group_size", [1, 2])
@pytest.mark.parametrize("q_stage", [1, 2])
@pytest.mark.parametrize("seqlen_q", [1, 127, 128, 129, 511, 1000])
def test_work_tile_rows_partition_q(cta_group_size, q_stage, seqlen_q):
    m_block_size = 128
    work_rows = m_block_size * cta_group_size * q_stage
    num_work_m = ceil_div(seqlen_q, work_rows)
    owner = {}
    for work_m, stage, cta_rank in itertools.product(
        range(num_work_m), range(q_stage), range(cta_group_size)
    ):
        q_block = work_q_block(
            work_m, stage, cta_rank, q_stage=q_stage, cta_group_size=cta_group_size
        )
        # Rank 0 is the stage's MMA-tile origin; partitions add the rank once.
        assert (
            q_block
            == work_q_block(work_m, stage, 0, q_stage=q_stage, cta_group_size=cta_group_size)
            + cta_rank
        )
        for row_in_cta in range(m_block_size):
            q_row = m_block_size * q_block + row_in_cta
            # Ordered layout (row_in_cta, cta_rank, stage, work_m) : (1, R, R*C, R*C*S).
            assert q_row == (
                row_in_cta
                + m_block_size * cta_rank
                + m_block_size * cta_group_size * stage
                + work_rows * work_m
            )
            assert q_row not in owner
            owner[q_row] = (work_m, stage, cta_rank)
            # Both CTAs of a work tile stay inside its row range.
            assert work_m * work_rows <= q_row < (work_m + 1) * work_rows
    assert sorted(owner) == list(range(num_work_m * work_rows))
    valid_rows = [q_row for q_row in owner if q_row < seqlen_q]
    assert sorted(valid_rows) == list(range(seqlen_q))
    # Only the last work tile holds tail rows.
    assert all(owner[q_row][0] == num_work_m - 1 for q_row in owner if q_row >= seqlen_q)
