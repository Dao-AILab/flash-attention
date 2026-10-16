# Copyright (c) 2025, Tri Dao.
"""Storage order of the fp32 accumulate tiles the SM100 backward hands to its postprocess.

dQ, and dK / dV for GQA, are accumulated with bulk reduce-adds into flat fp32 buffers of
rows * hdim values per tile, in whatever order the epilogue streams them out of TMEM, and
FlashAttentionBackwardPostprocess converts them to the output dtype. `accum_tile_layout` is the
one description of that order: the writer derives its addresses from it (or checks its copies
against it, see FlashAttentionBackwardSm100._check_accum_tile_orders) and the postprocess reads
the tile through it. The order is part of the postprocess's compile key.
"""

from enum import Enum

import cutlass.cute as cute


class AccumTileOrder(Enum):
    # 32x32b TMEM fragment order of a 128-lane accumulator whose lane is the row: 4-column
    # vectors of every row, then the next 4 columns. The reduce epilogues stream ncol-column
    # stages, each a contiguous block of this order, so the stage width does not appear.
    TMEM_FRAG_1CTA = "tmem_frag_1cta"
    # The 2-SM MMA splits the 128 rows across the CTA pair, one 64 * hdim block per CTA; in a
    # block the TMEM lane is row % 64 + 64 * column half, so the 4-column vectors of the 64
    # rows for columns [0, hdim / 2) interleave, 4 columns at a time, with those of the other
    # half.
    TMEM_FRAG_2CTA = "tmem_frag_2cta"
    # A plain (rows, hdim) block.
    ROW_MAJOR = "row_major"


def accum_tile_layout(order: AccumTileOrder, rows: int, hdim: int) -> cute.Layout:
    """(rows, hdim) -> offset in the flat accumulate tile stored in `order`."""
    if order == AccumTileOrder.TMEM_FRAG_1CTA:
        return cute.make_layout((rows, (4, hdim // 4)), stride=(4, (1, 4 * rows)))
    if order == AccumTileOrder.TMEM_FRAG_2CTA:
        assert rows == 128, "the 2CTA order pairs two 64-row CTA blocks"
        half = rows // 2
        return cute.make_layout(
            ((half, 2), (4, hdim // 8, 2)), stride=((4, half * hdim), (1, 8 * half, 4 * half))
        )
    assert order == AccumTileOrder.ROW_MAJOR
    return cute.make_layout((rows, hdim), stride=(hdim, 1))
