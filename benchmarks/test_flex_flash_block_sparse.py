import pytest
import torch
from flex_flash_block_sparse import (
    FlexBlockSparseCase,
    flex_block_sparse_reference,
    make_flex_block_sparse_inputs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mask_name", ["block_diagonal_128", "causal_window_128"])
def test_nightly_flex_flash_block_sparse_matches_float32(mask_name):
    case = FlexBlockSparseCase(
        name=f"correctness_{mask_name}",
        phase="holdout",
        profile="correctness",
        batch=2,
        q_heads=8,
        kv_heads=2,
        d=64,
        dv=64,
        seqlen_q=256,
        seqlen_k=256,
        mask_name=mask_name,
    )
    torch.manual_seed(0)
    inputs = make_flex_block_sparse_inputs(case)
    reference = flex_block_sparse_reference(case, inputs)

    with torch.no_grad():
        output = inputs.call(inputs.q, inputs.k, inputs.v, inputs.block_mask)
    torch.cuda.synchronize()

    torch.testing.assert_close(output.float(), reference, atol=0.04, rtol=0.04)
