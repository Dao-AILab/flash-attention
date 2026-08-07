"""Pin our aux-tensor cache keys to CuTe's own layout identity.

``get_aux_tensor_metadata`` reproduces CuTe's leading-dimension deduction from
torch metadata, because the launch path hands torch tensors straight to TVM-FFI
and reading ``__cache_key__`` would force a DLPack conversion on every call. A
DSL upgrade that changes the deduction must fail here rather than silently reuse
a kernel compiled for a different ABI.
"""

import sys
import types
from pathlib import Path

import pytest
import torch
from cutlass.cute.runtime import from_dlpack

if "flash_attn" not in sys.modules:
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "flash_attn")]
    sys.modules["flash_attn"] = package

from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    get_num_sms_for_selection,
    maybe_contiguous,
    to_cute_aux_tensor,
    validate_output_layout,
)


def _tagged(tensor: torch.Tensor, leading_dim: int) -> torch.Tensor:
    """Declare a leading dimension the way FlexAttention tags aux tensors."""
    tensor.__leading_dim__ = leading_dim
    return tensor


LAYOUTS = {
    "bf16_1d": torch.empty(4, dtype=torch.bfloat16),
    "bf16_1d_longer": torch.empty(8, dtype=torch.bfloat16),
    "fp32_1d": torch.empty(4, dtype=torch.float32),
    "bf16_1d_broadcast": torch.empty(1, dtype=torch.bfloat16).expand(4),
    "bf16_row_major": torch.empty(2, 3, dtype=torch.bfloat16),
    "bf16_row_major_larger": torch.empty(5, 7, dtype=torch.bfloat16),
    "bf16_col_major": torch.empty(3, 2, dtype=torch.bfloat16).t(),
    "bf16_unit_rows": torch.empty_strided((1, 4), (1, 1), dtype=torch.bfloat16),
    "bf16_unit_cols": torch.empty_strided((4, 1), (1, 1), dtype=torch.bfloat16),
    "bf16_tagged_last_dim": _tagged(torch.empty(2, 3, dtype=torch.bfloat16), 1),
    "bf16_tagged_negative_dim": _tagged(torch.empty(2, 3, dtype=torch.bfloat16), -1),
}


def _equivalence_classes(key_of) -> set[frozenset[str]]:
    """Group layout names by the cache key they produce."""
    classes: dict[object, set[str]] = {}
    for name, tensor in LAYOUTS.items():
        classes.setdefault(key_of(tensor), set()).add(name)
    return {frozenset(names) for names in classes.values()}


def test_fake_target_num_sms(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", "152")
    assert get_num_sms_for_selection(0, 103) == 152

    monkeypatch.delenv("FLASH_ATTENTION_NUM_SMS")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="FLASH_ATTENTION_NUM_SMS"):
        get_num_sms_for_selection(0, 103)


def test_layout_validation_uses_pointer_and_rejects_broadcast_output():
    storage = bytearray(256)
    base = torch.frombuffer(storage, dtype=torch.uint8)
    offset = 1 if (base.data_ptr() + 1) % 16 else 2
    unaligned = torch.frombuffer(storage, dtype=torch.uint8, count=128, offset=offset)

    assert unaligned.storage_offset() == 0
    assert unaligned.data_ptr() % 16 != 0
    aligned = maybe_contiguous(unaligned)
    assert aligned.data_ptr() % 16 == 0
    torch.testing.assert_close(aligned, unaligned)

    broadcast_out = torch.empty_strided((1, 4), (0, 1))
    with pytest.raises(AssertionError, match="must not have broadcast dimensions"):
        validate_output_layout(broadcast_out, "out", 16)


def test_metadata_groups_layouts_exactly_like_cute():
    cute_classes = _equivalence_classes(lambda t: to_cute_aux_tensor(t).__cache_key__)

    assert _equivalence_classes(lambda t: get_aux_tensor_metadata([t])[0]) == cute_classes
    # Sizes stay dynamic, so they must not split kernels; dtype, broadcast, and
    # leading-dimension placement are static and must.
    assert frozenset({"bf16_1d", "bf16_1d_longer"}) in cute_classes
    assert frozenset({"fp32_1d"}) in cute_classes
    assert frozenset({"bf16_1d_broadcast"}) in cute_classes
    assert frozenset({"bf16_col_major", "bf16_unit_cols"}) in cute_classes


@pytest.mark.parametrize("shape", [(1, 1), (2, 2)])
def test_ambiguous_leading_dimension_is_rejected(shape):
    tensor = torch.empty_strided(shape, (1, 1), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="no unique stride-1 leading dimension"):
        get_aux_tensor_metadata([tensor])
    with pytest.raises(RuntimeError, match="deduce the leading dimension"):
        from_dlpack(tensor).mark_layout_dynamic()
