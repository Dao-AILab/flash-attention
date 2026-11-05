import pytest

from flash_attn.cute.interface import _resolve_causal_local_window


@pytest.mark.parametrize(
    "causal,left,right,expected",
    [
        (False, -1, -1, (False, False, None, None)),
        (True, -1, -1, (True, False, None, None)),
        (False, -1, 0, (True, False, None, None)),
        (True, -1, 0, (True, False, None, None)),
        (False, None, 0, (True, False, None, None)),
        (False, -2, 3, (False, True, None, 3)),
        (False, 3, -2, (False, True, 3, None)),
        (False, 3, 0, (False, True, 3, 0)),
        (False, None, None, (False, False, None, None)),
    ],
)
def test_window_normalization(causal, left, right, expected):
    assert _resolve_causal_local_window(causal, left, right) == expected


def test_mask_mod_preserves_window():
    mask_mod = object()
    assert _resolve_causal_local_window(True, -1, 0, mask_mod) == (False, False, -1, 0)
