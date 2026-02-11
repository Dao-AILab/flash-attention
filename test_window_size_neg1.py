"""Test that window_size=(-1, -1) is normalized to (None, None) in the cute path.

The standard flash attention convention uses -1 to mean "no window" (infinite).
The cute path uses None for the same purpose. This test verifies that -1 is
correctly normalized to None so the kernel doesn't incorrectly enter local
(sliding window) mode.
"""

def simulate_window_logic(causal, window_size_left, window_size_right, mask_mod=None):
    """Reproduce the window size logic from _flash_attn_fwd (lines 260-271)."""
    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
                window_size_right = None
            else:
                causal, local = False, True
    else:
        causal, local = False, False
    return causal, local, window_size_left, window_size_right


def simulate_window_logic_with_fix(causal, window_size_left, window_size_right, mask_mod=None):
    """Same logic but with -1 -> None normalization applied first."""
    if window_size_left is not None and window_size_left < 0:
        window_size_left = None
    if window_size_right is not None and window_size_right < 0:
        window_size_right = None
    return simulate_window_logic(causal, window_size_left, window_size_right, mask_mod)


print("=== Without fix (current behavior) ===")

# Case 1: (None, None) — correct baseline
c, l, wl, wr = simulate_window_logic(False, None, None)
print(f"(None, None):  causal={c}, local={l}, wl={wl}, wr={wr}")
assert not c and not l, "Expected non-causal, non-local"

# Case 2: (-1, -1) — should be same as (None, None) but ISN'T
c, l, wl, wr = simulate_window_logic(False, -1, -1)
print(f"(-1,  -1):     causal={c}, local={l}, wl={wl}, wr={wr}")
if l:
    print("  BUG: local=True when it should be False!")

# Case 3: causal + (-1, -1) — should be same as causal + (None, None)
c, l, wl, wr = simulate_window_logic(True, -1, -1)
print(f"causal+(-1,-1): causal={c}, local={l}, wl={wl}, wr={wr}")
if l:
    print("  BUG: local=True when it should be False!")

print()
print("=== With fix (normalizing -1 to None) ===")

c, l, wl, wr = simulate_window_logic_with_fix(False, None, None)
print(f"(None, None):  causal={c}, local={l}, wl={wl}, wr={wr}")
assert not c and not l

c, l, wl, wr = simulate_window_logic_with_fix(False, -1, -1)
print(f"(-1,  -1):     causal={c}, local={l}, wl={wl}, wr={wr}")
assert not c and not l, "Should be non-causal, non-local after fix"

c, l, wl, wr = simulate_window_logic_with_fix(True, -1, -1)
print(f"causal+(-1,-1): causal={c}, local={l}, wl={wl}, wr={wr}")
assert c and not l, "Should be causal, non-local after fix"

# Verify real sliding window still works
c, l, wl, wr = simulate_window_logic_with_fix(False, 128, 0)
print(f"(128, 0):      causal={c}, local={l}, wl={wl}, wr={wr}")
assert not c and l, "Should be local with real window sizes"

c, l, wl, wr = simulate_window_logic_with_fix(False, 64, 64)
print(f"(64, 64):      causal={c}, local={l}, wl={wl}, wr={wr}")
assert not c and l, "Should be local with real window sizes"

# Edge case: one side -1, other side has a value
c, l, wl, wr = simulate_window_logic_with_fix(False, -1, 0)
print(f"(-1, 0):       causal={c}, local={l}, wl={wl}, wr={wr}")
assert c and not l, "(-1, 0) should be causal (window_left=None, window_right=0)"

print()
print("All assertions passed!")
