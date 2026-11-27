# Copyright (c) 2024, Tri Dao.

import ast
import inspect
import pytest


def test_sm90_bwd_launch_pdl_setting():
    """Verify FlashAttentionBackwardSm90 disables PDL when ragged_kv_tma is True

    to avoid workspace ordering hazards between preceding descriptor fills and consumer TMA initialization (#3198).
    """
    import os

    source_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "flash_attn", "cute", "flash_bwd_sm90.py"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse AST and look for .launch(..., use_pdl=...)
    tree = ast.parse(source)

    launch_use_pdl_values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "launch":
                for kw in node.keywords:
                    if kw.arg == "use_pdl":
                        launch_use_pdl_values.append(ast.unparse(kw.value))

    assert len(launch_use_pdl_values) > 0, "Could not find .launch(..., use_pdl=...) in flash_bwd_sm90.py"
    # Ensure use_pdl is gated on not self.ragged_kv_tma
    assert any("not self.ragged_kv_tma" in val for val in launch_use_pdl_values), (
        f"Expected 'use_pdl=not self.ragged_kv_tma' but found: {launch_use_pdl_values}"
    )
