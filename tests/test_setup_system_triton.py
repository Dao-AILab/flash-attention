from pathlib import Path


def test_rocm_triton_backend_can_keep_system_triton():
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    source = setup_py.read_text()

    assert "FLASH_ATTENTION_USE_SYSTEM_TRITON" in source
    assert "USE_SYSTEM_TRITON_REQUESTED" not in source
    assert (
        'USE_SYSTEM_TRITON = (\n'
        '    ROCM_BACKEND == "triton" and os.getenv("FLASH_ATTENTION_USE_SYSTEM_TRITON", "FALSE") == "TRUE"\n'
        ')'
    ) in source
    assert 'env["AITER_USE_SYSTEM_TRITON"] = "1"' in source
    assert "if not USE_SYSTEM_TRITON:" in source
    assert '"triton>=3.6.0" if sys.platform != "win32" else "triton-windows>=3.6.0"' in source
