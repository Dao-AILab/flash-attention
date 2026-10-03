import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("module_dir", ["utils", "cute"])
@pytest.mark.parametrize("nbytes", [0, 16 * 1024 * 1024])
def test_benchmark_memory_reports_decimal_gigabytes(module_dir, nbytes, capsys):
    # These Torch-only utilities need no compiled FlashAttention extension.
    path = (
        Path(__file__).resolve().parents[1] / "flash_attn" / module_dir / "benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(f"benchmark_{module_dir}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    value = module.benchmark_memory(
        lambda: torch.empty(nbytes, dtype=torch.uint8, device="cuda"), desc="allocation"
    )
    peak_bytes = torch.cuda.max_memory_allocated()
    assert peak_bytes >= nbytes
    assert value == peak_bytes / 1e9
    assert capsys.readouterr().out == f"allocation max memory: {value}GB\n"
