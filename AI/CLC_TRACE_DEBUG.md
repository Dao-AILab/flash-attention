# CLC Trace Debugging

Use this when you suspect the CLC work scheduler is making surprising tile assignment decisions and you want a raw scheduler trace from the current kernel.

## Current trace format

SM100 forward kernels emit one trace line per scheduler-warp query at `FA_LOG_LEVEL=3`:

```text
[CLC] query sm=<smid> cta=<blockIdx.x> (m_blk=<m>,h=<h>,b=<b>,s=<s>) valid=<0|1>
```

Current emit sites:
- `flash_attn/cute/flash_fwd_sm100.py`
- `flash_attn/cute/flash_fwd_mla_sm100.py`

## How to capture a trace

Important:
- `FA_LOG_LEVEL=3` is needed for the `[CLC] query ...` device-side prints. The scheduler warp
  emits them; kernels with `clc_late_claim` (1CTA LPT, no SplitKV, one load warp, the default
  dense causal case) claim from the load warp and emit no query records. To trace claims, use a
  2CTA or SplitKV CLC config, which keeps the scheduler-warp claim.
- Dense causal forward uses CLC by default on the measured cells in `NOTE [causal CLC]`
  (`flash_attn/cute/config.py`). Elsewhere, pass an explicit `FwdConfig` with
  `use_clc_scheduler=True` to `flash_attn.cute.interface._flash_attn_fwd(..., config=...)`;
  validation raises `ValueError` if the problem cannot run CLC.

Minimal repro pattern:

```bash
FA_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=0 python - <<'PY' \
  > agent_space/clc_trace.log 2>&1
import torch
from flash_attn.cute.interface import flash_attn_func

torch.manual_seed(0)
q = torch.randn(1, 512, 16, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
PY
```

If you want the run to say explicitly whether CLC was selected, keep the host log prefix too:

```text
[FA] TileScheduler=SingleTileLPTScheduler, scheduling_mode=CLC, USE_2CTA=False
```

## What to look for

- `scheduling_mode=CLC` in host logs confirms the shape actually used the CLC path.
- `valid=1` means the returned work tile is valid.
- `valid=0` means the scheduler is exhausted for that CTA/scheduler warp query.
- `m_blk`, `h`, `b`, `s` are the logical work coordinates after the scheduler mapping.
- `cta` is the physical `blockIdx.x`; for clustered launches multiple CTAs may participate in the same logical tile.

## Parse the trace

A lightweight parser lives in `AI/parse_clc_log.py`.

Text summary:

```bash
python AI/parse_clc_log.py agent_space/clc_trace.log
```

HTML view:

```bash
python AI/parse_clc_log.py agent_space/clc_trace.log --html -o agent_space/clc_trace.html
```

## Suggested workflow

1. Reproduce the surprising case with `FA_LOG_LEVEL=3` on a CLC config (default or explicit).
2. Save stdout/stderr to `agent_space/clc_trace.log`.
3. Run `AI/parse_clc_log.py` on that log to get a compact per-SM / per-CTA summary.
4. If the trace still looks suspicious, attach or paste that log in the investigation thread / agent notes.
5. Compare against the relevant mapping logic in `flash_attn/cute/tile_scheduler.py`.

## Caveats

- The trace is noisy and expensive; use a single small shape first.
- Because the print happens on scheduler queries, many lines may be terminal `valid=0` queries after work is exhausted.
- Default selection runs CLC only where `flash_attn/cute/config.py` measured a win; other shapes need an explicit config.
