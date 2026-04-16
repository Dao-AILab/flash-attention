#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

TRACE_RE = re.compile(
    r"\[CLC\]\s+query\s+sm=(?P<sm>\d+)\s+cta=(?P<cta>\d+)\s+"
    r"\(m_blk=(?P<m_blk>-?\d+),h=(?P<h>-?\d+),b=(?P<b>-?\d+),s=(?P<s>-?\d+)\)\s+"
    r"valid=(?P<valid>[01])"
)


@dataclass(frozen=True)
class TraceRow:
    sm: int
    cta: int
    m_blk: int
    h: int
    b: int
    s: int
    valid: int


def parse_rows(text: str) -> list[TraceRow]:
    rows: list[TraceRow] = []
    for line in text.splitlines():
        match = TRACE_RE.search(line)
        if match is None:
            continue
        rows.append(TraceRow(**{key: int(value) for key, value in match.groupdict().items()}))
    return rows


def summarize(rows: list[TraceRow]) -> dict:
    by_sm: dict[int, list[TraceRow]] = defaultdict(list)
    by_cta: dict[int, list[TraceRow]] = defaultdict(list)
    tile_counter: Counter[tuple[int, int, int, int, int]] = Counter()
    for row in rows:
        by_sm[row.sm].append(row)
        by_cta[row.cta].append(row)
        tile_counter[(row.m_blk, row.h, row.b, row.s, row.valid)] += 1

    def encode_group(grouped: dict[int, list[TraceRow]]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key, group_rows in sorted(grouped.items()):
            out[str(key)] = {
                "count": len(group_rows),
                "valid_count": sum(row.valid for row in group_rows),
                "invalid_count": sum(1 - row.valid for row in group_rows),
                "first": asdict(group_rows[0]),
                "last": asdict(group_rows[-1]),
                "unique_tiles": len({(r.m_blk, r.h, r.b, r.s, r.valid) for r in group_rows}),
            }
        return out

    top_tiles = [
        {
            "tile": {
                "m_blk": tile[0],
                "h": tile[1],
                "b": tile[2],
                "s": tile[3],
                "valid": tile[4],
            },
            "count": count,
        }
        for tile, count in tile_counter.most_common(20)
    ]

    return {
        "rows": len(rows),
        "valid_rows": sum(row.valid for row in rows),
        "invalid_rows": sum(1 - row.valid for row in rows),
        "unique_sms": len(by_sm),
        "unique_ctas": len(by_cta),
        "by_sm": encode_group(by_sm),
        "by_cta": encode_group(by_cta),
        "top_tiles": top_tiles,
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"rows={summary['rows']} valid={summary['valid_rows']} invalid={summary['invalid_rows']}",
        f"unique_sms={summary['unique_sms']} unique_ctas={summary['unique_ctas']}",
        "top_tiles:",
    ]
    for entry in summary["top_tiles"][:10]:
        tile = entry["tile"]
        lines.append(
            f"  count={entry['count']:>4} tile=(m_blk={tile['m_blk']}, h={tile['h']}, b={tile['b']}, s={tile['s']}, valid={tile['valid']})"
        )
    lines.append("by_sm:")
    for sm, sm_summary in summary["by_sm"].items():
        first = sm_summary["first"]
        last = sm_summary["last"]
        lines.append(
            f"  sm={sm:>3} count={sm_summary['count']:>4} valid={sm_summary['valid_count']:>4} invalid={sm_summary['invalid_count']:>4} "
            f"first=(cta={first['cta']},m_blk={first['m_blk']},h={first['h']},b={first['b']},s={first['s']},v={first['valid']}) "
            f"last=(cta={last['cta']},m_blk={last['m_blk']},h={last['h']},b={last['b']},s={last['s']},v={last['valid']})"
        )
    return "\n".join(lines)


def visualize_html(rows: list[TraceRow], summary: dict) -> str:
    by_sm: dict[int, list[TraceRow]] = defaultdict(list)
    for row in rows:
        by_sm[row.sm].append(row)

    data = [
        {
            "sm": sm,
            "tiles": [
                {
                    "id": r.m_blk,
                    "type": "INIT" if idx == 0 else "PULL",
                    "valid": bool(r.valid),
                    "m": r.m_blk,
                    "h": r.h,
                    "b": r.b,
                    "s": r.s,
                    "cta": r.cta,
                }
                for idx, r in enumerate(chain)
            ],
        }
        for sm, chain in sorted(by_sm.items())
    ]

    total_tiles = sum(len(d["tiles"]) for d in data)
    valid_pulls = sum(1 for d in data for t in d["tiles"] if t["type"] == "PULL" and t["valid"])
    work_per_sm = [sum(1 for t in d["tiles"] if t["valid"]) for d in data]
    histogram = defaultdict(int)
    for work in work_per_sm:
        histogram[work] += 1
    histogram_data = [{"work": k, "count": v} for k, v in sorted(histogram.items())]
    work_stats = {
        "min": min(work_per_sm) if work_per_sm else 0,
        "max": max(work_per_sm) if work_per_sm else 0,
        "mean": (sum(work_per_sm) / len(work_per_sm)) if work_per_sm else 0.0,
        "std": (
            sum((w - sum(work_per_sm) / len(work_per_sm)) ** 2 for w in work_per_sm) / len(work_per_sm)
        ) ** 0.5 if work_per_sm else 0.0,
    }

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<title>CLC Work Distribution Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 16px; }}
h1 {{ font-size: 1.4rem; margin-bottom: 12px; color: #fff; }}
.stats {{ display: flex; gap: 24px; margin-bottom: 16px; font-size: 0.85rem; color: #aaa; flex-wrap: wrap; }}
.stats span {{ background: #252540; padding: 6px 12px; border-radius: 4px; }}
.search-container {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
#search {{ padding: 8px 12px; font-size: 1rem; border: 1px solid #444; border-radius: 4px; background: #252540; color: #fff; width: 220px; }}
#search:focus {{ outline: none; border-color: #87CEEB; }}
#match-count {{ font-size: 0.85rem; color: #888; }}
.legend {{ display: flex; gap: 16px; margin-bottom: 16px; font-size: 0.8rem; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-box {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid #444; }}
.container {{ display: flex; gap: 20px; }}
#overview {{ flex: 1; max-height: 80vh; overflow-y: auto; }}
#detail {{ width: 360px; background: #252540; border-radius: 8px; padding: 16px; display: none; max-height: 80vh; overflow-y: auto; }}
#detail.visible {{ display: block; }}
#detail h2 {{ font-size: 1.1rem; margin-bottom: 12px; color: #87CEEB; }}
.detail-tiles {{ display: flex; flex-direction: column; gap: 8px; }}
.detail-tile {{ background: #1a1a2e; padding: 10px; border-radius: 6px; border-left: 4px solid #87CEEB; }}
.detail-tile.pulled {{ border-left-color: #FA8072; }}
.detail-tile.invalid {{ border-left-color: #666; opacity: 0.6; }}
.detail-tile-header {{ font-weight: 600; margin-bottom: 4px; }}
.detail-tile-meta {{ font-size: 0.8rem; color: #888; }}
.sm-row {{ display: flex; align-items: center; padding: 4px 8px; border-radius: 4px; cursor: pointer; margin-bottom: 2px; }}
.sm-row:hover {{ background: #252540; }}
.sm-row.selected {{ background: #303050; }}
.sm-label {{ width: 60px; font-size: 0.8rem; font-weight: 600; color: #888; flex-shrink: 0; }}
.tiles {{ display: flex; gap: 3px; flex-wrap: wrap; }}
.tile {{ width: 28px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; border-radius: 3px; border: 1px solid #444; transition: transform 0.1s, box-shadow 0.1s; }}
.tile.init {{ background: #87CEEB; color: #000; }}
.tile.pull {{ background: #FA8072; color: #000; }}
.tile.invalid {{ background: #444; color: #888; }}
.tile.highlight {{ transform: scale(1.3); box-shadow: 0 0 8px #fff; z-index: 10; position: relative; }}
.tile.dim {{ opacity: 0.3; }}
kbd {{ background: #333; padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; margin-left: 8px; }}
.histogram-section {{ background: #252540; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.histogram-section h2 {{ font-size: 1rem; margin-bottom: 12px; color: #87CEEB; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
.histogram-stats {{ font-size: 0.8rem; color: #888; font-weight: normal; }}
#histogram {{ display: flex; align-items: flex-end; gap: 2px; height: 120px; padding: 8px 0; }}
.hist-bar-container {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 20px; max-width: 40px; }}
.hist-bar {{ background: linear-gradient(to top, #FA8072, #87CEEB); border-radius: 3px 3px 0 0; width: 100%; transition: opacity 0.2s; cursor: pointer; }}
.hist-bar:hover {{ opacity: 0.8; }}
.hist-label {{ font-size: 0.65rem; color: #888; margin-top: 4px; }}
.hist-count {{ font-size: 0.6rem; color: #aaa; margin-bottom: 2px; }}
.note {{ margin-top: 12px; color: #999; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>CLC Work Distribution Viewer</h1>
<div class=\"stats\">
  <span>query-trace mode</span>
  <span>SMs: {len(data)}</span>
  <span>Total queries: {total_tiles}</span>
  <span>Valid pulls: {valid_pulls}</span>
  <span>Invalid queries: {summary['invalid_rows']}</span>
</div>
<div class=\"search-container\">
  <input type=\"text\" id=\"search\" placeholder=\"Search m_blk...\" autofocus>
  <span id=\"match-count\"></span>
  <span style=\"color:#666\">Press <kbd>Esc</kbd> to clear</span>
</div>
<div class=\"legend\">
  <div class=\"legend-item\"><div class=\"legend-box\" style=\"background:#87CEEB\"></div>First query on SM</div>
  <div class=\"legend-item\"><div class=\"legend-box\" style=\"background:#FA8072\"></div>Later query / pull</div>
  <div class=\"legend-item\"><div class=\"legend-box\" style=\"background:#444\"></div>Invalid / exhausted</div>
</div>
<div class=\"histogram-section\">
  <h2>Work Distribution Histogram <span class=\"histogram-stats\">min={work_stats['min']}, max={work_stats['max']}, mean={work_stats['mean']:.1f}, std={work_stats['std']:.2f}</span></h2>
  <div id=\"histogram\"></div>
</div>
<div class=\"container\">
  <div id=\"overview\"></div>
  <div id=\"detail\">
    <h2>SM <span id=\"detail-sm\"></span></h2>
    <div class=\"detail-tiles\" id=\"detail-tiles\"></div>
  </div>
</div>
<script>
const data = {json.dumps(data)};
const histogramData = {json.dumps(histogram_data)};
const overview = document.getElementById('overview');
const detail = document.getElementById('detail');
const detailSm = document.getElementById('detail-sm');
const detailTiles = document.getElementById('detail-tiles');
const searchInput = document.getElementById('search');
const matchCount = document.getElementById('match-count');
let selectedSm = null;
function renderOverview() {{
  overview.innerHTML = '';
  data.forEach(sm => {{
    const row = document.createElement('div');
    row.className = 'sm-row' + (selectedSm === sm.sm ? ' selected' : '');
    row.dataset.sm = sm.sm;
    const label = document.createElement('div');
    label.className = 'sm-label';
    label.textContent = 'SM' + sm.sm;
    row.appendChild(label);
    const tiles = document.createElement('div');
    tiles.className = 'tiles';
    sm.tiles.forEach((t, idx) => {{
      const tile = document.createElement('div');
      tile.className = 'tile';
      if (!t.valid) tile.classList.add('invalid');
      else if (t.type === 'INIT') tile.classList.add('init');
      else tile.classList.add('pull');
      tile.textContent = t.valid ? t.id : 'X';
      tile.title = 'cta=' + t.cta + ' m_blk=' + t.m + ' h=' + t.h + ' b=' + t.b + ' s=' + t.s + ' valid=' + t.valid;
      tile.dataset.id = String(t.id);
      tile.dataset.sm = sm.sm;
      tile.dataset.idx = idx;
      tiles.appendChild(tile);
    }});
    row.appendChild(tiles);
    row.addEventListener('click', () => selectSm(sm.sm));
    overview.appendChild(row);
  }});
}}
function selectSm(sm) {{
  selectedSm = sm;
  const smData = data.find(d => d.sm === sm);
  if (!smData) return;
  document.querySelectorAll('.sm-row').forEach(r => {{
    r.classList.toggle('selected', parseInt(r.dataset.sm) === sm);
  }});
  detailSm.textContent = sm;
  detailTiles.innerHTML = '';
  smData.tiles.forEach((t, idx) => {{
    const div = document.createElement('div');
    div.className = 'detail-tile';
    if (!t.valid) div.classList.add('invalid');
    else if (t.type === 'PULL') div.classList.add('pulled');
    div.innerHTML = `
      <div class=\"detail-tile-header\">${{idx === 0 ? 'Initial query' : 'Query #' + idx}}: m_blk ${{t.id}}</div>
      <div class=\"detail-tile-meta\">cta=${{t.cta}}, h=${{t.h}}, b=${{t.b}}, s=${{t.s}}${{t.valid ? '' : ' (invalid / exhausted)'}} </div>
    `;
    detailTiles.appendChild(div);
  }});
  detail.classList.add('visible');
}}
function doSearch(query) {{
  const allTiles = document.querySelectorAll('.tile');
  if (!query) {{
    allTiles.forEach(t => {{ t.classList.remove('highlight', 'dim'); }});
    matchCount.textContent = '';
    return;
  }}
  let matches = 0;
  allTiles.forEach(t => {{
    const id = t.dataset.id;
    if (id === query || id.includes(query)) {{
      t.classList.add('highlight');
      t.classList.remove('dim');
      matches++;
    }} else {{
      t.classList.remove('highlight');
      t.classList.add('dim');
    }}
  }});
  matchCount.textContent = matches + ' match' + (matches !== 1 ? 'es' : '');
}}
searchInput.addEventListener('input', e => doSearch(e.target.value.trim()));
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    searchInput.value = '';
    doSearch('');
    detail.classList.remove('visible');
    selectedSm = null;
    document.querySelectorAll('.sm-row').forEach(r => r.classList.remove('selected'));
  }}
}});
function renderHistogram() {{
  const histogram = document.getElementById('histogram');
  const maxCount = Math.max(...histogramData.map(d => d.count));
  histogram.innerHTML = '';
  histogramData.forEach(d => {{
    const container = document.createElement('div');
    container.className = 'hist-bar-container';
    const count = document.createElement('div');
    count.className = 'hist-count';
    count.textContent = d.count;
    container.appendChild(count);
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = (d.count / maxCount * 100) + 'px';
    bar.title = d.count + ' SMs with ' + d.work + ' valid queries';
    container.appendChild(bar);
    const label = document.createElement('div');
    label.className = 'hist-label';
    label.textContent = d.work;
    container.appendChild(label);
    histogram.appendChild(container);
  }});
}}
renderOverview();
renderHistogram();
</script>
</body>
</html>
"""


def read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse FlashAttention CLC trace lines.")
    parser.add_argument("logfile", nargs="?", default="-", help="Trace log path or - for stdin")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--rows", action="store_true", help="Emit parsed rows as JSON")
    parser.add_argument("--html", action="store_true", help="Emit HTML view")
    parser.add_argument("-o", "--output", help="Output path for --html")
    args = parser.parse_args()

    rows = parse_rows(read_text(args.logfile))
    if args.rows:
        print(json.dumps([asdict(row) for row in rows], indent=2))
        return

    summary = summarize(rows)
    if args.html:
        html_text = visualize_html(rows, summary)
        if args.output is not None:
            Path(args.output).write_text(html_text)
        else:
            print(html_text)
        return
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
