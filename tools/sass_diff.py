#!/usr/bin/env python3
"""Compare two SASS files, ignoring register assignments and addresses.

Normalizes registers per-instruction so that two instructions doing the
same operation with different register allocations compare as equal.
E.g. "UIADD3 UR30, UP1, UR30, 0x70, URZ" and
     "UIADD3 UR14, UP1, UR38, 0x70, URZ" both normalize to
     "UIADD3 UR_0, UP_0, UR_1, 0x70, URZ"

Usage:
    python scripts/sass_diff.py file_a.sass file_b.sass
    python scripts/sass_diff.py file_a.sass file_b.sass --context 5
    python scripts/sass_diff.py file_a.sass file_b.sass --all       # include metadata
    python scripts/sass_diff.py file_a.sass file_b.sass --summary-only
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher


# ── Parsing ──────────────────────────────────────────────────────────────────

ADDR_LINE_RE = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*$")
LABEL_RE = re.compile(r"^(\.L_x_\d+):\s*$")
METADATA_PREFIXES = (".byte", ".word", ".short", ".dword", ".string", ".align")

# Register pattern: match UR before R, UP before P
REG_RE = re.compile(r"\b(UP|UR|P|R)(\d+)\b")


@dataclass
class Line:
    """One parsed SASS line."""
    addr: str           # hex address or "" for labels
    raw: str            # original text (no addr prefix)
    normalized: str     # register-normalized for comparison
    lineno: int         # 1-based line number in file
    is_code: bool       # True for instructions/labels


def _normalize_instr(text: str) -> str:
    """Normalize one instruction by replacing registers with positional IDs.

    Each register class (R, UR, P, UP) gets its own counter, reset per
    instruction. Constants RZ, URZ, PT, UPT are preserved.
    """
    counters: dict[str, int] = {}
    mapping: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        name = m.group(0)
        if name in ("RZ", "URZ", "PT", "UPT"):
            return name
        if name in mapping:
            return mapping[name]
        prefix = m.group(1)
        idx = counters.get(prefix, 0)
        counters[prefix] = idx + 1
        mapping[name] = f"{prefix}_{idx}"
        return mapping[name]

    return REG_RE.sub(repl, text)


def parse_sass(path: str) -> list[Line]:
    """Extract instruction, label, and metadata lines from a SASS file."""
    lines: list[Line] = []

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.rstrip()

            m = LABEL_RE.match(raw)
            if m:
                label = m.group(1)
                lines.append(Line("", label, label, lineno, True))
                continue

            m = ADDR_LINE_RE.match(raw)
            if m:
                addr, text = m.group(1), m.group(2).strip()
                is_meta = any(text.startswith(p) for p in METADATA_PREFIXES)
                normalized = text if is_meta else _normalize_instr(text)
                lines.append(Line(addr, text, normalized, lineno, not is_meta))

    return lines


# ── Diffing ──────────────────────────────────────────────────────────────────

@dataclass
class DiffBlock:
    tag: str  # "equal", "replace", "insert", "delete"
    a_lines: list[Line] = field(default_factory=list)
    b_lines: list[Line] = field(default_factory=list)


def diff_sass(a_lines: list[Line], b_lines: list[Line]) -> list[DiffBlock]:
    a_norm = [l.normalized for l in a_lines]
    b_norm = [l.normalized for l in b_lines]
    sm = SequenceMatcher(None, a_norm, b_norm, autojunk=False)
    blocks: list[DiffBlock] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        blocks.append(DiffBlock(tag, a_lines[i1:i2], b_lines[j1:j2]))
    return blocks


# ── Display ──────────────────────────────────────────────────────────────────

RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def _fmt(line: Line, prefix: str, color: str, use_color: bool, show_norm: bool) -> str:
    addr = f"[{line.addr}]" if line.addr else "       "
    text = line.normalized if show_norm else line.raw
    if use_color:
        return f"{color}{prefix} {addr:>8s}  {text}{RESET}"
    return f"{prefix} {addr:>8s}  {text}"


def print_diff(blocks: list[DiffBlock], context: int = 3,
               use_color: bool = True, show_norm: bool = False):
    """Unified-diff-style output with context."""
    groups: list[list[str]] = []
    cur: list[str] = []
    last_changed = False

    for block in blocks:
        if block.tag == "equal":
            lines = block.a_lines
            if last_changed:
                for l in lines[:context]:
                    cur.append(_fmt(l, " ", DIM, use_color, show_norm))
                if len(lines) > 2 * context:
                    if cur:
                        groups.append(cur)
                    cur = []
                    for l in lines[-context:]:
                        cur.append(_fmt(l, " ", DIM, use_color, show_norm))
                elif len(lines) > context:
                    for l in lines[context:]:
                        cur.append(_fmt(l, " ", DIM, use_color, show_norm))
            else:
                for l in lines[-context:]:
                    cur.append(_fmt(l, " ", DIM, use_color, show_norm))
            last_changed = False
        else:
            last_changed = True
            if block.tag in ("replace", "delete"):
                for l in block.a_lines:
                    cur.append(_fmt(l, "-", RED, use_color, show_norm))
            if block.tag in ("replace", "insert"):
                for l in block.b_lines:
                    cur.append(_fmt(l, "+", GREEN, use_color, show_norm))

    if cur:
        groups.append(cur)

    sep = f"{CYAN}{'─' * 72}{RESET}" if use_color else "─" * 72
    for i, g in enumerate(groups):
        if i > 0:
            print(sep)
        for line in g:
            print(line)


def _get_opcode(raw: str) -> str | None:
    """Extract opcode from instruction, skipping predicates and labels."""
    for p in raw.split():
        if p.startswith("@") or p.startswith(".L_"):
            continue
        return p
    return None


def print_summary(a_all: list[Line], b_all: list[Line], blocks: list[DiffBlock]):
    a_code = [l for l in a_all if l.is_code]
    b_code = [l for l in b_all if l.is_code]

    n_equal = sum(len(b.a_lines) for b in blocks if b.tag == "equal")
    n_delete = sum(len(b.a_lines) for b in blocks if b.tag in ("replace", "delete"))
    n_insert = sum(len(b.b_lines) for b in blocks if b.tag in ("replace", "insert"))
    n_changed = sum(1 for b in blocks if b.tag != "equal")

    print(f"  File A: {len(a_code)} instructions")
    print(f"  File B: {len(b_code)} instructions")
    print(f"  Identical (normalized): {n_equal}")
    print(f"  Changed regions: {n_changed}")
    print(f"  Removed: {n_delete}, Added: {n_insert}")

    def opcode_counts(lines):
        counts: dict[str, int] = {}
        for l in lines:
            op = _get_opcode(l.raw)
            if op:
                counts[op] = counts.get(op, 0) + 1
        return counts

    a_ops, b_ops = opcode_counts(a_code), opcode_counts(b_code)
    all_ops = sorted(set(a_ops) | set(b_ops))
    diffs = {op: b_ops.get(op, 0) - a_ops.get(op, 0) for op in all_ops}
    diffs = {op: d for op, d in diffs.items() if d != 0}
    if diffs:
        print("\n  Opcode count changes (B - A):")
        for op, d in sorted(diffs.items(), key=lambda x: -abs(x[1])):
            sign = "+" if d > 0 else ""
            print(f"    {op:30s} {sign}{d}")
    else:
        print("\n  Opcode counts: identical")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare SASS files ignoring register assignments")
    p.add_argument("file_a", help="First SASS file")
    p.add_argument("file_b", help="Second SASS file")
    p.add_argument("-C", "--context", type=int, default=3, help="Context lines (default: 3)")
    p.add_argument("--no-color", action="store_true", help="Disable color output")
    p.add_argument("--summary-only", action="store_true", help="Only print summary")
    p.add_argument("--all", action="store_true", help="Include metadata in diff")
    p.add_argument("--show-normalized", action="store_true",
                   help="Show normalized form instead of raw instructions")
    args = p.parse_args()

    a_all = parse_sass(args.file_a)
    b_all = parse_sass(args.file_b)

    if args.all:
        a_lines, b_lines = a_all, b_all
    else:
        a_lines = [l for l in a_all if l.is_code]
        b_lines = [l for l in b_all if l.is_code]

    blocks = diff_sass(a_lines, b_lines)
    use_color = not args.no_color and sys.stdout.isatty()

    print("=== Summary ===")
    print_summary(a_all, b_all, blocks)

    if not args.summary_only:
        print("\n=== Diff (registers normalized) ===\n")
        print_diff(blocks, args.context, use_color, args.show_normalized)


if __name__ == "__main__":
    main()
