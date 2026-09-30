#!/usr/bin/env bash
# Fast two-pass testing: compile every kernel of a test selection in parallel under
# FakeTensorMode (no GPU work), then execute with the persistent compile cache, failing any
# test that still compiles (fake / real compile-key drift, or a test that skips a variant in
# fake mode). See CLAUDE.md "Fast two-pass testing".
#
#   tools/two_pass_tests.sh -k "mla_sparse" -f tests/cute/test_flash_attn_mla.py [-g 0,3] [-n 48] [-- extra pytest args]
#
# -g  GPUs for both passes; default: CUDA_VISIBLE_DEVICES or 0. The execution pass runs one
#     xdist worker per GPU. The compile pass does no GPU work, but each worker still opens a
#     CUDA context (a few hundred MB), round-robin over these GPUs only.
# -n  compile-pass workers; default 48
# -f  test file(s); default tests/cute/test_flash_attn.py (MLA: tests/cute/test_flash_attn_mla.py)
set -euo pipefail
K="" GPUS="${CUDA_VISIBLE_DEVICES:-0}" N=48 FILES="tests/cute/test_flash_attn.py"
while getopts "k:g:n:f:" opt; do
  case $opt in
    k) K="$OPTARG" ;; g) GPUS="$OPTARG" ;; n) N="$OPTARG" ;; f) FILES="$OPTARG" ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ -n "$K" ] || { echo "usage: $0 -k EXPR [-g GPUS] [-n N] [-f FILES] [-- pytest args]" >&2; exit 2; }
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
# --disable-warnings: the DSL emits thousands of deprecation warnings per run; failures and
# their tracebacks are still printed. A failed pass exits the script (set -e), so pass 2 never
# runs after a failed compile pass.
echo "== pass 1: compile (FakeTensorMode, -n $N)"
CUDA_VISIBLE_DEVICES="$GPUS" FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q -rfE --disable-warnings -n "$N" $FILES -k "$K" -p no:cacheprovider "$@"
echo "== pass 2: execute (GPUs $GPUS, -n $NGPU, compiles fail the test)"
CUDA_VISIBLE_DEVICES="$GPUS" FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_TEST_EXPECT_CACHED=1 \
  pytest -q -rfE --disable-warnings -n "$NGPU" $FILES -k "$K" -p no:cacheprovider "$@"
