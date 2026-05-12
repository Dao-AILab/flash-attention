#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"

TYPE="refactor"
MESSAGE=""
DRY_RUN=0

ALLOWED_REFACTOR_FILES=(
    "flash_attn/cute/sm100_hd256_2cta_fmha_forward.py"
    "flash_attn/cute/sm100_hd256_2cta_fmha_backward.py"
    "flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py"
    "flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py"
)

usage() {
    cat <<'EOF'
Usage:
  bash harness/harness/commit/commit.sh --type refactor -m "message"
  bash harness/harness/commit/commit.sh --type feature -m "message"
  bash harness/harness/commit/commit.sh --type refactor --dry-run -m "message"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            TYPE="${2:-}"
            shift 2
            ;;
        -m|--message)
            MESSAGE="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[commit] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$TYPE" != "refactor" && "$TYPE" != "feature" ]]; then
    echo "[commit] --type must be refactor or feature" >&2
    exit 2
fi

if [[ -z "$MESSAGE" ]]; then
    echo "[commit] commit message is required" >&2
    exit 2
fi

cd "$REPO"

is_allowed_refactor_file() {
    local path="$1"
    local allowed
    for allowed in "${ALLOWED_REFACTOR_FILES[@]}"; do
        if [[ "$path" == "$allowed" ]]; then
            return 0
        fi
    done
    return 1
}

configure_identity_if_needed() {
    local system_user git_user
    system_user="$(id -un 2>/dev/null || true)"
    git_user="$(git config --get user.name 2>/dev/null || true)"

    if [[ "$system_user" == "wangsiyu" || "$system_user" == "siyu.wsy" || "$git_user" == "wangsiyu" || "$git_user" == "siyu.wsy" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[commit] dry-run: would set git identity to wangsiyu <siyu.wsy@gmail.com>"
        else
            git config user.name "wangsiyu"
            git config user.email "siyu.wsy@gmail.com"
            echo "[commit] git identity set to wangsiyu <siyu.wsy@gmail.com>"
        fi
    fi
}

check_refactor_staged_scope() {
    local path
    while IFS= read -r path; do
        [[ -z "$path" ]] && continue
        if [[ "$path" == tests/* ]]; then
            echo "[commit] tests are forbidden in refactor commits: $path" >&2
            return 1
        fi
        if ! is_allowed_refactor_file "$path"; then
            echo "[commit] file outside refactor allowlist is staged: $path" >&2
            return 1
        fi
    done < <(git diff --cached --name-only)
}

configure_identity_if_needed

if [[ "$TYPE" == "refactor" ]]; then
    check_refactor_staged_scope
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[commit] dry-run: would stage refactor allowlist"
    else
        git add -- "${ALLOWED_REFACTOR_FILES[@]}"
    fi
    check_refactor_staged_scope
else
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[commit] dry-run: feature mode uses existing staged files"
    fi
fi

if ! git diff --cached --quiet; then
    echo "[commit] staged files:"
    git diff --cached --name-only | sed 's/^/  /'
else
    if [[ "$DRY_RUN" -eq 1 && "$TYPE" == "refactor" ]]; then
        echo "[commit] dry-run: no staged files; refactor scope checks passed"
        exit 0
    fi
    echo "[commit] no staged files to commit" >&2
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[commit] dry-run: checks passed"
    exit 0
fi

git commit -m "$MESSAGE"
