#!/usr/bin/env bash
# eval_all.sh — Budget Router Evaluator Wrapper
# ==============================================
# Runs heuristic + LLM eval and saves results to outputs/.
#
# Usage:
#   chmod +x eval_all.sh
#   ./eval_all.sh                      # quick: 3 seeds, heuristic + LLM
#   ./eval_all.sh --seeds 10           # full dev set
#   ./eval_all.sh --policies heuristic # no LLM (no API needed)
#   ./eval_all.sh --tasks hard hard_multi --seeds 5
#
# Prerequisites:
#   export HF_TOKEN=<your_huggingface_token>
#   export API_BASE_URL=https://router.huggingface.co/v1  (default)
#   export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct           (default)
#   uv or pip install -e . (to install budget_router package)
#
# Outputs (in outputs/ directory):
#   eval_results_<timestamp>.json    — full per-episode grader breakdown
#   eval_summary_<timestamp>.md      — markdown table ready for README

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
SEEDS=3
POLICIES="heuristic llm"
TASKS="easy medium hard hard_multi"
SEED_SET="dev"
OUT_DIR="$REPO_ROOT/outputs"
EXTRA_ARGS=()

# ── Parse CLI args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)      SEEDS="$2";    shift 2 ;;
        --seed-set)   SEED_SET="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2";  shift 2 ;;
        --policies)
            POLICIES=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                POLICIES="$POLICIES $1"; shift
            done
            ;;
        --tasks)
            TASKS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TASKS="$TASKS $1"; shift
            done
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Validate environment ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Budget Router Evaluator                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Config:"
echo "  Policies:  $POLICIES"
echo "  Tasks:     $TASKS"
echo "  Seeds:     $SEEDS (seed_set=$SEED_SET)"
echo "  Output:    $OUT_DIR/"
echo ""

# Check HF_TOKEN if LLM in policies
if echo "$POLICIES" | grep -q "llm"; then
    if [[ -z "${HF_TOKEN:-}" && -z "${API_KEY:-}" ]]; then
        echo "⚠️  WARNING: HF_TOKEN and API_KEY not set."
        echo "   LLM policy will be skipped. Set HF_TOKEN to enable."
        echo ""
    else
        TOKEN_PREVIEW="${HF_TOKEN:-${API_KEY:-}}"
        echo "  API key:   ${TOKEN_PREVIEW:0:8}... (${#TOKEN_PREVIEW} chars)"
        echo "  Model:     ${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
        echo "  Endpoint:  ${API_BASE_URL:-https://router.huggingface.co/v1}"
        echo ""
    fi
fi

# ── Build typer args ─────────────────────────────────────────────────────────
TYPER_ARGS=(
    "--seeds" "$SEEDS"
    "--seed-set" "$SEED_SET"
    "--out-dir" "$OUT_DIR"
)

for p in $POLICIES; do
    TYPER_ARGS+=("--policies" "$p")
done

for t in $TASKS; do
    TYPER_ARGS+=("--tasks" "$t")
done

# ── Run ──────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

if command -v uv &>/dev/null; then
    uv run python eval_all.py "${TYPER_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
elif command -v python3 &>/dev/null; then
    python3 eval_all.py "${TYPER_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "Error: neither uv nor python3 found." >&2
    exit 1
fi

echo ""
echo "✅ Evaluation complete. Results in $OUT_DIR/"