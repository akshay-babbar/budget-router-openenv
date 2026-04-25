#!/usr/bin/env bash
set -euo pipefail

# Canonical Google Colab launcher for the GRPO Budget Router experiment.
# Run from the repository root after cloning:
#
#   bash scripts/run_colab_grpo.sh
#
# Optional overrides:
#   MODEL_NAME=Qwen/Qwen3-0.6B MAX_STEPS=30 bash scripts/run_colab_grpo.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
MAX_STEPS="${MAX_STEPS:-60}"
DATASET_N="${DATASET_N:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-0.95}"
PROMPT_STYLE="${PROMPT_STYLE:-explore}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3500}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_DIR="${LOG_DIR:-outputs}"

mkdir -p "$LOG_DIR" .colab_runtime

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -q uv
fi

uv sync --extra grpo --extra training --extra dev

echo "== GPU / dtype check =="
CUDA_BF16_SUPPORTED="$(
  uv run python - <<'PY'
import torch
print(bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
PY
)"

TRAIN_SCRIPT="train/learn_experiment.py"
if [[ "$CUDA_BF16_SUPPORTED" != "True" ]]; then
  TRAIN_SCRIPT=".colab_runtime/learn_experiment_colab.py"
  uv run python - <<'PY'
from pathlib import Path

src = Path("train/learn_experiment.py")
dst = Path(".colab_runtime/learn_experiment_colab.py")
text = src.read_text()
old = '    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32'
new = '''    dtype = (
        torch.bfloat16
        if device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
        if device == "cuda"
        else torch.float32
    )'''
if old not in text:
    raise SystemExit("Expected dtype line not found; refusing to patch temporary Colab trainer.")
dst.write_text(text.replace(old, new))
print(f"Using temporary Colab-safe trainer: {dst}")
PY
else
  echo "CUDA bf16 is supported; using canonical train/learn_experiment.py directly."
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
LOG_FILE="$LOG_DIR/grpo_colab_${SAFE_MODEL_NAME}_steps${MAX_STEPS}_${STAMP}.log"

echo "== Launching GRPO =="
echo "model=$MODEL_NAME steps=$MAX_STEPS generations=$NUM_GENERATIONS max_completion_length=$MAX_COMPLETION_LENGTH"
echo "log=$LOG_FILE"

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python "$TRAIN_SCRIPT" \
  --model-name "$MODEL_NAME" \
  --max-steps "$MAX_STEPS" \
  --dataset-n "$DATASET_N" \
  --save-steps "$SAVE_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --prompt-style "$PROMPT_STYLE" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  2>&1 | tee "$LOG_FILE"
