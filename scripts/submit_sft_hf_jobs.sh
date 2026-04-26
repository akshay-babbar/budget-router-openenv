#!/usr/bin/env bash
set -euo pipefail

# Local data generation reads tokens from the current terminal environment.
# Default teacher is PPO, so this costs 0 large-LLM calls.
: "${HF_TOKEN:?HF_TOKEN must be set in this terminal}"

export TEACHER_POLICY="${TEACHER_POLICY:-ppo}"
export DATASET_REPO="${DATASET_REPO:-akshay4/budget-router-sft-data}"
export OUTPUT_REPO="${OUTPUT_REPO:-akshay4/budget-router-sft-qwen1.5b}"
export SFT_MODEL_REPO="${SFT_MODEL_REPO:-$OUTPUT_REPO}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export TASK_NAME="${TASK_NAME:-hard_multi}"
export SFT_START_SEED="${SFT_START_SEED:-1000}"
export SFT_N_EPISODES="${SFT_N_EPISODES:-100}"
export SFT_TOP_FRACTION="${SFT_TOP_FRACTION:-0.30}"
export SFT_MIN_KEEP="${SFT_MIN_KEEP:-20}"
export SFT_MIN_DELTA="${SFT_MIN_DELTA:-0.0}"
export PPO_MODEL_PATH="${PPO_MODEL_PATH:-trained_models/ppo_hard_multi_100k.zip}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export N_SEEDS="${N_SEEDS:-10}"
export EVAL_SEED_VALUES="${EVAL_SEED_VALUES:-}"
export HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-a10g-large}"
export HF_JOB_NAMESPACE="${HF_JOB_NAMESPACE:-${OUTPUT_REPO%%/*}}"
export TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-2h}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1h}"

if [[ "${TEACHER_POLICY}" == "llm" ]]; then
  calls=$((SFT_N_EPISODES * 20))
  echo "[submit-sft] WARNING: teacher=llm will make up to ${calls} large-model calls."
  echo "[submit-sft] Default teacher=ppo avoids this cost."
fi

if [[ "${SKIP_DATA_GENERATION:-0}" != "1" ]]; then
  echo "[submit-sft] Step 1: generating and pushing SFT data locally..."
  if [[ "${TEACHER_POLICY}" == "ppo" ]]; then
    uv run --extra training --with datasets --with huggingface_hub python generate_sft_data.py
  else
    uv run --with datasets --with huggingface_hub python generate_sft_data.py
  fi
else
  echo "[submit-sft] Step 1 skipped: using existing dataset ${DATASET_REPO}"
fi

echo "[submit-sft] Step 2/3: submitting train and eval as Hugging Face Jobs..."
uv run --with "huggingface_hub>=1.0.0" python - <<'PY'
import os
import time
from huggingface_hub import inspect_job, run_uv_job


def wait_for_job(job, timeout_label: str, namespace: str) -> None:
    print(f"[submit-sft] job_url={job.url}", flush=True)
    while True:
        info = inspect_job(job_id=job.id, namespace=namespace)
        stage = str(info.status.stage)
        print(f"[submit-sft] job={job.id} stage={stage}", flush=True)
        if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
            if stage != "COMPLETED":
                raise SystemExit(f"HF Job {job.id} ended with stage={stage}; see {job.url}")
            return
        time.sleep(30)


token = os.environ["HF_TOKEN"]
common_secret = {"HF_TOKEN": token}
flavor = os.environ["HF_JOB_FLAVOR"]
namespace = os.environ["HF_JOB_NAMESPACE"]

train_args = [
    "--base-model", os.environ["BASE_MODEL"],
    "--dataset-repo", os.environ["DATASET_REPO"],
    "--output-repo", os.environ["OUTPUT_REPO"],
    "--num-epochs", os.environ["NUM_EPOCHS"],
    "--learning-rate", os.environ["LEARNING_RATE"],
    "--lora-r", os.environ["LORA_R"],
    "--lora-alpha", os.environ["LORA_ALPHA"],
    "--max-length", os.environ["MAX_SEQ_LENGTH"],
]
print(f"[submit-sft] launching train_sft.py on {flavor}", flush=True)
train_job = run_uv_job(
    "train_sft.py",
    script_args=train_args,
    flavor=flavor,
    namespace=namespace,
    secrets=common_secret,
    timeout=os.environ["TRAIN_TIMEOUT"],
)
wait_for_job(train_job, os.environ["TRAIN_TIMEOUT"], namespace)

eval_args = [
    "--model-repo", os.environ["SFT_MODEL_REPO"],
    "--task", os.environ["TASK_NAME"],
    "--n-seeds", os.environ["N_SEEDS"],
]
if os.environ.get("EVAL_SEED_VALUES"):
    eval_args.extend(["--seed-values", os.environ["EVAL_SEED_VALUES"]])

print(f"[submit-sft] launching eval_sft.py on {flavor}", flush=True)
eval_job = run_uv_job(
    "eval_sft.py",
    script_args=eval_args,
    flavor=flavor,
    namespace=namespace,
    secrets=common_secret,
    timeout=os.environ["EVAL_TIMEOUT"],
)
wait_for_job(eval_job, os.environ["EVAL_TIMEOUT"], namespace)
print(f"[submit-sft] Pipeline complete. Model: {os.environ['OUTPUT_REPO']}", flush=True)
PY
