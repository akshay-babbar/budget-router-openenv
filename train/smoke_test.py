"""
GRPO Smoke Test — 10 gradient steps, M4 Mac MPS (or CUDA/CPU).

PURPOSE
    Validate the full TRL training loop (model → rollout → reward → gradient)
    works end-to-end with BudgetRouterGRPOEnv before a full training run.
    NOT for actual learning — 10 steps is statistical noise.

USAGE
    Requires optional GRPO deps (`uv sync --extra grpo`), then e.g.:

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train/smoke_test.py

EXPECTED RUNTIME
    ~5-10 min on M4 Mac 48 GB (MPS, Qwen2.5-0.5B-Instruct)

HYPERPARAMETERS (source)
    - learning_rate, beta, temperature: DeepSeek-R1 GRPO paper + TRL Wordle example
    - num_generations=4: minimum GRPO group; 8+ for real training
    - max_completion_length=512: enough for ~10 multi-turn tool calls at 0.5B
    - optim=adamw_torch: paged_adamw_8bit is CUDA-only
    - No vLLM, no load_in_4bit: both CUDA-only

PASS CRITERIA
    - 10 gradient steps complete without exception
    - reward_mean is a finite float (0.0 acceptable — model is untrained)
    - loss is finite
"""

from __future__ import annotations

import math
import os
import sys
import time

# Must be set before importing torch — causes MPS to fall back to CPU for
# unsupported Metal ops (e.g. some GRPOTrainer matmul variants).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Suppress tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
except ModuleNotFoundError as exc:
    name = getattr(exc, "name", None) or str(exc)
    print(
        "\nGRPO smoke test requires optional packages (torch, datasets, trl, …).\n"
        f"Missing: {name}\n\n"
        "Install with:\n"
        "  uv sync --extra grpo\n\n"
        "Then re-run this script.\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from budget_router.reward import grade_episode
from train.grpo_env import BudgetRouterGRPOEnv

# ── Constants ────────────────────────────────────────────────────────────────

# Smallest Qwen2.5 with validated function-calling support.
# Smoke test only — use Qwen2.5-1.5B-Instruct for real training.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a budget-aware API router. "
    "Use the available tools to route each request to the best provider. "
    "Adapt when providers degrade — switch away from failing providers early."
)

# ── Reward function ──────────────────────────────────────────────────────────

def reward_func(environments, **kwargs):
    """
    TRL reads env instances after each rollout. Returns List[float] in [0, 1].
    grade_episode() is the calibrated grader used by the eval pipeline — keeps
    training and eval metrics consistent.
    """
    rewards = []
    for env in environments:
        history = env._env._internal.history
        if not history:
            # Model made no tool calls — assign 0, not an error
            rewards.append(0.0)
        else:
            rewards.append(float(grade_episode(history)["overall_score"]))
    return rewards

# ── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset(n: int = 32) -> Dataset:
    """
    Minimal dataset. Columns become **kwargs in BudgetRouterGRPOEnv.reset().
    'prompt' is required by GRPOTrainer (messages format).
    'scenario' and 'seed' are passed to reset() for episode configuration.
    """
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Route the incoming requests optimally."},
            ],
            "scenario": "hard_multi",
            "seed": i,
        }
        for i in range(n)
    ])

# ── Step logger ──────────────────────────────────────────────────────────────

class SmokeTestCallback(TrainerCallback):
    """Captures per-step metrics for PASS/FAIL evaluation."""

    def __init__(self):
        self.steps: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step == 0:
            return
        # TRL 1.x logs reward under "reward" or "train/reward"
        reward_mean = logs.get("reward", logs.get("train/reward", float("nan")))
        reward_std  = logs.get("reward_std", logs.get("train/reward_std", float("nan")))
        loss        = logs.get("loss", logs.get("train/loss", float("nan")))
        entry = {
            "step": state.global_step,
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "loss": float(loss),
        }
        self.steps.append(entry)
        print(
            f"  Step {entry['step']:02d}/10 | "
            f"loss={entry['loss']:.4f} | "
            f"reward_mean={entry['reward_mean']:.4f} | "
            f"reward_std={entry['reward_std']:.4f}"
        )

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("=" * 62)
    print("GRPO Smoke Test — Budget Router")
    print("=" * 62)
    print(f"Device   : {device.upper()}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Steps    : 10  (num_generations=4 → 40 rollouts total)")
    print(f"Torch    : {torch.__version__}")
    if device == "cpu":
        print("⚠️  WARNING: Running on CPU. Expect ~30-60 min for 10 steps.")
    print("=" * 62)

    # Load model — explicit dtype for MPS (bfloat16 supported on M-series)
    print("\nLoading model (may download on first run)...")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA: small rank for smoke test — keeps memory and step time low
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPOConfig — hyperparams per TRL/OpenEnv Wordle example + DeepSeek-R1
    # Source: https://huggingface.co/docs/trl/openenv (Wordle section)
    #         DeepSeek-R1 paper: lr=1e-6, temp=1.0, beta=0.001
    args = GRPOConfig(
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,              # min for GRPO; use 8 for real runs
        generation_batch_size=4,      # TRL 1.x: must be divisible by num_generations (see learn_experiment.py)
        max_completion_length=512,    # ~10 multi-turn tool-call turns
        temperature=1.0,                # diverse exploration (DeepSeek-R1)
        beta=0.001,                     # KL penalty; small for verifiable tasks
        learning_rate=5e-7,             # conservative; real training: 1e-6
        optim="adamw_torch",            # paged_adamw_8bit is CUDA-only
        report_to="none",               # no WandB prompt
        logging_steps=1,                # log every step for smoke visibility
        remove_unused_columns=False,    # CRITICAL: keeps scenario/seed cols for reset()
        dataloader_num_workers=0,       # avoid MPS multiprocessing issues
        output_dir="/tmp/grpo_smoke",
    )

    dataset = build_dataset(n=32)
    logger = SmokeTestCallback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
        environment_factory=BudgetRouterGRPOEnv,
        callbacks=[logger],
    )

    print("\nStarting training loop...\n")
    try:
        trainer.train()
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n❌ Training loop raised {type(exc).__name__} after {elapsed:.0f}s:")
        print(f"   {exc}")
        print("\n=== SMOKE TEST: FAIL ===")
        sys.exit(1)

    elapsed = time.time() - t0

    # Evaluate
    if not logger.steps:
        print("\n❌ No steps were logged — trainer may have exited early.")
        print("=== SMOKE TEST: FAIL ===")
        sys.exit(1)

    last = logger.steps[-1]
    reward_mean = last["reward_mean"]
    reward_std  = last["reward_std"]
    loss        = last["loss"]

    passed = (
        len(logger.steps) >= 10
        and not math.isnan(reward_mean)
        and not math.isnan(loss)
        and not math.isinf(loss)
    )

    print("\n" + "=" * 62)
    print("SMOKE TEST RESULT")
    print("=" * 62)
    print(f"Steps completed : {len(logger.steps)}/10")
    print(f"reward_mean     : {reward_mean:.4f}")
    print(f"reward_std      : {reward_std:.4f}")
    print(f"loss            : {loss:.4f}")
    print(f"elapsed         : {elapsed:.0f}s")
    print()
    if passed:
        print("✅ PASS — Loop is functional. Scale up with Qwen2.5-1.5B + num_generations=8.")
    else:
        print("❌ FAIL — Fix issues above before full training run.")
    print("=" * 62)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
