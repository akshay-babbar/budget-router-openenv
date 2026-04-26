"""
grpo_train.py — GRPO training for Budget Router LLM agent.

OBJECTIVE
    Train a Qwen LLM to directly learn the routing policy from environment
    rewards via GRPO. The model interacts with BudgetRouterEnv through tool
    calls, receives grader scores, and updates its policy.

WHY GRPO (not SFT)
    • SFT is bounded by teacher ceiling. GRPO learns from the actual reward.
    • The environment has a deterministic verifier (grade_episode) — textbook
      GRPO/RLVR territory per the hackathon guide.
    • PPO (SB3 MlpPolicy, 100k steps) already proves the environment is
      learnable and scores 0.691. GRPO on an LLM should discover similar or
      better strategies through direct interaction.

ROOT CAUSE OF PREVIOUS FAILURE (learn_experiment.py)
    episode_completion_rate = 0.00 — the model NEVER completed a single
    20-step episode because max_completion_length=1024 was too small.
    Each tool call needs ~130 tokens (call XML + response). 20 calls ×
    130 tokens = 2600 tokens minimum. The model was starved for tokens.

    FIX: max_completion_length=3072, max_tool_calling_iterations=25.

USAGE — LOCAL (smoke test only, ~10 min on Mac MPS)
    uv run python train/grpo_train.py --smoke

USAGE — HF JOBS (a10g-large, recommended)
    uv run python train/launch_grpo_job.py --smoke   # ~10 min, ~$0.15
    uv run python train/launch_grpo_job.py --full    # ~90-120 min, ~$2.00

EVALUATION
    uv run python train/eval_trained.py \\
        --model-path <hub-repo-or-local-path> \\
        --n-episodes 10

DESIGN DECISIONS (world-class RL conventions)
    • max_completion_length=3072: allows full 20-step episodes (critical fix)
    • num_generations=8: better GRPO variance estimate (GRPO gradient ∝ advantage)
    • temperature=1.0: standard GRPO exploration temperature
    • beta=0.04: moderate KL penalty — prevents mode collapse while allowing
      policy improvement (DeepSeek-R1 uses 0.001-0.04 depending on stage)
    • LoRA r=16 on q/k/v/o_proj: sufficient capacity for policy learning
    • reward = grade_episode()["overall_score"] for complete episodes,
      scaled partial credit for incomplete (encourages full-episode completion)
    • SFT warm-start optional via --warm-start-from: teaches action format,
      then GRPO refines the policy from environment rewards.
    • Training seeds [200..N] — disjoint from eval [0..9] and heldout [100..109]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import Counter
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from budget_router.reward import grade_episode
from train.grpo_env import BudgetRouterGRPOEnv

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

# Training seeds start at 200 — disjoint from dev [0..9] and heldout [100..109]
TRAIN_SEED_OFFSET = 200

# ── Reward function ──────────────────────────────────────────────────────────

LAST_DIAGNOSTICS: dict = {}


def reward_func(environments, **kwargs) -> list[float]:
    """Compute rewards from environment episodes.

    Complete episodes get the full grader score (overall_score ∈ [0,1]).
    Incomplete episodes get scaled partial credit to encourage completion.
    """
    global LAST_DIAGNOSTICS
    rewards = []
    diagnostics = {
        "env_steps": [], "completions": [], "graders": [],
        "budget_exhausted": [], "action_seqs": [],
    }

    for env in environments:
        internal = env._env._internal
        history = internal.history
        steps = float(internal.current_step)
        max_steps = float(max(1, internal.max_steps))
        done = internal.episode_done

        if not history:
            rewards.append(0.0)
            diagnostics["env_steps"].append(0.0)
            diagnostics["completions"].append(0.0)
            diagnostics["graders"].append(0.0)
            diagnostics["budget_exhausted"].append(0.0)
            diagnostics["action_seqs"].append("<empty>")
            continue

        grader = float(grade_episode(history)["overall_score"])
        budget_bust = any(h.get("budget_exhausted", False) for h in history)

        if done:
            # Full episode — use raw grader score
            reward = grader
        else:
            # Incomplete — scaled by progress, small penalty for not finishing
            progress = steps / max_steps
            reward = grader * progress * 0.8  # 20% penalty for not completing

        rewards.append(reward)
        diagnostics["env_steps"].append(steps)
        diagnostics["completions"].append(1.0 if done else 0.0)
        diagnostics["graders"].append(grader)
        diagnostics["budget_exhausted"].append(1.0 if budget_bust else 0.0)
        actions = [str(h.get("action_type", "?")) for h in history]
        diagnostics["action_seqs"].append(" ".join(actions) if actions else "<empty>")

    LAST_DIAGNOSTICS = diagnostics
    return rewards


# ── Dataset ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a budget-aware API router. "
    "Use the available tools to route each request to the best provider. "
    "Providers can degrade mid-episode — monitor health and switch early.\n\n"
    "TOOL CALL FORMAT — respond with EXACTLY this XML-wrapped JSON:\n"
    '<tool_call>\n{"name": "route_to_b", "arguments": {}}\n</tool_call>\n\n'
    "Available tools: route_to_a, route_to_b, route_to_c, shed_load.\n"
    "Route to the cheapest healthy provider. "
    "Switch early if a provider's health drops."
)


def build_dataset(n: int, seed_offset: int = TRAIN_SEED_OFFSET) -> Dataset:
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Route the incoming requests optimally."},
            ],
            "scenario": "hard_multi",
            "seed": seed_offset + i,
        }
        for i in range(n)
    ])


# ── Callback ─────────────────────────────────────────────────────────────────

class GRPOCallback(TrainerCallback):
    """Tracks rollout diagnostics and prints training progress."""

    def __init__(self):
        self.rollout_rewards: list[float] = []
        self.rollout_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step == 0:
            return
        if "train_runtime" in logs:
            return

        step = state.global_step
        loss = float(logs.get("loss", float("nan")))

        if "reward" not in logs:
            # Gradient-only step
            return

        reward = float(logs.get("reward", 0.0))
        reward_std = float(logs.get("reward_std", 0.0))
        tool_freq = float(logs.get("tools/call_frequency", 0.0))
        self.rollout_count += 1
        self.rollout_rewards.append(reward)

        diag = LAST_DIAGNOSTICS
        env_steps = diag.get("env_steps", [])
        completions = diag.get("completions", [])
        graders = diag.get("graders", [])
        action_seqs = diag.get("action_seqs", [])

        avg_steps = sum(env_steps) / len(env_steps) if env_steps else 0
        comp_rate = sum(completions) / len(completions) if completions else 0
        avg_grader = sum(graders) / len(graders) if graders else 0
        unique_seqs = len(set(action_seqs))
        total_seqs = len(action_seqs)

        # Rolling average
        recent = self.rollout_rewards[-10:]
        rolling = sum(recent) / len(recent)

        print(
            f"  [{step:04d}] ROLLOUT #{self.rollout_count:03d} | "
            f"reward={reward:.4f}±{reward_std:.3f} | "
            f"grader={avg_grader:.4f} | "
            f"tools={tool_freq:.1f} | "
            f"steps={avg_steps:.1f}/20 | "
            f"done={comp_rate:.0%} | "
            f"seqs={unique_seqs}/{total_seqs} | "
            f"rolling10={rolling:.4f} | "
            f"loss={loss:.4f}"
        )

        # Print action sequences if collapsed (danger sign)
        if unique_seqs <= 2 and action_seqs:
            counts = Counter(action_seqs)
            top = counts.most_common(2)
            for seq, cnt in top:
                print(f"       {cnt}x: {seq[:80]}...")

        # Trend every 20 rollouts
        if self.rollout_count % 20 == 0 and len(self.rollout_rewards) >= 8:
            q = max(1, len(self.rollout_rewards) // 4)
            first_q = sum(self.rollout_rewards[:q]) / q
            last_q = sum(self.rollout_rewards[-q:]) / q
            print(
                f"\n  ── Trend @ rollout {self.rollout_count} ──\n"
                f"     first-quarter: {first_q:.4f}\n"
                f"     last-quarter:  {last_q:.4f}\n"
                f"     delta:         {last_q - first_q:+.4f}\n"
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    parser = argparse.ArgumentParser(description="GRPO training for Budget Router LLM agent.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model to train.")
    parser.add_argument(
        "--warm-start-from", default=None,
        help="Path to SFT-trained model for warm start (teaches action format).",
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (~10 min on A10G).")
    parser.add_argument("--full", action="store_true", help="Full training (~90-120 min on A10G).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo id to push model to.")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--dataset-n", type=int, default=None, help="Override dataset size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    cli = parser.parse_args()

    if cli.smoke and cli.full:
        print("❌ Pick --smoke or --full, not both.")
        sys.exit(2)
    if not cli.smoke and not cli.full:
        cli.full = True

    # ── Hyperparameters ──────────────────────────────────────────────────
    if cli.smoke:
        dataset_n = cli.dataset_n or 50
        max_steps = cli.max_steps or 40     # ~10 rollouts (1 rollout + 3 grad = 4 steps)
        save_steps = 20
        suffix = "_smoke"
    else:
        dataset_n = cli.dataset_n or 400
        max_steps = cli.max_steps or 600    # ~150 rollouts
        save_steps = 100
        suffix = ""

    model_name = cli.warm_start_from or cli.model_name
    safe_name = cli.model_name.replace("/", "_").replace(":", "_")
    output_dir = cli.output_dir or f"trained_models/grpo_{safe_name}{suffix}"

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    print("=" * 70)
    print("GRPO Training — Budget Router LLM Agent")
    print("=" * 70)
    print(f"Mode      : {'SMOKE' if cli.smoke else 'FULL'}")
    print(f"Model     : {model_name}")
    print(f"Warm start: {cli.warm_start_from or 'None (cold start)'}")
    print(f"Device    : {device} | Dtype: {dtype}")
    print(f"Dataset   : {dataset_n} episodes | Max steps: {max_steps}")
    print(f"Output    : {output_dir}")
    print("=" * 70)

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA ─────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPO Config ──────────────────────────────────────────────────────
    # CRITICAL: max_completion_length=3072 to allow full 20-step episodes
    # Previous failure: 1024 tokens → only ~8 tool calls → 0% completion
    grpo_config = GRPOConfig(
        output_dir=output_dir + "_checkpoints",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,                    # Better GRPO variance estimate
        generation_batch_size=8,              # Generate all at once
        max_completion_length=3072,           # CRITICAL FIX: was 1024
        max_tool_calling_iterations=25,       # Buffer above 20 steps
        temperature=1.0,                      # Standard GRPO exploration
        top_p=1.0,
        beta=0.04,                            # Moderate KL penalty
        learning_rate=5e-6,                   # Standard GRPO lr (DeepSeek-R1)
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        report_to="none",
        logging_steps=1,
        save_steps=save_steps,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        chat_template_kwargs={"enable_thinking": False},  # Qwen3: no <think> blocks
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = build_dataset(n=dataset_n)
    print(f"Dataset: {len(dataset)} episodes (seeds [{TRAIN_SEED_OFFSET}..{TRAIN_SEED_OFFSET + dataset_n - 1}])")

    # ── Trainer ──────────────────────────────────────────────────────────
    callback = GRPOCallback()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        environment_factory=BudgetRouterGRPOEnv,
        callbacks=[callback],
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nStarting GRPO training ({max_steps} steps)...\n")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    elapsed = time.time() - t0

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\nMerging LoRA and saving to {output_dir}...")
    try:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Saved to {output_dir}")
    except Exception as e:
        print(f"⚠️  Save failed: {e}")
        merged = None

    # ── Push to Hub ──────────────────────────────────────────────────────
    if cli.push_to_hub and merged is not None:
        print(f"\nPushing to https://huggingface.co/{cli.push_to_hub} ...")
        merged.push_to_hub(cli.push_to_hub, private=cli.hub_private)
        tokenizer.push_to_hub(cli.push_to_hub, private=cli.hub_private)
        print(f"✅ Pushed to {cli.push_to_hub}")

    # ── Verdict ──────────────────────────────────────────────────────────
    rewards = callback.rollout_rewards
    print(f"\n{'=' * 70}")
    print("GRPO TRAINING — VERDICT")
    print(f"{'=' * 70}")
    print(f"Rollouts completed : {len(rewards)}")
    print(f"Elapsed            : {elapsed / 60:.1f} min")

    if len(rewards) >= 8:
        q = max(1, len(rewards) // 4)
        first_q = sum(rewards[:q]) / q
        last_q = sum(rewards[-q:]) / q
        delta = last_q - first_q
        overall = sum(rewards) / len(rewards)
        print(f"First-quarter mean : {first_q:.4f}")
        print(f"Last-quarter mean  : {last_q:.4f}")
        print(f"Delta              : {delta:+.4f}")
        print(f"Overall mean       : {overall:.4f}")
        print(f"Heuristic baseline : ~0.608")
        print()
        if delta > 0.05:
            print("✅ LEARNING SIGNAL DETECTED — reward improving across training.")
        elif delta < -0.02:
            print("❌ NOT LEARNING — reward trending down.")
        else:
            print("⚠️  INCONCLUSIVE — need more steps or reward variance is high.")
    else:
        print("⚠️  Too few rollouts for verdict.")

    if merged is not None:
        print(f"\nEvaluate with:")
        src = cli.push_to_hub or output_dir
        print(f"  uv run python train/eval_trained.py --model-path {src} --n-episodes 10")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
