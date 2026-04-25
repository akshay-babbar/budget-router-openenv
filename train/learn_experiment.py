"""
GRPO Learning Experiment — Budget Router (~90 min on M4 Mac MPS)

GOAL
    Determine (80-90% confidence) whether BudgetRouterGRPOEnv provides a
    learnable signal for GRPO on Qwen/Qwen3-0.6B.

TIMING (from smoke test)
    Outer iteration = 1 rollout step (~50s) + 3 gradient steps (~6s) = ~56s
    max_steps=360 → ~90 outer iterations → ~84 min on M4 48GB MPS

VERDICT LOGIC
    Compare mean reward of first 25% vs last 25% of rollout steps.
    - LEARNING DETECTED   : last_quarter > first_quarter + 0.05
    - NOT LEARNING        : last_quarter < first_quarter - 0.02
    - INCONCLUSIVE        : otherwise (high variance, too few tool calls)

USAGE
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train/learn_experiment.py

KEY DIFFERENCES FROM smoke_test.py
    - max_steps=360 (vs 10)
    - learning_rate=5e-6 (vs 5e-7 — standard GRPO lr per DeepSeek-R1 paper)
    - Proper rollout-only callback: separates data-collection from gradient steps
    - Rolling reward average printed every 10 rollout steps
    - VERDICT analysis at the end
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from collections import Counter

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from budget_router.reward import grade_episode
from train.grpo_env import BudgetRouterGRPOEnv

# ── Config ───────────────────────────────────────────────────────────────────

# NOTE: model is configurable via --model-name (defaults to a stronger tool caller).
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
PROMPT_STYLES = ("strict", "explore")

STRICT_SYSTEM_PROMPT = (
    "You are a budget-aware API router. "
    "Use the available tools to route each request to the best provider. "
    "Providers can degrade mid-episode — monitor health and switch early.\n\n"
    "TOOL CALL FORMAT — you must use this EXACT format to call a tool:\n"
    "<tool_call>\n{\"name\": \"route_to_a\", \"arguments\": {}}\n</tool_call>\n\n"
    "EXAMPLES:\n"
    "  Observation: A=0.833, B=0.500, C=0.500, budget=95%\n"
    "  Action: <tool_call>\n{\"name\": \"route_to_a\", \"arguments\": {}}\n</tool_call>\n\n"
    "  Observation: A=0.100 (degraded!), B=0.833, C=0.500, budget=70%\n"
    "  Action: <tool_call>\n{\"name\": \"route_to_b\", \"arguments\": {}}\n</tool_call>\n\n"
    "Route to the cheapest healthy provider (A=$0.01 < B=$0.05 < C=$0.10). "
    "Switch early if a provider's health drops below 0.52."
)

EXPLORE_SYSTEM_PROMPT = (
    "You are a budget-aware API router. "
    "Use exactly one available tool after each observation to route the current request.\n\n"
    "TOOL CALL FORMAT — respond with only this XML-wrapped JSON shape:\n"
    "<tool_call>\n{\"name\": \"route_to_a\", \"arguments\": {}}\n</tool_call>\n\n"
    "Available tools: route_to_a, route_to_b, route_to_c, shed_load.\n"
    "Objective: maximize successful routed requests while preserving budget and avoiding "
    "unhealthy or overloaded providers. Providers can degrade mid-episode, so use the "
    "latest observed health, latency, queue, budget, and steps-left information.\n"
    "Do not follow a fixed provider cycle. Change actions only when the observations "
    "make a different provider or shedding load look better."
)


def build_system_prompt(prompt_style: str = "strict") -> str:
    if prompt_style == "strict":
        return STRICT_SYSTEM_PROMPT
    if prompt_style == "explore":
        return EXPLORE_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_style={prompt_style!r}; expected one of {PROMPT_STYLES}")

# ── Reward function ──────────────────────────────────────────────────────────

LAST_ROLLOUT_DIAGNOSTICS: dict[str, object] = {}


def episode_training_reward(env: BudgetRouterGRPOEnv) -> float:
    internal = env._env._internal
    history = internal.history
    if not history:
        return 0.0

    grader = float(grade_episode(history)["overall_score"])
    if internal.episode_done:
        return grader

    progress = internal.current_step / max(1, internal.max_steps)
    return grader * progress


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _action_sequence(history) -> str:
    actions = [str(step.get("action_type", "unknown")) for step in history]
    return " ".join(actions) if actions else "<empty>"


def summarize_training_rollout(environments) -> dict[str, object]:
    env_steps = []
    progress = []
    raw_graders = []
    training_rewards = []
    completions = []
    budget_exhaustions = []
    action_sequences = []

    for env in environments:
        internal = env._env._internal
        history = internal.history
        env_steps.append(float(internal.current_step))
        progress.append(float(internal.current_step / max(1, internal.max_steps)))
        raw_graders.append(float(grade_episode(history)["overall_score"]) if history else 0.0)
        training_rewards.append(episode_training_reward(env))
        completions.append(1.0 if internal.episode_done else 0.0)
        budget_exhaustions.append(
            1.0 if any(step.get("budget_exhausted", False) for step in history) else 0.0
        )
        action_sequences.append(_action_sequence(history))

    sequence_counts = dict(Counter(action_sequences))

    return {
        "env_steps_mean": _mean(env_steps),
        "env_steps_min": min(env_steps) if env_steps else 0.0,
        "env_steps_max": max(env_steps) if env_steps else 0.0,
        "progress_mean": _mean(progress),
        "raw_grader_mean": _mean(raw_graders),
        "training_reward_mean": _mean(training_rewards),
        "training_rewards": training_rewards,
        "episode_completion_rate": _mean(completions),
        "budget_exhaustion_rate": _mean(budget_exhaustions),
        "action_sequences": action_sequences,
        "unique_action_sequences": len(sequence_counts),
        "action_sequence_counts": sequence_counts,
    }


def reward_func(environments, **kwargs):
    global LAST_ROLLOUT_DIAGNOSTICS
    LAST_ROLLOUT_DIAGNOSTICS = summarize_training_rollout(environments)
    return [episode_training_reward(env) for env in environments]

# ── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset(n: int = 200, prompt_style: str = "strict") -> Dataset:
    system_prompt = build_system_prompt(prompt_style)
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Route the incoming requests optimally."},
            ],
            "scenario": "hard_multi",
            "seed": i,
        }
        for i in range(n)
    ])

# ── Callback ─────────────────────────────────────────────────────────────────

class LearnCallback(TrainerCallback):
    """
    Tracks rollout steps (data-collection) separately from gradient-only steps.
    GRPO pattern: 1 rollout step + 3 gradient-only steps = 1 outer iteration.
    Only rollout steps carry reward/tool_call metrics.
    """

    ROLLOUT_PRINT_EVERY = 10  # print rolling average every N rollout steps

    def __init__(self):
        self.rollout_rewards: list[float] = []
        self.total_grad_steps: int = 0
        self.tool_call_freqs: list[float] = []
        self.env_step_means: list[float] = []
        self.completion_rates: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step == 0:
            return
        if "train_runtime" in logs:
            return

        step = state.global_step
        self.total_grad_steps = max(self.total_grad_steps, step)
        loss = float(logs.get("loss", float("nan")))

        # Gradient-only steps have no reward key
        if "reward" not in logs:
            print(f"  [{step:03d}] grad-only | loss={loss:.4f}")
            return

        reward = float(logs.get("reward", float("nan")))
        reward_std = float(logs.get("reward_std", float("nan")))
        tool_freq = float(logs.get("tools/call_frequency", float("nan")))
        diagnostics = LAST_ROLLOUT_DIAGNOSTICS

        if not math.isnan(reward):
            self.rollout_rewards.append(reward)
        if not math.isnan(tool_freq):
            self.tool_call_freqs.append(tool_freq)
        if diagnostics:
            self.env_step_means.append(float(diagnostics["env_steps_mean"]))
            self.completion_rates.append(float(diagnostics["episode_completion_rate"]))

        rollout_n = len(self.rollout_rewards)
        rolling_avg = (
            sum(self.rollout_rewards[-10:]) / len(self.rollout_rewards[-10:])
            if self.rollout_rewards else float("nan")
        )
        sequence_counts = diagnostics.get("action_sequence_counts", {}) if diagnostics else {}
        unique_sequences = int(diagnostics.get("unique_action_sequences", 0)) if diagnostics else 0
        group_size = len(diagnostics.get("action_sequences", [])) if diagnostics else 0
        reward_values = diagnostics.get("training_rewards", []) if diagnostics else []
        reward_preview = ",".join(f"{float(v):.4f}" for v in reward_values)

        print(
            f"  [{step:03d}] ROLLOUT #{rollout_n:02d} | "
            f"reward={reward:.4f} | std={reward_std:.4f} | "
            f"tool_freq={tool_freq:.2f} | "
            f"env_steps={diagnostics.get('env_steps_mean', float('nan')):.1f} | "
            f"complete={diagnostics.get('episode_completion_rate', float('nan')):.2f} | "
            f"raw={diagnostics.get('raw_grader_mean', float('nan')):.4f} | "
            f"seqs={unique_sequences}/{group_size} | "
            f"rewards=[{reward_preview}] | "
            f"rolling10={rolling_avg:.4f} | loss={loss:.4f}"
        )
        if unique_sequences <= 3 and sequence_counts:
            counts = " || ".join(
                f"{count}x {sequence}" for sequence, count in sequence_counts.items()
            )
            print(f"       action_sequences: {counts}")

        if rollout_n % self.ROLLOUT_PRINT_EVERY == 0:
            self._print_trend_summary()

    def _print_trend_summary(self):
        n = len(self.rollout_rewards)
        if n < 4:
            return
        q = max(1, n // 4)
        first_q = sum(self.rollout_rewards[:q]) / q
        last_q  = sum(self.rollout_rewards[-q:]) / q
        avg_tool = (
            sum(self.tool_call_freqs) / len(self.tool_call_freqs)
            if self.tool_call_freqs else float("nan")
        )
        avg_env_steps = (
            sum(self.env_step_means) / len(self.env_step_means)
            if self.env_step_means else float("nan")
        )
        avg_completion = (
            sum(self.completion_rates) / len(self.completion_rates)
            if self.completion_rates else float("nan")
        )
        print(
            f"\n  ── Trend @ rollout {n} ──\n"
            f"     first-quarter mean : {first_q:.4f}\n"
            f"     last-quarter  mean : {last_q:.4f}\n"
            f"     delta               : {last_q - first_q:+.4f}\n"
            f"     avg tool_call_freq  : {avg_tool:.3f}\n"
            f"     avg env_steps       : {avg_env_steps:.2f}\n"
            f"     avg completion_rate : {avg_completion:.3f}\n"
        )

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    parser = argparse.ArgumentParser(description="GRPO Learning Experiment — Budget Router")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HF model id to train (default: Qwen/Qwen3-1.7B).",
    )
    parser.add_argument("--max-steps", type=int, default=360, help="Total GRPO max_steps (outer iterations).")
    parser.add_argument("--dataset-n", type=int, default=200, help="Number of episodes in training dataset.")
    parser.add_argument("--save-steps", type=int, default=60, help="Checkpoint save frequency in steps.")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO generations per prompt.")
    parser.add_argument("--max-completion-length", type=int, default=1024, help="Completion token budget across tool loop.")
    parser.add_argument("--max-tool-calling-iterations", type=int, default=20, help="Maximum tool loop iterations per rollout.")
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="strict",
        help="Prompt style: strict preserves the original heuristic prompt; explore reduces deterministic policy bias.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for GRPO rollouts.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p for GRPO rollouts.")
    cli = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("=" * 68)
    print("GRPO Learning Experiment — Budget Router")
    print("=" * 68)
    print(f"Device   : {device.upper()}")
    print(f"Model    : {cli.model_name}")
    print(f"Steps    : {cli.max_steps}")
    print(f"Prompt   : {cli.prompt_style}")
    print(f"Sampling : temperature={cli.temperature} top_p={cli.top_p}")
    print(f"Torch    : {torch.__version__}")
    print("=" * 68)

    print("\nLoading model...")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cli.model_name, dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cli.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Hyperparams: TRL/OpenEnv Wordle example + DeepSeek-R1 paper
    # lr=1e-6: standard GRPO (smoke test used 5e-7, too conservative for 360 steps)
    # enable_thinking=False: reclaims ~400 tokens/step from Qwen3 reasoning blocks,
    #   allowing max_completion_length to drop from 512→256 without clipping valid tool calls.
    args = GRPOConfig(
        max_steps=cli.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=cli.num_generations,
        generation_batch_size=cli.num_generations,
        max_completion_length=cli.max_completion_length,
        max_tool_calling_iterations=cli.max_tool_calling_iterations,
        temperature=cli.temperature,
        top_p=cli.top_p,
        beta=0.001,
        learning_rate=5e-6,
        optim="adamw_torch",
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_steps=cli.save_steps,
        output_dir="trained_models/grpo_checkpoints",
        chat_template_kwargs={"enable_thinking": False},  # Qwen3: skip <think> blocks
    )

    dataset = build_dataset(n=cli.dataset_n, prompt_style=cli.prompt_style)
    cb = LearnCallback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
        environment_factory=BudgetRouterGRPOEnv,
        callbacks=[cb],
    )

    def _save_merged(label: str) -> None:
        """
        Merge LoRA weights into the base model and save to disk.
        The merged model is a plain HuggingFace model — loadable with
        AutoModelForCausalLM.from_pretrained() without any PEFT dependency.
        """
        safe_name = (
            cli.model_name.replace("/", "_")
            .replace(":", "_")
            .replace("@", "_")
        )
        save_path = f"trained_models/grpo_{safe_name}"
        print(f"\n[{label}] Merging LoRA into base model and saving to {save_path} ...")
        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"[{label}] ✅ Saved. Load with: AutoModelForCausalLM.from_pretrained('{save_path}')")
        except Exception as e:
            print(f"[{label}] ⚠️  Save failed: {e}")

    print("\nStarting experiment (Ctrl+C to stop early — partial results still printed)...\n")
    try:
        trainer.train()
        _save_merged("END")
    except KeyboardInterrupt:
        print("\n[Interrupted by user — computing verdict on partial results...]")
        _save_merged("INTERRUPT")
    except Exception as exc:
        print(f"\n❌ Training error: {type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = time.time() - t0

    # ── VERDICT ──────────────────────────────────────────────────────────────
    rewards = cb.rollout_rewards
    tool_freqs = cb.tool_call_freqs
    env_step_means = cb.env_step_means
    completion_rates = cb.completion_rates

    print("\n" + "=" * 68)
    print("LEARNING EXPERIMENT — VERDICT")
    print("=" * 68)
    print(f"Grad steps completed : {cb.total_grad_steps}/{cli.max_steps}")
    print(f"Rollout steps        : {len(rewards)}")
    print(f"Elapsed              : {elapsed/60:.1f} min")

    if len(rewards) < 4:
        print("\n⚠️  Too few rollout steps for a verdict. Run longer.")
        sys.exit(0)

    q = max(1, len(rewards) // 4)
    first_q_mean = sum(rewards[:q]) / q
    last_q_mean  = sum(rewards[-q:]) / q
    delta = last_q_mean - first_q_mean
    avg_tool_freq = sum(tool_freqs) / len(tool_freqs) if tool_freqs else 0.0
    avg_env_steps = sum(env_step_means) / len(env_step_means) if env_step_means else 0.0
    avg_completion_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
    overall_mean = sum(rewards) / len(rewards)

    print(f"\nReward summary:")
    print(f"  First-quarter mean : {first_q_mean:.4f}")
    print(f"  Last-quarter  mean : {last_q_mean:.4f}")
    print(f"  Delta (improvement): {delta:+.4f}")
    print(f"  Overall mean       : {overall_mean:.4f}")
    print(f"  Avg tool_call_freq : {avg_tool_freq:.3f}")
    print(f"  Avg env_steps      : {avg_env_steps:.2f}")
    print(f"  Avg completion_rate: {avg_completion_rate:.3f}")

    print(f"\nHeuristic baseline   : ~0.60-0.65 (from environment benchmark)")
    print(f"Random agent         : ~0.15-0.20")

    print()
    if delta > 0.05:
        print("✅ VERDICT: LEARNING SIGNAL DETECTED")
        print("   Reward improved meaningfully across the run.")
        print("   Recommend: scale up to Qwen2.5-1.5B-Instruct + num_generations=8.")
    elif delta < -0.02:
        print("❌ VERDICT: NOT LEARNING")
        print("   Reward trended downward. Possible causes:")
        print("   - lr too high (try 5e-7)")
        print("   - too few tool calls (model not generating tool syntax reliably)")
        print("   - environment reward too sparse")
    else:
        print("⚠️  VERDICT: INCONCLUSIVE")
        print(f"   Delta={delta:+.4f} is within noise range.")
        if avg_tool_freq < 0.3:
            print("   Root cause likely: tool_call_freq too low — model rarely uses tools.")
            print("   Fix: add few-shot tool-call examples to the system prompt.")
        elif avg_env_steps < 10:
            print("   Root cause likely: rollouts are too short for 20-step routing.")
            print("   Fix: inspect completion budget, compact tool responses, and tool-loop limits.")
        else:
            print("   More steps needed. Try 600-800 steps for clearer trend.")

    print("=" * 68)


if __name__ == "__main__":
    main()
