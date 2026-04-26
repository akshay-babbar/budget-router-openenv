"""
sft_ppo_distill.py — Distill the trained PPO agent into a Qwen LLM via SFT.

OBJECTIVE
    Train a language model to replicate the PPO agent's routing strategy.
    The PPO agent (MlpPolicy, SB3, 100k steps) scores 0.691 mean grader on
    hard_multi — the highest teacher available.  By distilling its behaviour
    into a chat-format LLM, we produce a deployable language model that
    inherits the PPO's anticipatory routing strategy.

WHY PPO-DISTILLATION (not heuristic-distillation)
    • PPO scores 0.691 vs heuristic 0.608 — a 13.6% advantage.
    • PPO's strategy is simple and consistent (B×10→C/shed interleaving→shed)
      which makes it highly learnable by SFT.
    • PPO learned genuine temporal planning: conserve budget before B's cascade
      at step 10, then interleave C and shed_load to manage remaining budget.
    • The heuristic's reactive A→B→C pattern caps SFT ceiling at ~0.608.
    • "RL distillation" (RL agent → SFT → LLM) is standard practice
      (used by DeepSeek, OpenAI, Anthropic).

SEED ISOLATION
    • Training seeds: [200..N+199] — disjoint from dev [0..9] and heldout [100..109]
    • Smoke eval: seeds [0, 1, 2] (dev set)
    • Full eval: use train/eval_trained.py on seeds [0..9] and [100..109]
    • Zero seed leakage between train and eval.

USAGE — LOCAL
    uv sync --extra grpo
    uv run python train/sft_ppo_distill.py --smoke

USAGE — HF JOBS (recommended, a10g-large)
    uv run python train/launch_sft_ppo_job.py --smoke   # ~5 min, ~$0.10
    uv run python train/launch_sft_ppo_job.py --full    # ~30–45 min, ~$0.90

EVALUATION
    uv run python train/eval_trained.py \\
        --model-path <hub-repo-or-local-path> \\
        --n-episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from stable_baselines3 import PPO

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI
from train.gym_wrapper import BudgetRouterGymEnv

from train.eval_trained import (
    SYSTEM_PROMPT,
    _apply_chat_template,
    _obs_to_text,
    run_episode_heuristic,
    run_episode_llm,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_PPO_PATH = "trained_models/ppo_hard_multi_100k.zip"
TRAIN_SEED_OFFSET = 200  # disjoint from dev [0..9] and heldout [100..109]

_ACTION_MAP = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]


def collect_ppo_trajectories(
    ppo_model: PPO,
    seeds: list[int],
) -> list[dict]:
    """Roll out the PPO agent on hard_multi for each seed.

    Returns chat-format conversations (system + alternating user/assistant).
    Each assistant message is the PPO agent's chosen action rendered as text.
    """
    examples: list[dict] = []
    for seed in seeds:
        # Run PPO through the Gym wrapper to get actions
        gym_env = BudgetRouterGymEnv(scenario=HARD_MULTI, seed=seed)
        gym_obs, _ = gym_env.reset()

        # Also run the OpenEnv environment in parallel to get observations
        env = BudgetRouterEnv()
        obs = env.reset(scenario=HARD_MULTI, seed=seed)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        while not obs.done:
            # Get PPO's action
            action_idx, _ = ppo_model.predict(gym_obs, deterministic=True)
            action_str = _ACTION_MAP[int(action_idx)]

            # Build chat messages from observation
            messages.append({"role": "user", "content": _obs_to_text(obs)})
            messages.append({"role": "assistant", "content": action_str})

            # Step both environments
            action = Action(action_type=ActionType(action_str))
            obs = env.step(action)
            gym_obs, _, terminated, truncated, _ = gym_env.step(int(action_idx))
            done = terminated or truncated

        history = env._internal.history
        if any(h.get("budget_exhausted") for h in history):
            continue
        grader = float(grade_episode(history)["overall_score"])
        examples.append({"messages": messages, "grader": grader})

    return examples


def filter_top_half(examples: list[dict]) -> list[dict]:
    """Best-of-N filtered behavior cloning: keep only the top-grader half."""
    if len(examples) <= 4:
        return examples
    examples_sorted = sorted(examples, key=lambda e: e["grader"], reverse=True)
    return examples_sorted[: len(examples_sorted) // 2]


def build_action_only_examples(
    episodes: list[dict],
    tokenizer,
    max_length: int,
) -> list[dict]:
    """Convert filtered episodes into per-decision causal-LM examples.

    Input tokens contain system prompt + history + current observation. Labels
    are -100 for all prompt/history tokens and real ids only for the target
    action string. This trains exactly the thing eval needs: next action.
    """
    eos = tokenizer.eos_token or ""
    rows: list[dict] = []
    for episode in episodes:
        messages = episode["messages"]
        context = [messages[0]]
        for idx in range(1, len(messages), 2):
            if idx + 1 >= len(messages):
                break
            user_msg = messages[idx]
            assistant_msg = messages[idx + 1]
            prompt_messages = context + [user_msg]
            prompt = _apply_chat_template(
                tokenizer, prompt_messages, add_generation_prompt=True
            )
            action_text = assistant_msg["content"].strip()
            target_text = action_text + eos
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=False
            )["input_ids"]
            target_ids = tokenizer(
                target_text, add_special_tokens=False, truncation=False
            )["input_ids"]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if len(input_ids) > max_length:
                overflow = len(input_ids) - max_length
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]
            if all(label == -100 for label in labels):
                continue
            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "action": action_text,
                }
            )
            context.extend([user_msg, assistant_msg])
    return rows


@dataclass
class ActionOnlyCollator:
    tokenizer: object

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT-distill the trained PPO agent into a chat LLM."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ppo-path", default=DEFAULT_PPO_PATH)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (~200 episodes, max 80 steps, in-script eval). ~5-8 min on A10G.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run (~1000 episodes, 3 epochs). ~30-45 min on A10G.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="HF Hub repo id to push the merged model to.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the pushed Hub repo private.",
    )
    args = parser.parse_args()

    if args.smoke and args.full:
        print("❌ Pick exactly one of --smoke or --full.")
        sys.exit(2)
    if not args.smoke and not args.full:
        args.full = True

    if args.smoke:
        n_episodes = 200
        n_epochs = 1
        max_steps = 80
        smoke_eval_seeds = [0, 1, 2]
        suffix = "_smoke"
    else:
        n_episodes = 1000
        n_epochs = 3
        max_steps = -1
        smoke_eval_seeds = []
        suffix = ""

    safe_name = args.model_name.replace("/", "_").replace(":", "_")
    output_dir = args.output_dir or f"trained_models/sft_ppo_{safe_name}{suffix}"

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    print("=" * 70)
    print(f"PPO → SFT Distillation — {args.model_name}")
    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'} | Device: {device} | Dtype: {dtype}")
    print(f"PPO model: {args.ppo_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # ── Load PPO teacher ─────────────────────────────────────────────────
    print(f"\n[0/4] Loading PPO teacher from {args.ppo_path}...")
    ppo_model = PPO.load(args.ppo_path)
    print("      PPO model loaded.")

    # ── Collect PPO trajectories ─────────────────────────────────────────
    train_seeds = list(range(TRAIN_SEED_OFFSET, TRAIN_SEED_OFFSET + n_episodes))
    print(
        f"\n[1/4] Collecting {n_episodes} PPO trajectories on seeds "
        f"[{train_seeds[0]}..{train_seeds[-1]}] (disjoint from eval seeds)..."
    )
    raw_examples = collect_ppo_trajectories(ppo_model, train_seeds)
    if not raw_examples:
        print("❌ No completed PPO episodes — model or env regression?")
        sys.exit(1)

    teacher_mean = sum(e["grader"] for e in raw_examples) / len(raw_examples)
    examples = filter_top_half(raw_examples)
    kept_mean = sum(e["grader"] for e in examples) / len(examples)
    print(
        f"      Collected {len(raw_examples)} completed → kept top {len(examples)}; "
        f"teacher_mean={teacher_mean:.4f} kept_mean={kept_mean:.4f}"
    )

    # ── Load LLM and build training data ────────────────────────────────
    print(f"\n[2/4] Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = 1024
    action_rows = build_action_only_examples(examples, tokenizer, max_length=max_length)
    if not action_rows:
        print("❌ No action-only training examples produced.")
        sys.exit(1)
    action_counts = {
        action: sum(1 for row in action_rows if row["action"] == action)
        for action in sorted({row["action"] for row in action_rows})
    }
    supervised_tokens = sum(
        sum(1 for label in row["labels"] if label != -100) for row in action_rows
    )
    print(
        f"      Built {len(action_rows)} action-only decisions; "
        f"supervised_action_tokens={supervised_tokens}; actions={action_counts}"
    )
    dataset = Dataset.from_list(
        [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": row["labels"],
            }
            for row in action_rows
        ]
    )

    # ── LoRA and Trainer ────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if device == "cuda":
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=output_dir + "_checkpoints",
        num_train_epochs=n_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=ActionOnlyCollator(tokenizer),
    )

    print("\n[3/4] Training SFT (PPO distillation)...\n")
    trainer.train()

    print(f"\n[4/4] Merging LoRA and saving to {output_dir}...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved locally to {output_dir}")

    # ── Smoke eval ──────────────────────────────────────────────────────
    if args.smoke:
        print(f"\n[smoke eval] hard_multi seeds {smoke_eval_seeds}  (PPO-SFT vs Heuristic)")
        merged.eval()
        merged.to(device)
        sft_scores, heur_scores = [], []
        for seed in smoke_eval_seeds:
            sft_r = run_episode_llm(merged, tokenizer, seed, device)
            heur_r = run_episode_heuristic(seed)
            sft_scores.append(sft_r)
            heur_scores.append(heur_r)
            print(f"  seed={seed}: PPO-SFT={sft_r:.4f}  HEU={heur_r:.4f}")
        sft_mean = sum(sft_scores) / len(sft_scores)
        heur_mean = sum(heur_scores) / len(heur_scores)
        delta_pct = ((sft_mean - heur_mean) / heur_mean) * 100
        print(f"\n  Mean: PPO-SFT={sft_mean:.4f}  HEU={heur_mean:.4f}  Δ={delta_pct:+.1f}%")
        print()
        if sft_mean >= heur_mean:
            print("  ✅ PPO-SFT BEATS heuristic baseline! Proceed to --full training.")
        elif sft_mean >= heur_mean - 0.02:
            print("  ⚠️  PPO-SFT is near heuristic. Full training may close the gap.")
        elif sft_mean >= 0.40:
            print("  ⚠️  PPO-SFT learned valid routing but trails heuristic.")
        else:
            print("  ❌ PPO-SFT looks broken (format mismatch or under-training).")

    # ── Push to HF Hub ──────────────────────────────────────────────────
    if args.push_to_hub:
        print(f"\nPushing merged model to https://huggingface.co/{args.push_to_hub} ...")
        merged.push_to_hub(args.push_to_hub, private=args.hub_private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"✅ Pushed to hub: {args.push_to_hub}")
        print(f"   Eval with:  uv run python train/eval_trained.py --model-path {args.push_to_hub}")


if __name__ == "__main__":
    main()
