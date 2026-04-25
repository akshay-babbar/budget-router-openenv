"""
sft_distill.py — Distill the heuristic policy into Qwen3-1.7B with action-only SFT.

OBJECTIVE
    Beat the heuristic baseline on hard_multi by supervised-fine-tuning a 1.7B
    chat model on heuristic trajectories. Honest, non-cheating, non-overfitting:
    train on seeds [200..N+199], evaluate on disjoint dev seeds [0..9] and the
    held-out [100..109] set used by eval_all.py — zero seed leakage.

WHY THIS IS PARETO-OPTIMAL FOR THE HACKATHON
    • The heuristic already scores ~0.61 mean grader on hard_multi. Imitation
      learning recovers most of that as a starting policy in one short run.
    • SAME observation format and SAME system prompt as eval_trained.py — zero
      train/eval distribution shift.
    • Filters teacher trajectories to top-50% by grader (best-of-N filtered BC):
      free quality lift, no extra compute, well-documented technique.
    • LoRA r=8 on q_proj/v_proj — same low-capacity adapter as the GRPO setup,
      keeps overfit risk small and matches the existing repo conventions.
    • Action-only loss: prompt/history tokens are masked out; the model is
      trained only on the next action string. This avoids the failure mode where
      token accuracy improves by reconstructing repeated observation text while
      policy rollout remains poor.
    • One file, two modes (--smoke / --full). Reuses train/eval_trained.py for
      the in-script smoke evaluation — single source of truth for eval logic.

USAGE — LOCAL (Mac MPS is ~20× slower than A10G; use only as last resort)
    uv sync --extra grpo
    uv run python train/sft_distill.py --smoke

USAGE — HF JOBS  (a10g-large)  — recommended path

    REPO: https://github.com/akshay-babbar/budget-router-openenv  (branch: feat/grpo-training-v5)

    Prereqs (one-time):
        - Commit and push `train/sft_distill.py` and `train/launch_sft_hf_job.py`.
        - `hf auth login`  with a token that has WRITE scope (needed to push the
          trained model back to your Hub namespace; container storage is ephemeral).

    IMPORTANT:
        Do NOT use `hf jobs run ... /bin/bash -lc "..."` directly. Current `hf`
        CLI releases parse `-lc` as `-l c` (`--label c`) and submit the wrong
        command. Use `train/launch_sft_hf_job.py`, which calls the official
        Python API and submits `command=["/bin/bash", "-lc", script]` exactly.

    1) SMOKE TEST  (~3–5 min, ~$0.10)  — confirms loss decreases and policy is non-degenerate.

        uv run python train/launch_sft_hf_job.py --smoke

    2) FULL TRAINING  (~30–45 min, ~$0.90)  — run only if smoke prints ✅.

        uv run python train/launch_sft_hf_job.py --full

EVALUATION  (after the job finishes — model lives on the Hub, NOT on your laptop)
    `eval_trained.py` calls `AutoModelForCausalLM.from_pretrained(model_path)`,
    which transparently accepts a Hub repo id. So you can either:

    Option A — eval directly from Hub (no local download step):
        uv run python train/eval_trained.py \\
            --model-path akshay-babbar/sft-qwen3-1.7b-budget-router \\
            --n-episodes 10

    Option B — pull the model locally first (faster re-runs):
        hf download akshay-babbar/sft-qwen3-1.7b-budget-router \\
            --local-dir trained_models/sft_Qwen_Qwen3-1.7B
        uv run python train/eval_trained.py \\
            --model-path trained_models/sft_Qwen_Qwen3-1.7B \\
            --n-episodes 10

DESIGN NOTES (anti-overfit, no over-engineering)
    • Action-only labels are created manually instead of relying on TRL template
      generation masks. This is explicit, version-tolerant, and directly matches
      the decision task.
    • No gradient checkpointing — Qwen3-1.7B + LoRA fits A10G-large (24 GB)
      easily at batch_size=4.
    • No remote dataset upload — trajectories are generated in-process from
      the local environment in seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make `from train.eval_trained import ...` work when invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from budget_router.environment import BudgetRouterEnv
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI

# Reuse the EXACT prompt and observation format used at eval time.
# This guarantees zero train/eval distribution shift.
from train.eval_trained import (  # noqa: E402
    SYSTEM_PROMPT,
    _apply_chat_template,
    _obs_to_text,
    run_episode_heuristic,
    run_episode_llm,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
TRAIN_SEED_OFFSET = 200  # disjoint from dev [0..9] and heldout [100..109]


def collect_heuristic_trajectories(seeds: list[int]) -> list[dict]:
    """Roll out the heuristic on hard_multi for each seed; return chat-format conversations."""
    examples: list[dict] = []
    for seed in seeds:
        env = BudgetRouterEnv()
        obs = env.reset(scenario=HARD_MULTI, seed=seed)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        while not obs.done:
            messages.append({"role": "user", "content": _obs_to_text(obs)})
            action = heuristic_baseline_policy(obs)
            messages.append({"role": "assistant", "content": action.action_type.value})
            obs = env.step(action)
        history = env._internal.history
        # Drop episodes the heuristic itself blew up on — don't teach failure modes.
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
    """
    Convert filtered episodes into per-decision causal-LM examples.

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
        description="SFT-distill the heuristic policy into a small chat LLM."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (~200 episodes, max 30 steps, in-script eval). ~5–8 min on A10G.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Real run (~1000 episodes, 2 epochs, full training). ~45–60 min on A10G.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help=(
            "HF Hub repo id to push the merged model to (e.g. 'akshay-babbar/sft-qwen3-1.7b-budget-router'). "
            "REQUIRED when running on HF Jobs — container storage is ephemeral. "
            "Requires HF_TOKEN in the env (passed via --secrets HF_TOKEN)."
        ),
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the pushed Hub repo private (default: public).",
    )
    args = parser.parse_args()

    if args.smoke and args.full:
        print("❌ Pick exactly one of --smoke or --full.")
        sys.exit(2)
    if not args.smoke and not args.full:
        args.full = True  # safe default for HF Jobs invocations w/o flag

    if args.smoke:
        n_episodes = 200
        n_epochs = 1
        max_steps = 80
        smoke_eval_seeds = [0, 1, 2]
        suffix = "_smoke"
    else:
        n_episodes = 1000
        n_epochs = 2
        max_steps = -1
        smoke_eval_seeds = []
        suffix = ""

    safe_name = args.model_name.replace("/", "_").replace(":", "_")
    output_dir = args.output_dir or f"trained_models/sft_{safe_name}{suffix}"

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    print("=" * 70)
    print(f"SFT Distill — {args.model_name}")
    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'} | Device: {device} | Dtype: {dtype}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    train_seeds = list(range(TRAIN_SEED_OFFSET, TRAIN_SEED_OFFSET + n_episodes))
    print(
        f"\n[1/3] Collecting {n_episodes} heuristic trajectories on seeds "
        f"[{train_seeds[0]}..{train_seeds[-1]}] (disjoint from eval seeds)..."
    )
    raw_examples = collect_heuristic_trajectories(train_seeds)
    if not raw_examples:
        print("❌ No completed heuristic episodes — env or scenario regression?")
        sys.exit(1)

    teacher_mean = sum(e["grader"] for e in raw_examples) / len(raw_examples)
    examples = filter_top_half(raw_examples)
    kept_mean = sum(e["grader"] for e in examples) / len(examples)
    print(
        f"      Collected {len(raw_examples)} completed → kept top {len(examples)}; "
        f"teacher_mean={teacher_mean:.4f} kept_mean={kept_mean:.4f}"
    )

    print(f"\n[2/3] Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = 2048
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

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir + "_checkpoints",
        num_train_epochs=n_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        logging_steps=2,
        save_strategy="no",  # we save the merged model manually below
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

    print("\n[3/3] Training SFT...\n")
    trainer.train()

    print(f"\nMerging LoRA and saving to {output_dir}...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved locally to {output_dir}")

    # ── Smoke eval: prove the pipeline is end-to-end working ────────────────
    if args.smoke:
        print(f"\n[smoke eval] hard_multi seeds {smoke_eval_seeds}  (SFT vs Heuristic)")
        merged.eval()
        merged.to(device)
        sft_scores, heur_scores = [], []
        for seed in smoke_eval_seeds:
            sft_r = run_episode_llm(merged, tokenizer, seed, device)
            heur_r = run_episode_heuristic(seed)
            sft_scores.append(sft_r)
            heur_scores.append(heur_r)
            print(f"  seed={seed}: SFT={sft_r:.4f}  HEU={heur_r:.4f}")
        sft_mean = sum(sft_scores) / len(sft_scores)
        heur_mean = sum(heur_scores) / len(heur_scores)
        print(f"\n  Mean: SFT={sft_mean:.4f}  HEU={heur_mean:.4f}")
        print()
        # Decision rule for the hackathon objective:
        # - <0.40: format/policy broken.
        # - >= heuristic - 0.02: worth scaling or using as a GRPO warm start.
        # Anything in between is useful evidence but not enough for a full run.
        if sft_mean >= heur_mean - 0.02:
            print("  ✅ STRONG EVIDENCE: action-only SFT is near the heuristic baseline.")
            print("     Proceed to 10-seed eval, then consider --full or GRPO warm start.")
        elif sft_mean >= 0.40:
            print("  ⚠️  PARTIAL EVIDENCE: action-only SFT learned valid routing but trails heuristic.")
            print("     Do a 10-seed eval before spending on --full.")
        else:
            print("  ❌ NO EVIDENCE: SFT output looks broken (likely format mismatch or under-training).")
            print("     Investigate before spending money on --full or GRPO.")

    # ── Push to HF Hub (REQUIRED on ephemeral HF Jobs containers) ──────────
    if args.push_to_hub:
        print(f"\nPushing merged model to https://huggingface.co/{args.push_to_hub} ...")
        merged.push_to_hub(args.push_to_hub, private=args.hub_private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"✅ Pushed to hub: {args.push_to_hub}")
        print(f"   Eval locally with:  uv run python train/eval_trained.py --model-path {args.push_to_hub}")


if __name__ == "__main__":
    main()
