"""
sft_distill.py — Distill the heuristic policy into Qwen3-1.7B via TRL SFTTrainer.

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
    • assistant_only_loss is OFF — Qwen3's stock chat template lacks
      {% generation %} markers, and pulling a remote template at job-time adds
      fragility. The action strings are short ("route_to_a") so loss dilution
      from user-token training is minor and bounded.
    • No gradient checkpointing — Qwen3-1.7B + LoRA fits A10G-large (24 GB)
      easily at batch_size=4.
    • No remote dataset upload — trajectories are generated in-process from
      the local environment in seconds.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make `from train.eval_trained import ...` work when invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from budget_router.environment import BudgetRouterEnv
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI

# Reuse the EXACT prompt and observation format used at eval time.
# This guarantees zero train/eval distribution shift.
from train.eval_trained import (  # noqa: E402
    SYSTEM_PROMPT,
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
        max_steps = 30
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

    dataset = Dataset.from_list([{"messages": e["messages"]} for e in examples])

    print(f"\n[2/3] Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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

    sft_config = SFTConfig(
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
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
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
        # Decision rule: random/garbage policies score ~0.15–0.25 on hard_multi.
        # If SFT > 0.40 we have evidence the format & policy are being learned.
        if sft_mean >= 0.40:
            print("  ✅ EVIDENCE: SFT learned the action format and is producing useful actions.")
            print("     Proceed to --full for the real comparison.")
        else:
            print("  ❌ NO EVIDENCE: SFT output looks broken (likely format mismatch or under-training).")
            print("     Investigate before spending money on --full.")

    # ── Push to HF Hub (REQUIRED on ephemeral HF Jobs containers) ──────────
    if args.push_to_hub:
        print(f"\nPushing merged model to https://huggingface.co/{args.push_to_hub} ...")
        merged.push_to_hub(args.push_to_hub, private=args.hub_private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"✅ Pushed to hub: {args.push_to_hub}")
        print(f"   Eval locally with:  uv run python train/eval_trained.py --model-path {args.push_to_hub}")


if __name__ == "__main__":
    main()
