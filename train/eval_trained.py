"""
eval_trained.py — Evaluate the GRPO-trained model against the heuristic baseline.

Loads the merged model from trained_models/grpo_qwen3_0.6b/ directly (no API server needed).
Runs N episodes on hard_multi and prints mean reward vs heuristic baseline.

USAGE
    uv run python train/eval_trained.py

HOW IT WORKS
    The trained model is loaded as a plain AutoModelForCausalLM (LoRA already merged).
    At each step, we feed the current observation as a chat message and parse the
    model's text output as a tool call (same _parse_llm_action logic as inference.py).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI

N_EPISODES = 10
SCENARIO = HARD_MULTI

SYSTEM_PROMPT = (
    "You are a budget-aware API router. "
    "Use the available tools to route each request to the best provider. "
    "Providers can degrade mid-episode — monitor health and switch early.\n\n"
    "At each step output EXACTLY ONE action string from: "
    "route_to_a | route_to_b | route_to_c | shed_load"
)

_VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]


def _parse_action(text: str) -> str:
    text = text.strip().lower()
    for a in _VALID_ACTIONS:
        if a in text:
            return a
    return "shed_load"


def _obs_to_text(obs: Observation) -> str:
    return (
        f"provider_a_status: {obs.provider_a_status:.3f}\n"
        f"provider_b_status: {obs.provider_b_status:.3f}\n"
        f"provider_c_status: {obs.provider_c_status:.3f}\n"
        f"budget_remaining:  {obs.budget_remaining:.3f}\n"
        f"step_count:        {obs.step_count:.3f}\n"
        f"Your action:"
    )


def run_episode_llm(model, tokenizer, seed: int, device: str) -> float:
    env = BudgetRouterEnv()
    obs = env.reset(scenario=SCENARIO, seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.done:
        messages.append({"role": "user", "content": _obs_to_text(obs)})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            # Older Transformers versions may not expose chat_template_kwargs here.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        action_str = _parse_action(generated)
        messages.append({"role": "assistant", "content": action_str})

        action = Action(action_type=ActionType(action_str))
        obs = env.step(action)

    return float(grade_episode(env._internal.history)["overall_score"])


def run_episode_heuristic(seed: int) -> float:
    env = BudgetRouterEnv()
    obs = env.reset(scenario=SCENARIO, seed=seed)
    while not obs.done:
        action = heuristic_baseline_policy(obs)
        obs = env.step(action)
    return float(grade_episode(env._internal.history)["overall_score"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model vs heuristic baseline.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models/grpo_Qwen_Qwen3-1.7B",
        help="Path to merged trained model directory (default: trained_models/grpo_Qwen_Qwen3-1.7B).",
    )
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES, help="Number of eval episodes.")
    args = parser.parse_args()

    model_path = args.model_path
    is_hub_id = "/" in model_path and not os.path.exists(model_path)
    if not os.path.exists(model_path) and not is_hub_id:
        print(f"❌ Trained model not found at {model_path}")
        print("   Pass a local model directory or a Hugging Face Hub model id.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading trained model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"\nRunning {args.n_episodes} episodes on {SCENARIO.name} ...")
    print(f"{'Seed':<6} {'LLM':>8} {'Heuristic':>12}")
    print("-" * 30)

    llm_scores, heuristic_scores = [], []
    for seed in range(args.n_episodes):
        llm_r = run_episode_llm(model, tokenizer, seed, device)
        heur_r = run_episode_heuristic(seed)
        llm_scores.append(llm_r)
        heuristic_scores.append(heur_r)
        print(f"{seed:<6} {llm_r:>8.4f} {heur_r:>12.4f}")

    llm_mean = sum(llm_scores) / len(llm_scores)
    heur_mean = sum(heuristic_scores) / len(heuristic_scores)

    print("-" * 30)
    print(f"{'Mean':<6} {llm_mean:>8.4f} {heur_mean:>12.4f}")
    print()
    if llm_mean >= heur_mean:
        print(f"✅ LLM ({llm_mean:.4f}) >= Heuristic ({heur_mean:.4f}) — BEATS BASELINE")
    else:
        gap = heur_mean - llm_mean
        print(f"⚠️  LLM ({llm_mean:.4f}) < Heuristic ({heur_mean:.4f}) — gap={gap:.4f}")


if __name__ == "__main__":
    main()
