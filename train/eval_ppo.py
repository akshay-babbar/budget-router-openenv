"""
Evaluate the trained PPO agent against the heuristic baseline.

Usage:
    uv run python train/eval_ppo.py

Loads trained_models/ppo_easy_50k.zip and runs 10 episodes (seeds 0-9),
reporting per-episode and mean grader scores.
"""
from __future__ import annotations

import statistics
from pathlib import Path

from stable_baselines3 import PPO

from train.gym_wrapper import BudgetRouterGymEnv
from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import EASY

MODEL_PATH = "trained_models/ppo_easy_50k.zip"
EVAL_SEEDS = list(range(10))    # seeds 0-9 (development set)
HEURISTIC_BASELINE = 0.7958     # confirmed grader score from README


def _grader_score_from_history(history: list[dict]) -> float:
    """Compute grader score directly from the environment's history dict."""
    return float(grade_episode(history)["overall_score"])


def eval_ppo(model: PPO, seeds: list[int]) -> list[float]:
    """Run PPO policy for each seed, return list of grader scores."""
    scores = []
    for seed in seeds:
        env = BudgetRouterGymEnv(scenario=EASY, seed=seed)
        inner_env = env._env          # direct access to BudgetRouterEnv for history

        obs, _ = env.reset()
        done = False
        while not done:
            action_idx, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action_idx))
            done = terminated or truncated

        score = _grader_score_from_history(inner_env._internal.history)
        scores.append(score)
        print(f"  seed={seed:2d}  grader={score:.4f}")
    return scores


def eval_heuristic(seeds: list[int]) -> list[float]:
    """Run heuristic policy for each seed, return list of grader scores."""
    scores = []
    for seed in seeds:
        env = BudgetRouterEnv()
        obs = env.reset(seed=seed, scenario=EASY)
        while not obs.done:
            action = heuristic_baseline_policy(obs)
            obs = env.step(action)
        score = _grader_score_from_history(env._internal.history)
        scores.append(score)
    return scores


def main() -> None:
    if not Path(MODEL_PATH).exists():
        print(f"[eval] Model not found at {MODEL_PATH}. Run train/train_ppo.py first.")
        return

    print(f"[eval] Loading {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    print("\n[eval] PPO agent (deterministic):")
    ppo_scores = eval_ppo(model, EVAL_SEEDS)
    ppo_mean = statistics.mean(ppo_scores)

    print("\n[eval] Heuristic baseline:")
    heuristic_scores = eval_heuristic(EVAL_SEEDS)
    heuristic_mean = statistics.mean(heuristic_scores)

    print("\n── Results ──────────────────────────────────")
    print(f"  PPO mean grader score  : {ppo_mean:.4f}")
    print(f"  Heuristic mean grader  : {heuristic_mean:.4f}  (expected ≈ {HEURISTIC_BASELINE})")
    delta = ppo_mean - heuristic_mean
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta (PPO - heuristic): {sign}{delta:.4f}")
    if ppo_mean > 0.60:
        print("  ✅ PPO > 0.60 threshold — README update warranted.")
    else:
        print("  ⚠️  PPO < 0.60 — keep scaffolding but skip README PPO row.")


if __name__ == "__main__":
    main()
