"""
Statistical evaluation: PPO vs Heuristic on Hard_Multi.

This experiment produces the primary quality signal:
- If PPO materially and consistently exceeds the heuristic on Hard_Multi,
  it demonstrates the environment contains a learnable signal that reactive
  policies cannot exploit — i.e., the environment is high quality.

- If PPO merely ties or is worse, it suggests the 100k step budget is
  insufficient or the signal is too sparse.

Usage:
    uv run python train/eval_hard_multi.py

Output:
    Printed statistical report + per-seed breakdown.
    If improvements are significant, a README snippet is printed.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from train.gym_wrapper import BudgetRouterGymEnv
from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI

MODEL_PATH = "trained_models/ppo_hard_multi_100k.zip"
EVAL_SEEDS = list(range(10))
HEURISTIC_BASELINE_GRADER = 0.6094  # confirmed from README (dev seeds 0-9)


def _grader(history: list[dict]) -> float:
    return float(grade_episode(history)["overall_score"])


def _grader_breakdown(history: list[dict]) -> dict:
    g = grade_episode(history)
    return {k: round(float(v), 4) for k, v in g.items()}


def eval_ppo(model: PPO, seeds: list[int]) -> tuple[list[float], list[dict]]:
    scores, breakdowns = [], []
    for seed in seeds:
        env = BudgetRouterGymEnv(scenario=HARD_MULTI, seed=seed)
        inner = env._env

        obs, _ = env.reset()
        done = False
        while not done:
            action_idx, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action_idx))
            done = terminated or truncated

        bd = _grader_breakdown(inner._internal.history)
        scores.append(bd["overall_score"])
        breakdowns.append(bd)
        print(f"  [PPO]  seed={seed:2d}  overall={bd['overall_score']:.4f}"
              f"  adapt={bd['adaptation_score']:.4f}"
              f"  budget={bd['budget_score']:.4f}"
              f"  success={bd['success_score']:.4f}")
    return scores, breakdowns


def eval_heuristic(seeds: list[int]) -> tuple[list[float], list[dict]]:
    scores, breakdowns = [], []
    for seed in seeds:
        env = BudgetRouterEnv()
        obs = env.reset(seed=seed, scenario=HARD_MULTI)
        while not obs.done:
            obs = env.step(heuristic_baseline_policy(obs))
        bd = _grader_breakdown(env._internal.history)
        scores.append(bd["overall_score"])
        breakdowns.append(bd)
        print(f"  [HEU]  seed={seed:2d}  overall={bd['overall_score']:.4f}"
              f"  adapt={bd['adaptation_score']:.4f}"
              f"  budget={bd['budget_score']:.4f}"
              f"  success={bd['success_score']:.4f}")
    return scores, breakdowns


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """95% CI using t-distribution (small sample)."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, mean
    se = statistics.stdev(values) / math.sqrt(n)
    # t-critical ≈ 2.262 for df=9 (n=10), 95% two-tailed
    t_crit = 2.262
    margin = t_crit * se
    return mean - margin, mean + margin


def main() -> None:
    if not Path(MODEL_PATH).exists():
        print(f"[eval] Model not found at {MODEL_PATH}. Run train/train_ppo_hard_multi.py first.")
        return

    print(f"[eval] Loading {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    print("\n─── PPO agent (deterministic, Hard_Multi) ───")
    ppo_scores, ppo_breakdowns = eval_ppo(model, EVAL_SEEDS)

    print("\n─── Heuristic baseline (Hard_Multi) ───")
    heuristic_scores, heuristic_breakdowns = eval_heuristic(EVAL_SEEDS)

    # ── Statistics ──────────────────────────────────────────────────────────
    ppo_mean    = statistics.mean(ppo_scores)
    ppo_std     = statistics.stdev(ppo_scores)
    heu_mean    = statistics.mean(heuristic_scores)
    heu_std     = statistics.stdev(heuristic_scores)

    delta       = ppo_mean - heu_mean
    delta_pct   = (delta / heu_mean) * 100 if heu_mean > 0 else float("nan")

    ppo_lo, ppo_hi    = _confidence_interval_95(ppo_scores)
    heu_lo, heu_hi    = _confidence_interval_95(heuristic_scores)

    # Win rate: fraction of seeds where PPO > heuristic
    win_rate = sum(p > h for p, h in zip(ppo_scores, heuristic_scores)) / len(ppo_scores)

    # Sub-score deltas
    avg_adapt_ppo = statistics.mean(b["adaptation_score"] for b in ppo_breakdowns)
    avg_adapt_heu = statistics.mean(b["adaptation_score"] for b in heuristic_breakdowns)
    avg_budget_ppo = statistics.mean(b["budget_score"] for b in ppo_breakdowns)
    avg_budget_heu = statistics.mean(b["budget_score"] for b in heuristic_breakdowns)

    sign = "+" if delta >= 0 else ""

    print(f"""
══════════════════════════════════════════════════════════
  HARD_MULTI: PPO vs Heuristic — Statistical Report
══════════════════════════════════════════════════════════

  PPO  grader:  {ppo_mean:.4f} ± {ppo_std:.4f}   95% CI [{ppo_lo:.4f}, {ppo_hi:.4f}]
  HEU  grader:  {heu_mean:.4f} ± {heu_std:.4f}   95% CI [{heu_lo:.4f}, {heu_hi:.4f}]
  Delta:        {sign}{delta:.4f}  ({sign}{delta_pct:.1f}%)
  Win rate:     {win_rate:.0%}  ({int(win_rate*len(ppo_scores))}/{len(ppo_scores)} seeds PPO wins)

  ── Sub-score breakdown ──
  Adaptation:   PPO={avg_adapt_ppo:.4f}  HEU={avg_adapt_heu:.4f}  Δ={avg_adapt_ppo-avg_adapt_heu:+.4f}
  Budget:       PPO={avg_budget_ppo:.4f}  HEU={avg_budget_heu:.4f}  Δ={avg_budget_ppo-avg_budget_heu:+.4f}
""")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if ppo_lo > heu_hi:
        verdict = "✅ STRONG: PPO 95% CI is entirely above heuristic 95% CI — non-overlapping."
    elif ppo_mean > heu_mean and win_rate >= 0.70:
        verdict = f"✅ CLEAR: PPO wins {win_rate:.0%} of seeds with positive mean improvement."
    elif ppo_mean > heu_mean and win_rate >= 0.50:
        verdict = f"⚠️  MODERATE: PPO wins {win_rate:.0%} of seeds — improvement present but with variance."
    elif ppo_mean > heu_mean:
        verdict = f"⚠️  WEAK: Mean improvement is positive but PPO wins only {win_rate:.0%} of seeds."
    else:
        verdict = f"❌ NO IMPROVEMENT: PPO ({ppo_mean:.4f}) ≤ heuristic ({heu_mean:.4f}) — more training needed."

    print(f"  VERDICT: {verdict}")
    print("══════════════════════════════════════════════════════════\n")

    if ppo_mean > heu_mean:
        print("  README snippet (paste into benchmark table):")
        print(f"  Hard_Multi PPO row: {ppo_mean:.4f}  (Δ {sign}{delta_pct:.1f}% vs heuristic)")


if __name__ == "__main__":
    main()
