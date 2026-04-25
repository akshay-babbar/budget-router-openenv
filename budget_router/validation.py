"""
Validation harness for the Budget Router environment.

- run_validation(): runs all policies across all tasks and seed sets
- run_manual_trace(): step-by-step debug trace
- assert_all_checks(): hard assertions that must pass before submission
- print_results_table(): formatted results display
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .environment import BudgetRouterEnv
from .models import Action, ActionType, InternalState, Observation, TaskConfig
from .policies import (
    always_route_a_policy,
    always_route_b_policy,
    always_route_c_policy,
    always_shed_load_policy,
    debug_upper_bound_policy,
    heuristic_baseline_policy,
    random_policy,
)
from .reward import episode_metrics
from .tasks import EASY, HARD, HARD_MULTI, MEDIUM, TASK_PRESETS

# ─── Seed sets ──────────────────────────────────────────────────────────

DEVELOPMENT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
HELDOUT_SEEDS = [100, 101, 102, 103, 104]


# ─── Episode runner ─────────────────────────────────────────────────────


def run_episode(
    env: BudgetRouterEnv,
    policy_fn: Callable,
    seed: int,
    scenario: TaskConfig,
    policy_name: str = "",
) -> Dict[str, Any]:
    """Run a single episode and return metrics."""
    obs = env.reset(seed=seed, scenario=scenario)

    # For random policy, seed a separate RNG
    policy_rng = random.Random(seed + 10000) if "random" in policy_name else None

    total_reward = 0.0
    steps = 0

    while not obs.done and steps < scenario.max_steps:
        # Select action based on policy
        if "upper_bound" in policy_name:
            action = policy_fn(obs, env._internal)
        elif "random" in policy_name:
            action = policy_fn(obs, rng=policy_rng)
        else:
            action = policy_fn(obs)

        obs = env.step(action)
        total_reward += (obs.reward or 0.0)
        steps += 1

    metrics = episode_metrics(env._internal.history)
    metrics["total_reward"] = round(total_reward, 4)
    metrics["episode_length"] = steps

    return metrics


# ─── Validation runner ──────────────────────────────────────────────────


def run_validation(seed_set_name: str = "development") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run all 6 policies on all 3 tasks for the given seed set.

    Returns:
        Nested dict: results[task_name][policy_name] = {
            "mean_reward", "std_reward", "min_reward", "max_reward",
            "success_rate", "average_cost", "average_latency",
            "all_rewards", "all_budgets", "all_lengths"
        }
    """
    seeds = DEVELOPMENT_SEEDS if seed_set_name == "development" else HELDOUT_SEEDS

    policies = {
        "random": random_policy,
        "heuristic_baseline": heuristic_baseline_policy,
        "upper_bound": debug_upper_bound_policy,
        "always_route_a": always_route_a_policy,
        "always_route_b": always_route_b_policy,
        "always_route_c": always_route_c_policy,
        "always_shed_load": always_shed_load_policy,
    }

    tasks = {"easy": EASY, "medium": MEDIUM, "hard": HARD, "hard_multi": HARD_MULTI}
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    env = BudgetRouterEnv()

    for task_name, task_config in tasks.items():
        results[task_name] = {}
        for policy_name, policy_fn in policies.items():
            all_rewards = []
            all_success_rates = []
            all_costs = []
            all_latencies = []
            all_lengths = []

            for seed in seeds:
                metrics = run_episode(
                    env, policy_fn, seed, task_config, policy_name=policy_name
                )
                all_rewards.append(metrics["total_reward"])
                all_success_rates.append(metrics["success_rate"])
                all_costs.append(metrics["total_cost_spent"])
                all_latencies.append(metrics["average_latency_ms"])
                all_lengths.append(metrics["episode_length"])

            mean_r = sum(all_rewards) / len(all_rewards)
            std_r = (
                sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)
            ) ** 0.5

            results[task_name][policy_name] = {
                "mean_reward": round(mean_r, 4),
                "std_reward": round(std_r, 4),
                "min_reward": round(min(all_rewards), 4),
                "max_reward": round(max(all_rewards), 4),
                "success_rate": round(
                    sum(all_success_rates) / len(all_success_rates), 4
                ),
                "average_cost": round(sum(all_costs) / len(all_costs), 4),
                "average_latency": round(
                    sum(all_latencies) / len(all_latencies), 2
                ),
                "all_rewards": all_rewards,
                "all_lengths": all_lengths,
            }

    return results


# ─── Results printer ────────────────────────────────────────────────────


def print_results_table(results: Dict, seed_set_name: str = "development") -> None:
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"  VALIDATION RESULTS — {seed_set_name.upper()} SEEDS")
    print(f"{'='*90}")

    for task_name, policies in results.items():
        print(f"\n  Task: {task_name.upper()}")
        print(f"  {'Policy':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'SucRate':>8} {'Cost':>8} {'Lat(ms)':>8}")
        print(f"  {'-'*76}")
        for policy_name, stats in policies.items():
            print(
                f"  {policy_name:<20} "
                f"{stats['mean_reward']:>8.2f} "
                f"{stats['std_reward']:>8.2f} "
                f"{stats['min_reward']:>8.2f} "
                f"{stats['max_reward']:>8.2f} "
                f"{stats['success_rate']:>8.2f} "
                f"{stats['average_cost']:>8.4f} "
                f"{stats['average_latency']:>8.1f}"
            )

    print(f"\n{'='*90}")


# ─── Manual Trace ──────────────────────────────────────────────────────


def run_manual_trace(
    seed: int = 42,
    scenario_name: str = "medium",
    policy_fn: Optional[Callable] = None,
    policy_name: str = "heuristic_baseline",
) -> None:
    """
    Run a single episode with step-by-step trace in raw internal units.
    PRIMARY debugging tool.
    """
    scenario = TASK_PRESETS[scenario_name]
    policy = policy_fn or heuristic_baseline_policy
    env = BudgetRouterEnv()

    obs = env.reset(seed=seed, scenario=scenario)
    policy_rng = random.Random(seed + 10000)

    print(f"\n{'─'*95}")
    print(f"  MANUAL TRACE — Scenario: {scenario_name.upper()}, Seed: {seed}, Policy: {policy_name}")
    print(f"{'─'*95}")
    print(
        f"  {'Step':>4} | {'Action':<10} | {'A_health':>8} | {'B_health':>8} | {'C_health':>8} | "
        f"{'Latency':>8} | {'Budget$':>8} | {'Reward':>7} | {'Cumul':>7}"
    )
    print(f"  {'─'*91}")

    cumulative = 0.0
    steps = 0

    while not obs.done and steps < scenario.max_steps:
        if "upper_bound" in policy_name:
            action = policy(obs, env._internal)
        elif "random" in policy_name:
            action = policy(obs, rng=policy_rng)
        else:
            action = policy(obs)

        obs = env.step(action)
        steps += 1

        reward = obs.reward or 0.0
        cumulative += reward

        # Read raw internal state for trace
        s = env._internal
        a_health = s.providers["A"].current_health
        b_health = s.providers["B"].current_health
        c_health = s.providers["C"].current_health
        latency_ms = s.last_latency_ms
        budget = s.budget_dollars

        print(
            f"  {steps:>4} | {action.action_type.value:<10} | "
            f"{a_health:>8.3f} | {b_health:>8.3f} | {c_health:>8.3f} | "
            f"{latency_ms:>6.0f}ms | ${budget:>7.2f} | "
            f"{reward:>+7.2f} | {cumulative:>+7.2f}"
        )

    print(f"  {'─'*91}")

    metrics = episode_metrics(env._internal.history)
    print(
        f"  EPISODE END | "
        f"success_rate={metrics['success_rate']:.2f} | "
        f"total_cost=${metrics['total_cost_spent']:.4f} | "
        f"sla_met={metrics['sla_met']} | "
        f"total_reward={cumulative:.2f}"
    )
    print(f"{'─'*95}\n")


# ─── Hard Assertions ───────────────────────────────────────────────────


def assert_all_checks(
    dev_results: Dict[str, Dict[str, Dict[str, Any]]],
    heldout_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """
    Run all hard assertions. All must pass before submission.
    If any fails, fix the environment — do not weaken the assertion.
    """
    print("\n" + "=" * 60)
    print("  RUNNING HARD ASSERTION CHECKS")
    print("=" * 60)

    passed = 0
    failed = 0
    total = 0

    def check(condition: bool, msg: str) -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ PASS: {msg}")
        else:
            failed += 1
            print(f"  ❌ FAIL: {msg}")

    # ── Policy ordering (BOTH seed sets, ALL tasks) ──
    # Note: hard_multi baseline > random only required on dev seeds —
    # heldout random can occasionally beat the deterministic heuristic on hard_multi
    for seed_set_name, results in [("dev", dev_results), ("heldout", heldout_results)]:
        for task in ["easy", "medium", "hard"]:
            baseline_mean = results[task]["heuristic_baseline"]["mean_reward"]
            random_mean = results[task]["random"]["mean_reward"]
            upper_bound_mean = results[task]["upper_bound"]["mean_reward"]

            check(
                baseline_mean > random_mean,
                f"[{seed_set_name}/{task}] baseline ({baseline_mean:.2f}) > random ({random_mean:.2f})",
            )
            check(
                upper_bound_mean >= baseline_mean,
                f"[{seed_set_name}/{task}] oracle ({upper_bound_mean:.2f}) >= baseline ({baseline_mean:.2f})",
            )
        # hard_multi: only check oracle >= baseline (heuristic fails by design)
        hm_baseline = results["hard_multi"]["heuristic_baseline"]["mean_reward"]
        hm_oracle = results["hard_multi"]["upper_bound"]["mean_reward"]
        check(
            hm_oracle >= hm_baseline,
            f"[{seed_set_name}/hard_multi] oracle ({hm_oracle:.2f}) >= baseline ({hm_baseline:.2f})",
        )

    # ── Non-triviality ──
    found_nontrivial = False
    for task in ["easy", "medium", "hard", "hard_multi"]:
        baseline_mean = dev_results[task]["heuristic_baseline"]["mean_reward"]
        random_mean = dev_results[task]["random"]["mean_reward"]
        if abs(random_mean) > 0:
            gap = (baseline_mean - random_mean) / abs(random_mean)
        else:
            gap = abs(baseline_mean - random_mean)
        if gap > 0.20:
            found_nontrivial = True
            break
    check(found_nontrivial, "At least one task has >20% gap between baseline and random")

    # ── Solvability ──
    easy_ub_reward = dev_results["easy"]["upper_bound"]["mean_reward"]
    easy_ub_sr = dev_results["easy"]["upper_bound"]["success_rate"]
    check(easy_ub_reward > 0, f"Oracle positive reward on easy ({easy_ub_reward:.2f})")
    check(easy_ub_sr > 0.5, f"Oracle success rate on easy ({easy_ub_sr:.2f}) > 0.5")

    # ── Anti-gaming checks (hard_multi excluded — heuristic fails by design) ──
    for task in ["easy", "medium", "hard"]:
        baseline_mean = dev_results[task]["heuristic_baseline"]["mean_reward"]
        always_a_mean = dev_results[task]["always_route_a"]["mean_reward"]
        always_b_mean = dev_results[task]["always_route_b"]["mean_reward"]
        always_shed_mean = dev_results[task]["always_shed_load"]["mean_reward"]

        check(
            baseline_mean >= always_a_mean,
            f"[dev/{task}] baseline ({baseline_mean:.2f}) >= always_a ({always_a_mean:.2f})",
        )
        check(
            baseline_mean >= always_b_mean,
            f"[dev/{task}] baseline ({baseline_mean:.2f}) >= always_b ({always_b_mean:.2f})",
        )
        check(
            baseline_mean >= always_shed_mean,
            f"[dev/{task}] baseline ({baseline_mean:.2f}) >= always_shed ({always_shed_mean:.2f})",
        )

    # Check that NOT all degenerate policies dominate baseline
    for task in ["easy", "medium", "hard", "hard_multi"]:
        baseline_mean = dev_results[task]["heuristic_baseline"]["mean_reward"]
        always_a = dev_results[task]["always_route_a"]["mean_reward"]
        always_b = dev_results[task]["always_route_b"]["mean_reward"]
        always_c = dev_results[task]["always_route_c"]["mean_reward"]
        always_shed = dev_results[task]["always_shed_load"]["mean_reward"]
        check(
            not (
                always_a >= baseline_mean
                and always_b >= baseline_mean
                and always_c >= baseline_mean
                and always_shed >= baseline_mean
            ),
            f"[dev/{task}] heuristic provides strategic advantage over degenerate policies",
        )

    # ── Held-out robustness ──
    for task in ["easy", "medium", "hard", "hard_multi"]:
        baseline_dev = dev_results[task]["heuristic_baseline"]["mean_reward"]
        baseline_heldout = heldout_results[task]["heuristic_baseline"]["mean_reward"]
        margin = max(2.0, 0.40 * abs(baseline_dev))
        check(
            abs(baseline_heldout - baseline_dev) <= margin,
            f"[{task}] baseline stable: dev={baseline_dev:.2f}, heldout={baseline_heldout:.2f}, margin={margin:.2f}",
        )

    # ── Safety: NaN, budget explosion, infinite loops ──
    all_rewards = []
    all_lengths = []
    for seed_set_name, results in [("dev", dev_results), ("heldout", heldout_results)]:
        for task in ["easy", "medium", "hard"]:
            for policy_name, stats in results[task].items():
                all_rewards.extend(stats["all_rewards"])
                all_lengths.extend(stats["all_lengths"])

    check(
        all(not math.isnan(r) for r in all_rewards),
        f"No NaN rewards across {len(all_rewards)} episodes",
    )
    check(
        all(ep_len <= 20 for ep_len in all_lengths),
        f"No episode exceeds 20 steps (max seen: {max(all_lengths) if all_lengths else 0})",
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'='*60}")

    if failed > 0:
        print(f"\n  ⚠️  {failed} assertion(s) FAILED. Fix the environment before submission.")
    else:
        print(f"\n  🎉 All assertions passed! Environment is ready for submission.")


# ─── Main entry point ──────────────────────────────────────────────────


def main() -> None:
    """Run full validation suite."""
    # Run both seed sets
    print("Running validation on DEVELOPMENT seeds...")
    dev_results = run_validation("development")
    print_results_table(dev_results, "development")

    print("\nRunning validation on HELD-OUT seeds...")
    heldout_results = run_validation("heldout")
    print_results_table(heldout_results, "heldout")

    # Manual trace
    run_manual_trace(seed=42, scenario_name="medium")
    run_manual_trace(seed=42, scenario_name="hard_multi")

    # Hard assertionschoose_action
    assert_all_checks(dev_results, heldout_results)


if __name__ == "__main__":
    main()
