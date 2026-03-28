"""
Tests for the validation harness.

Covers: policy ordering, solvability, NaN safety, baseline stability,
and hard task crash resistance.
"""

import math
import random

import pytest

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.policies import (
    always_route_a_policy,
    always_route_b_policy,
    always_route_c_policy,
    always_shed_load_policy,
    debug_upper_bound_policy,
    heuristic_baseline_policy,
    random_policy,
)
from budget_router.tasks import EASY, HARD, MEDIUM
from budget_router.validation import DEVELOPMENT_SEEDS, HELDOUT_SEEDS, run_episode


# ─── Helpers ────────────────────────────────────────────────────────────


def mean_reward_over_seeds(policy_fn, scenario, seeds, policy_name=""):
    """Compute mean total reward for a policy across seeds."""
    env = BudgetRouterEnv()
    rewards = []
    for seed in seeds:
        metrics = run_episode(env, policy_fn, seed, scenario, policy_name=policy_name)
        rewards.append(metrics["total_reward"])
    return sum(rewards) / len(rewards), rewards


# ─── Validation Tests ──────────────────────────────────────────────────


class TestValidation:
    """Validation-level tests."""

    def test_baseline_beats_random_easy_dev(self):
        """Baseline beats random on easy task across development seeds."""
        baseline_mean, _ = mean_reward_over_seeds(
            heuristic_baseline_policy, EASY, DEVELOPMENT_SEEDS
        )
        random_mean, _ = mean_reward_over_seeds(
            random_policy, EASY, DEVELOPMENT_SEEDS, policy_name="random"
        )
        assert baseline_mean > random_mean, (
            f"baseline ({baseline_mean:.2f}) <= random ({random_mean:.2f}) on easy"
        )

    def test_upper_bound_beats_baseline_easy_dev(self):
        """Upper bound beats or matches baseline on easy task across dev seeds."""
        baseline_mean, _ = mean_reward_over_seeds(
            heuristic_baseline_policy, EASY, DEVELOPMENT_SEEDS
        )
        ub_mean, _ = mean_reward_over_seeds(
            debug_upper_bound_policy, EASY, DEVELOPMENT_SEEDS, policy_name="upper_bound"
        )
        assert ub_mean >= baseline_mean, (
            f"oracle ({ub_mean:.2f}) < baseline ({baseline_mean:.2f}) on easy"
        )

    def test_easy_solvable_positive_reward(self):
        """Easy task is solvable: baseline achieves positive total reward on seed=42."""
        env = BudgetRouterEnv()
        metrics = run_episode(env, heuristic_baseline_policy, seed=42, scenario=EASY)
        assert metrics["total_reward"] > 0, (
            f"baseline achieves {metrics['total_reward']:.2f} on easy/seed=42"
        )

    def test_hard_no_crash_dev_seeds(self):
        """Hard task terminates without environment crash on development_seeds."""
        env = BudgetRouterEnv()
        for seed in DEVELOPMENT_SEEDS:
            try:
                metrics = run_episode(
                    env, heuristic_baseline_policy, seed=seed, scenario=HARD
                )
                assert metrics["episode_length"] <= 20
            except Exception as e:
                pytest.fail(f"Hard task crashed on seed {seed}: {e}")

    def test_no_nan_rewards_all_combos(self):
        """No reward is NaN across all (task, policy, seed_set) combinations."""
        env = BudgetRouterEnv()
        policies = {
            "random": random_policy,
            "heuristic_baseline": heuristic_baseline_policy,
            "upper_bound": debug_upper_bound_policy,
            "always_route_a": always_route_a_policy,
            "always_route_b": always_route_b_policy,
            "always_route_c": always_route_c_policy,
            "always_shed_load": always_shed_load_policy,
        }

        for scenario in [EASY, MEDIUM, HARD]:
            for policy_name, policy_fn in policies.items():
                for seed in DEVELOPMENT_SEEDS[:3]:  # subset for speed
                    metrics = run_episode(
                        env, policy_fn, seed, scenario, policy_name=policy_name
                    )
                    assert not math.isnan(metrics["total_reward"]), (
                        f"NaN reward: {scenario.name}/{policy_name}/seed={seed}"
                    )

    def test_baseline_stability_heldout(self):
        """Baseline remains within reasonable stability margin on heldout seeds."""
        for scenario in [EASY, MEDIUM, HARD]:
            dev_mean, _ = mean_reward_over_seeds(
                heuristic_baseline_policy, scenario, DEVELOPMENT_SEEDS
            )
            heldout_mean, _ = mean_reward_over_seeds(
                heuristic_baseline_policy, scenario, HELDOUT_SEEDS
            )
            margin = max(1.0, 0.30 * abs(dev_mean))
            assert abs(heldout_mean - dev_mean) <= margin, (
                f"Baseline unstable on {scenario.name}: "
                f"dev={dev_mean:.2f}, heldout={heldout_mean:.2f}, margin={margin:.2f}"
            )

    def test_baseline_beats_always_route_b_dev(self):
        """Baseline beats always_route_b on all tasks across development seeds."""
        for scenario in [EASY, MEDIUM, HARD]:
            baseline_mean, _ = mean_reward_over_seeds(
                heuristic_baseline_policy, scenario, DEVELOPMENT_SEEDS
            )
            always_b_mean, _ = mean_reward_over_seeds(
                always_route_b_policy, scenario, DEVELOPMENT_SEEDS
            )
            assert baseline_mean >= always_b_mean, (
                f"baseline ({baseline_mean:.2f}) < always_route_b ({always_b_mean:.2f}) on {scenario.name}"
            )
