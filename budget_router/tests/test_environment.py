"""
Tests for the Budget Router environment core correctness and reward sanity.

All tests from <test_requirements> are implemented here.
"""

import math
import random

import pytest

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation
from budget_router.policies import (
    always_route_a_policy,
    always_route_b_policy,
    always_route_c_policy,
    always_shed_load_policy,
    heuristic_baseline_policy,
    random_policy,
)
from budget_router.reward import step_reward
from budget_router.tasks import EASY, HARD, MEDIUM


# ─── Helpers ────────────────────────────────────────────────────────────


def run_full_episode(env, policy_fn, seed, scenario, policy_name=""):
    """Run a full episode and return (observations, rewards, done_flag, steps)."""
    obs = env.reset(seed=seed, scenario=scenario)
    observations = [obs]
    rewards = []
    steps = 0
    rng = random.Random(seed + 10000) if "random" in policy_name else None

    while not obs.done and steps < scenario.max_steps:
        if "random" in policy_name:
            action = policy_fn(obs, rng=rng)
        else:
            action = policy_fn(obs)
        obs = env.step(action)
        observations.append(obs)
        rewards.append(obs.reward)
        steps += 1

    return observations, rewards, obs.done, steps


# ─── Core Correctness Tests ────────────────────────────────────────────


class TestCoreCorrectness:
    """Core environment correctness tests."""

    def test_reset_returns_valid_observation(self):
        """reset() returns Observation with ALL values in [0.0, 1.0]."""
        env = BudgetRouterEnv()
        obs = env.reset(seed=42, scenario=EASY)

        assert isinstance(obs, Observation)
        assert 0.0 <= obs.provider_a_status <= 1.0
        assert 0.0 <= obs.provider_b_status <= 1.0
        assert 0.0 <= obs.provider_c_status <= 1.0
        assert 0.0 <= obs.budget_remaining <= 1.0
        assert 0.0 <= obs.queue_backlog <= 1.0
        assert 0.0 <= obs.system_latency <= 1.0
        assert 0.0 <= obs.step_count <= 1.0

    def test_step_after_reset_no_crash(self):
        """step() after reset() does not crash and returns valid types."""
        env = BudgetRouterEnv()
        obs = env.reset(seed=42, scenario=EASY)
        action = Action(action_type=ActionType.ROUTE_TO_A)
        obs = env.step(action)

        assert isinstance(obs, Observation)
        assert isinstance(obs.done, bool)
        assert isinstance(obs.reward, (int, float))

    def test_episode_terminates_at_or_before_20(self):
        """Episode terminates at or before step 20."""
        env = BudgetRouterEnv()
        for scenario in [EASY, MEDIUM, HARD]:
            obs = env.reset(seed=42, scenario=scenario)
            steps = 0
            while not obs.done and steps < 25:  # give extra margin to catch bugs
                action = Action(action_type=ActionType.ROUTE_TO_B)
                obs = env.step(action)
                steps += 1
            assert steps <= 20, f"Episode ran {steps} steps on {scenario.name}"

    def test_deterministic_trajectories_same_seed(self):
        """Two reset() calls with same seed produce identical full trajectories."""
        env = BudgetRouterEnv()

        # Run 1
        obs1_list, rewards1, _, _ = run_full_episode(
            env, heuristic_baseline_policy, seed=42, scenario=MEDIUM
        )

        # Run 2
        obs2_list, rewards2, _, _ = run_full_episode(
            env, heuristic_baseline_policy, seed=42, scenario=MEDIUM
        )

        assert len(rewards1) == len(rewards2)
        for r1, r2 in zip(rewards1, rewards2):
            assert r1 == r2, f"Rewards differ: {r1} vs {r2}"

    def test_budget_remaining_never_nan(self):
        """budget_remaining never returns NaN."""
        env = BudgetRouterEnv()
        for scenario in [EASY, MEDIUM, HARD]:
            observations, _, _, _ = run_full_episode(
                env, heuristic_baseline_policy, seed=42, scenario=scenario
            )
            for obs in observations:
                assert not math.isnan(obs.budget_remaining), "budget_remaining is NaN"

    def test_provider_status_in_bounds(self):
        """All provider_status values stay in [0.0, 1.0] throughout episode."""
        env = BudgetRouterEnv()
        for scenario in [EASY, MEDIUM, HARD]:
            observations, _, _, _ = run_full_episode(
                env, heuristic_baseline_policy, seed=0, scenario=scenario
            )
            for obs in observations:
                assert 0.0 <= obs.provider_a_status <= 1.0
                assert 0.0 <= obs.provider_b_status <= 1.0
                assert 0.0 <= obs.provider_c_status <= 1.0

    def test_system_latency_not_always_zero(self):
        """system_latency is NOT always 0.0 across a full episode (dead channel guard)."""
        env = BudgetRouterEnv()
        observations, _, _, _ = run_full_episode(
            env, heuristic_baseline_policy, seed=42, scenario=MEDIUM
        )
        # Skip first observation (from reset) — latency may be initial value
        latencies = [obs.system_latency for obs in observations[1:]]
        assert any(lat > 0.0 for lat in latencies), "system_latency is always 0.0 — dead channel"

    def test_all_observation_fields_in_range(self):
        """All Observation fields remain within [0.0, 1.0] at every step."""
        env = BudgetRouterEnv()
        for scenario in [EASY, MEDIUM, HARD]:
            for seed in [0, 1, 2]:
                observations, _, _, _ = run_full_episode(
                    env, heuristic_baseline_policy, seed=seed, scenario=scenario
                )
                for obs in observations:
                    assert 0.0 <= obs.provider_a_status <= 1.0
                    assert 0.0 <= obs.provider_b_status <= 1.0
                    assert 0.0 <= obs.provider_c_status <= 1.0
                    assert 0.0 <= obs.budget_remaining <= 1.0
                    assert 0.0 <= obs.queue_backlog <= 1.0
                    assert 0.0 <= obs.system_latency <= 1.0
                    assert 0.0 <= obs.step_count <= 1.0


# ─── Reward Sanity Tests ───────────────────────────────────────────────


class TestRewardSanity:
    """Reward correctness tests."""

    def test_shed_load_reward_less_than_successful_route_c(self):
        """shed_load reward < successful route_to_c reward (holding all else equal)."""
        shed_r = step_reward("shed_load", False, 0.0, 5.0, 0.0, 500.0)
        route_c_r = step_reward("route_to_c", True, 0.10, 5.0, 200.0, 500.0)
        assert shed_r < route_c_r, f"shed ({shed_r}) >= route_c success ({route_c_r})"

    def test_failed_route_less_than_successful_route(self):
        """Failed route reward < successful route reward."""
        failed_r = step_reward("route_to_a", False, 0.01, 5.0, 300.0, 500.0)
        success_r = step_reward("route_to_a", True, 0.01, 5.0, 100.0, 500.0)
        assert failed_r < success_r, f"failed ({failed_r}) >= success ({success_r})"

    def test_route_a_cost_less_than_route_c_cost(self):
        """route_to_a cost < route_to_c cost in info dict."""
        env = BudgetRouterEnv()
        env.reset(seed=42, scenario=EASY)

        obs_a = env.step(Action(action_type=ActionType.ROUTE_TO_A))
        cost_a = obs_a.metadata.get("cost", 0)

        env.reset(seed=42, scenario=EASY)
        obs_c = env.step(Action(action_type=ActionType.ROUTE_TO_C))
        cost_c = obs_c.metadata.get("cost", 0)

        assert cost_a < cost_c, f"cost_a ({cost_a}) >= cost_c ({cost_c})"

    def test_route_a_under_hard_degradation_lower_cumulative(self):
        """route_to_a under hard degradation gets lower cumulative reward than route_to_c."""
        env = BudgetRouterEnv()
        seeds = [0, 1, 2, 3, 4]

        total_a = 0.0
        total_c = 0.0

        for seed in seeds:
            _, rewards_a, _, _ = run_full_episode(
                env, always_route_a_policy, seed=seed, scenario=HARD
            )
            total_a += sum(r or 0 for r in rewards_a)

            _, rewards_c, _, _ = run_full_episode(
                env, always_route_c_policy, seed=seed, scenario=HARD
            )
            total_c += sum(r or 0 for r in rewards_c)

        assert total_a < total_c, (
            f"always_route_a ({total_a:.2f}) >= always_route_c ({total_c:.2f}) on HARD"
        )


# ─── Degenerate Policy Sanity ──────────────────────────────────────────


class TestDegeneratePolicySanity:
    """Degenerate policy tests."""

    def test_always_route_a_does_not_dominate_baseline_medium(self):
        """always_route_a does not dominate heuristic baseline on medium across dev seeds."""
        env = BudgetRouterEnv()
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        baseline_rewards = []
        always_a_rewards = []

        for seed in seeds:
            _, rewards, _, _ = run_full_episode(
                env, heuristic_baseline_policy, seed=seed, scenario=MEDIUM
            )
            baseline_rewards.append(sum(r or 0 for r in rewards))

            _, rewards, _, _ = run_full_episode(
                env, always_route_a_policy, seed=seed, scenario=MEDIUM
            )
            always_a_rewards.append(sum(r or 0 for r in rewards))

        baseline_mean = sum(baseline_rewards) / len(baseline_rewards)
        always_a_mean = sum(always_a_rewards) / len(always_a_rewards)

        assert baseline_mean >= always_a_mean, (
            f"always_route_a ({always_a_mean:.2f}) dominates baseline ({baseline_mean:.2f}) on medium"
        )

    def test_always_route_c_does_not_dominate_baseline_overall(self):
        """always_route_c does not dominate heuristic baseline across all tasks."""
        env = BudgetRouterEnv()
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        baseline_total = 0.0
        always_c_total = 0.0

        for scenario in [EASY, MEDIUM, HARD]:
            for seed in seeds:
                _, rewards, _, _ = run_full_episode(
                    env, heuristic_baseline_policy, seed=seed, scenario=scenario
                )
                baseline_total += sum(r or 0 for r in rewards)

                _, rewards, _, _ = run_full_episode(
                    env, always_route_c_policy, seed=seed, scenario=scenario
                )
                always_c_total += sum(r or 0 for r in rewards)

        assert baseline_total >= always_c_total, (
            f"always_route_c ({always_c_total:.2f}) dominates baseline ({baseline_total:.2f}) overall"
        )

    def test_always_route_b_does_not_dominate_baseline_medium(self):
        """always_route_b does not dominate heuristic baseline on medium across dev seeds."""
        env = BudgetRouterEnv()
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        baseline_rewards = []
        always_b_rewards = []

        for seed in seeds:
            _, rewards, _, _ = run_full_episode(
                env, heuristic_baseline_policy, seed=seed, scenario=MEDIUM
            )
            baseline_rewards.append(sum(r or 0 for r in rewards))

            _, rewards, _, _ = run_full_episode(
                env, always_route_b_policy, seed=seed, scenario=MEDIUM
            )
            always_b_rewards.append(sum(r or 0 for r in rewards))

        baseline_mean = sum(baseline_rewards) / len(baseline_rewards)
        always_b_mean = sum(always_b_rewards) / len(always_b_rewards)

        assert baseline_mean >= always_b_mean, (
            f"always_route_b ({always_b_mean:.2f}) dominates baseline ({baseline_mean:.2f}) on medium"
        )

    def test_always_shed_load_worse_than_baseline_easy(self):
        """always_shed_load performs materially worse than heuristic baseline on easy."""
        env = BudgetRouterEnv()
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        baseline_rewards = []
        always_shed_rewards = []

        for seed in seeds:
            _, rewards, _, _ = run_full_episode(
                env, heuristic_baseline_policy, seed=seed, scenario=EASY
            )
            baseline_rewards.append(sum(r or 0 for r in rewards))

            _, rewards, _, _ = run_full_episode(
                env, always_shed_load_policy, seed=seed, scenario=EASY
            )
            always_shed_rewards.append(sum(r or 0 for r in rewards))

        baseline_mean = sum(baseline_rewards) / len(baseline_rewards)
        shed_mean = sum(always_shed_rewards) / len(always_shed_rewards)

        assert baseline_mean > shed_mean, (
            f"always_shed ({shed_mean:.2f}) >= baseline ({baseline_mean:.2f}) on easy"
        )
