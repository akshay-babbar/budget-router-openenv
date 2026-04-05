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

    def test_step_before_reset_no_crash(self):
        """step() before reset() auto-initializes so the default OpenEnv web UI is safe."""
        env = BudgetRouterEnv()
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


# ─── Grader Semantic Tests ──────────────────────────────────────────────


class TestGraderSemantics:
    """Pin the exact grader semantics changed by the abstention and hard_multi fixes.

    These tests defend against regressions to grade_episode() — the most
    judge-sensitive function in the repo.
    """

    def _make_step(self, step, action, succeeded, cost, latency, degrade=999, secondary=None):
        return {
            "step": step, "action_type": action,
            "request_succeeded": succeeded, "cost": cost,
            "latency_ms": latency, "reward": 0.9,
            "sla_ceiling_ms": 500.0, "initial_budget": 1.0,
            "degradation_start_step": degrade,
            "secondary_degradation_start_step": secondary,
        }

    def test_pure_abstention_scores_below_0_40_on_easy(self):
        """A policy that sheds all load must score < 0.40 overall on easy.

        Before the fix this scored ~0.70 (sla=1.0, latency=1.0 on empty routing set).
        """
        from budget_router.reward import grade_episode

        history = [
            self._make_step(i, "shed_load", False, 0.0, 0.0, degrade=999)
            for i in range(1, 21)
        ]
        result = grade_episode(history)

        assert result["overall_score"] < 0.40, (
            f"Pure abstention scored {result['overall_score']} >= 0.40 on easy "
            f"(grader exploit not fixed)"
        )
        assert result["sla_score"] == 0.0, "sla_score should be 0.0 when no requests routed"
        assert result["latency_score"] == 0.0, "latency_score should be 0.0 when no requests routed"
        assert result["success_score"] == 0.0, "success_score should be 0.0 when no requests routed"
        assert result["budget_score"] == 1.0, "budget_score should be 1.0 when nothing spent"
        assert result["adaptation_score"] == 1.0, "adaptation_score should be 1.0 on easy (no degradation)"

    def test_partial_abstention_scores_less_than_full_service(self):
        """A policy that sheds 50% of load must score < a policy that serves all 20 steps.

        Before the success_score denominator fix, partial abstention could outscore
        full service because budget_score rewarded not spending.
        """
        from budget_router.reward import grade_episode

        # Mixed: 10 sheds then 10 successful routes
        mixed = (
            [self._make_step(i, "shed_load", False, 0.0, 0.0) for i in range(1, 11)]
            + [self._make_step(i, "route_to_a", True, 0.01, 110.0) for i in range(11, 21)]
        )
        # Full service: 20 successful routes
        full = [self._make_step(i, "route_to_a", True, 0.01, 110.0) for i in range(1, 21)]

        r_mixed = grade_episode(mixed)
        r_full = grade_episode(full)

        assert r_mixed["overall_score"] < r_full["overall_score"], (
            f"Partial abstention ({r_mixed['overall_score']}) >= full service "
            f"({r_full['overall_score']}) — grader still rewards low-throughput"
        )
        assert r_mixed["success_score"] < r_full["success_score"], (
            f"success_score should be lower for 10/20 served ({r_mixed['success_score']}) "
            f"than 20/20 served ({r_full['success_score']})"
        )

    def test_hard_multi_adaptation_uses_secondary_window(self):
        """grade_episode computes blended adaptation for hard_multi (secondary window included).

        Verifies that secondary_degradation_start_step=10 in step_info causes
        grade_episode to split the adaptation window at step 10 and blend 0.5/0.5.
        """
        from budget_router.reward import grade_episode

        # Build a hard_multi episode: steps 1-10 primary window (route A, succeeds),
        # steps 11-20 secondary window (route A, fails — B degraded, agent stuck)
        history = []
        for i in range(1, 11):
            history.append(self._make_step(i, "route_to_a", True, 0.01, 110.0, degrade=0, secondary=10))
        for i in range(11, 21):
            history.append(self._make_step(i, "route_to_a", False, 0.01, 700.0, degrade=0, secondary=10))

        result = grade_episode(history)

        # primary_window: steps > max(0,1)=1 and <= 10 → steps 2..10 → 9 steps, all succeed → 1.0
        # secondary_window: steps > 10 → steps 11..20 → 10 steps, all fail → 0.0
        # blended = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        expected_adaptation = 0.5
        assert abs(result["adaptation_score"] - expected_adaptation) < 0.01, (
            f"hard_multi blended adaptation expected ~{expected_adaptation}, "
            f"got {result['adaptation_score']}"
        )

        # Compare with an equivalent hard (non-multi) episode to confirm they diverge
        history_hard = []
        for i in range(1, 11):
            history_hard.append(self._make_step(i, "route_to_a", True, 0.01, 110.0, degrade=0, secondary=None))
        for i in range(11, 21):
            history_hard.append(self._make_step(i, "route_to_a", False, 0.01, 700.0, degrade=0, secondary=None))

        result_hard = grade_episode(history_hard)
        # hard (no secondary): post_degrade = steps > max(0,1)=1 → steps 2..20 → 19 steps
        # 9 succeed (steps 2-10), 10 fail (steps 11-20) → 9/19 ≈ 0.473
        assert result["adaptation_score"] != result_hard["adaptation_score"], (
            f"hard_multi and hard got identical adaptation_score={result['adaptation_score']} "
            f"— secondary window not being used"
        )
