"""
Reward computation for the Budget Router environment.

Per-step reward (4 additive terms max) and episode-level grader metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


BUDGET_WEIGHT = 5.0  # Scales cost penalty so it's meaningful vs success/failure signal


def step_reward(
    action_type: str,
    request_succeeded: bool,
    provider_cost: float,
    initial_budget: float,
    latency_ms: float,
    sla_ceiling_ms: float,
) -> float:
    """
    Compute single-step reward. Maximum 4 additive terms.

    For shed_load: fixed penalty of -0.5 (replaces routing terms).
    For routing actions:
      +1.0 if request succeeded, -2.0 if failed
      -(provider_cost / initial_budget) * BUDGET_WEIGHT as cost penalty
      -(excess_latency / sla_ceiling_ms) if latency exceeds SLA

    Returns:
        float: The step reward. Never returns NaN.
    """
    # Safety: prevent NaN from division by zero
    if initial_budget <= 0:
        initial_budget = 1.0
    if sla_ceiling_ms <= 0:
        sla_ceiling_ms = 1.0

    # shed_load: flat penalty, no routing terms
    if action_type == "shed_load":
        return -0.5

    reward = 0.0

    # Term 1: Success / failure
    if request_succeeded:
        reward += 1.0
    else:
        reward += -2.0

    # Term 2: Cost penalty (always applied for routing actions)
    cost_penalty = -(provider_cost / initial_budget) * BUDGET_WEIGHT
    reward += cost_penalty

    # Term 3: Latency breach penalty
    if latency_ms > sla_ceiling_ms:
        excess = latency_ms - sla_ceiling_ms
        latency_penalty = -(excess / sla_ceiling_ms)
        reward += latency_penalty

    # Safety: NaN guard
    if math.isnan(reward):
        reward = -2.0

    return reward


def episode_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute deterministic episode-level grader metrics.

    Args:
        history: List of step info dicts from the episode.

    Returns:
        Dict with grader metrics:
        - total_reward
        - success_rate
        - total_cost_spent
        - average_latency_ms
        - sla_met (bool)
        - queue_overflow_events (int)
    """
    if not history:
        return {
            "total_reward": 0.0,
            "success_rate": 0.0,
            "total_cost_spent": 0.0,
            "average_latency_ms": 0.0,
            "sla_met": True,
            "queue_overflow_events": 0,
        }

    total_reward = sum(h.get("reward", 0.0) for h in history)

    # Only count routing steps (not shed_load) for success rate
    routing_steps = [h for h in history if h.get("action_type") != "shed_load"]
    if routing_steps:
        successes = sum(1 for h in routing_steps if h.get("request_succeeded", False))
        success_rate = successes / len(routing_steps)
    else:
        success_rate = 0.0

    total_cost = sum(h.get("cost", 0.0) for h in history)

    latencies = [h.get("latency_ms", 0.0) for h in routing_steps]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    sla_ceiling = history[0].get("sla_ceiling_ms", 500.0)
    sla_met = all(lat <= sla_ceiling for lat in latencies) if latencies else True

    queue_overflows = sum(1 for h in history if h.get("queue_overflow", False))

    return {
        "total_reward": round(total_reward, 4),
        "success_rate": round(success_rate, 4),
        "total_cost_spent": round(total_cost, 4),
        "average_latency_ms": round(avg_latency, 2),
        "sla_met": sla_met,
        "queue_overflow_events": queue_overflows,
    }


def grade_episode(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute episode-level grader score in [0, 1] with weighted breakdown.

    overall = 0.30 × success_score
            + 0.20 × latency_score
            + 0.15 × budget_score
            + 0.15 × sla_score
            + 0.20 × adaptation_score

    Component definitions:
        success_score: Fraction of ALL episode steps with a successful routed request.
            Denominator = total steps (not routed steps), so partial abstention is penalised.
        latency_score: 1.0 - (avg_latency / sla_ceiling), clamped to [0, 1].
        budget_score:  Fraction of initial budget NOT spent, clamped to [0, 1].
        sla_score:     Fraction of routed requests with latency <= sla_ceiling.
        adaptation_score: Post-degradation success rate — measures whether the
            agent detected and adapted to provider degradation.

    Adaptation score window semantics by task:
        - easy (no degradation):  No post-degradation window exists.
            adaptation_score = 1.0 (adaptation not required → full marks).
        - medium (A degrades after step 5): Window = routing steps with
            step > 5. Measures success rate after A begins failing.
        - hard (A degrades from step 0): Window = routing steps with
            step > 1 (one warm-up step allowed). Covers nearly all steps.
        - hard_multi (A from step 0, B from step 10): Blended score:
            0.5 × primary_adaptation (steps between primary and secondary)
            + 0.5 × secondary_adaptation (steps after secondary event).

    All component scores are clamped to [0.0, 1.0].

    Args:
        history: List of step info dicts from the episode.

    Returns:
        Dict with 'overall_score' and per-component breakdown.
    """
    # Note: step_reward() is shaped for learning signal (dense + budget cliff).
    # grade_episode() is the semantic evaluation metric. Divergence is intentional.
    if not history:
        return {
            "overall_score": 0.0,
            "success_score": 0.0,
            "latency_score": 0.0,
            "budget_score": 0.0,
            "sla_score": 0.0,
            "adaptation_score": 0.0,
        }

    metrics = episode_metrics(history)

    # success_score: fraction of ALL episode steps that resulted in a successful routed request.
    # Denominator is total steps, not routed steps, so partial abstention is penalised.
    # A policy that serves 10/20 and succeeds each time scores 0.50, not 1.0.
    total_steps = len(history)
    routing_steps = [h for h in history if h.get("action_type") != "shed_load"]
    routed_successes = sum(1 for h in routing_steps if h.get("request_succeeded", False))
    success_score = routed_successes / total_steps if total_steps > 0 else 0.0

    sla_ceiling_ms = float(history[0].get("sla_ceiling_ms", 500.0) or 500.0)
    avg_latency_ms = float(metrics.get("average_latency_ms", 0.0))

    if sla_ceiling_ms <= 0:
        sla_ceiling_ms = 1.0


    # Fix 1: No routing attempts = no service delivered. Quality scores must reflect this.
    if routing_steps:
        latency_score = 1.0 - min(1.0, avg_latency_ms / sla_ceiling_ms)
        sla_ok = sum(1 for h in routing_steps if float(h.get("latency_ms", 0.0)) <= sla_ceiling_ms)
        sla_score = sla_ok / len(routing_steps)
    else:
        latency_score = 0.0
        sla_score = 0.0

    # Budget score: penalize spending relative to initial budget, not theoretical max
    total_cost = float(metrics.get("total_cost_spent", 0.0))
    initial_budget = float(history[0].get("initial_budget", 1.0) or 1.0)
    budget_score = max(0.0, 1.0 - total_cost / initial_budget)

    # Adaptation score: measures post-degradation success rate.
    # Directly measures whether the agent detected and adapted to degradation.
    adaptation_score = 0.0
    _raw_degrade = history[0].get("degradation_start_step")
    degradation_start = int(_raw_degrade) if _raw_degrade is not None else 999
    _raw_secondary = history[0].get("secondary_degradation_start_step")
    secondary_start = int(_raw_secondary) if _raw_secondary is not None else None

    if degradation_start < 999:
        if secondary_start is not None:
            # Fix 2: hard_multi — blended adaptation across primary and secondary windows
            primary_window = [h for h in routing_steps
                              if int(h.get("step", 0)) > max(degradation_start, 1)
                              and int(h.get("step", 0)) <= secondary_start]
            secondary_window = [h for h in routing_steps
                                if int(h.get("step", 0)) > secondary_start]

            if primary_window:
                primary_adaptation = sum(1 for h in primary_window if h.get("request_succeeded", False)) / len(primary_window)
            else:
                primary_adaptation = 0.0

            if secondary_window:
                secondary_adaptation = sum(1 for h in secondary_window if h.get("request_succeeded", False)) / len(secondary_window)
            else:
                secondary_adaptation = 0.0

            if not primary_window and not secondary_window:
                adaptation_score = 0.0
            else:
                adaptation_score = 0.5 * primary_adaptation + 0.5 * secondary_adaptation
        else:
            # Single degradation event: existing logic unchanged
            # Use max(degradation_start, 1) to ensure at least one warm-up step
            # before post-degradation tracking, even when degradation_start=0
            post_degrade = [h for h in routing_steps
                            if int(h.get("step", 0)) > max(degradation_start, 1)]
            if post_degrade:
                post_successes = sum(1 for h in post_degrade if h.get("request_succeeded", False))
                adaptation_score = post_successes / len(post_degrade)
    else:
        # No degradation event. Award adaptation based on routing quality instead.
        # A do-nothing (always shed_load) policy gets 0, not 1.0.
        if routing_steps:
            quality_successes = sum(1 for h in routing_steps if h.get("request_succeeded", False))
            adaptation_score = quality_successes / total_steps  # total_steps denominator penalizes abstention
        else:
            adaptation_score = 0.0

    overall = (
        0.3 * success_score
        + 0.2 * latency_score
        + 0.15 * budget_score
        + 0.15 * sla_score
        + 0.2 * adaptation_score
    )

    # Hard penalty for budget exhaustion: incomplete episodes are not reliable systems.
    # A policy that routes aggressively and goes bankrupt at step 17 should not outscore
    # one that completes all 20 steps. 0.75x preserves partial credit for good routing
    # before exhaustion, but makes budget-exhausted policies non-competitive.
    episode_terminated_early = any(h.get('budget_exhausted', False) for h in history)
    if episode_terminated_early:
        overall = overall * 0.75

    overall = max(0.0, min(1.0, overall))
    return {
        "overall_score": round(overall, 4),
        "success_score": round(max(0.0, min(1.0, success_score)), 4),
        "latency_score": round(max(0.0, min(1.0, latency_score)), 4),
        "budget_score": round(max(0.0, min(1.0, budget_score)), 4),
        "sla_score": round(max(0.0, min(1.0, sla_score)), 4),
        "adaptation_score": round(max(0.0, min(1.0, adaptation_score)), 4),
    }
