"""
Reward computation for the Budget Router environment.

Per-step reward (4 additive terms max) and episode-level grader metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


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
      -(provider_cost / initial_budget) as cost penalty
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
    cost_penalty = -(provider_cost / initial_budget)
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
