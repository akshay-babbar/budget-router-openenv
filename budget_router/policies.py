"""
Policies for the Budget Router environment.

6 policies:
- random_policy: uniform random baseline (lower bound)
- heuristic_baseline_policy: stateless cheapest-viable routing
- debug_upper_bound_policy: oracle with internal state access (test only)
- always_route_a_policy: degenerate (always cheapest)
- always_route_b_policy: degenerate (always balanced fallback)
- always_route_c_policy: degenerate (always most reliable)
- always_shed_load_policy: degenerate (always shed)
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional

from .models import Action, ActionType, InternalState, Observation
from .reward import BUDGET_WEIGHT


def random_policy(obs: Observation, rng: Optional[stdlib_random.Random] = None) -> Action:
    """Uniform random over all 4 actions. No state awareness."""
    r = rng or stdlib_random.Random()
    choice = r.choice(list(ActionType))
    return Action(action_type=choice)


def heuristic_baseline_policy(obs: Observation) -> Action:
    """
    Stateless heuristic: prefer cheapest provider with status > threshold.
    Fallback to next cheapest. shed_load only if ALL below threshold.

    Budget-aware: when budget is critically low, only use the cheapest
    viable provider or shed load to avoid the -10 budget exhaustion penalty.
    No privileged information. Uses only what the agent can observe.
    """
    threshold = 0.52

    # Providers ordered by cost (cheapest first): A, B, C
    providers = [
        ("route_to_a", obs.provider_a_status),
        ("route_to_b", obs.provider_b_status),
        ("route_to_c", obs.provider_c_status),
    ]

    # Budget safety: when critically low, exclude expensive providers
    # to prevent the -10.0 terminal budget exhaustion penalty.
    # Only blocks C ($0.10/req) when budget can't absorb it.
    if obs.budget_remaining < 0.10:
        # Only consider A ($0.01) and B ($0.05) — skip C
        for action_name, status in providers[:2]:
            if status > threshold or status == 0.5:
                return Action(action_type=ActionType(action_name))
        return Action(action_type=ActionType.SHED_LOAD)

    for action_name, status in providers:
        if status > threshold or status == 0.5:
            return Action(action_type=ActionType(action_name))

    # All providers below threshold → shed load
    return Action(action_type=ActionType.SHED_LOAD)


def debug_upper_bound_policy(obs: Observation, internal_state: InternalState) -> Action:
    """
    Oracle policy with access to true internal health values.
    Used ONLY for debugging and validation — NOT a fair benchmark.

    Strategy: expected-value routing using true health, with hard budget
    feasibility constraint. Routes to the cheapest provider whose health
    is high enough, but won't pick an expensive provider if it would
    exhaust the budget.
    """
    initial_budget = internal_state.initial_budget_dollars
    if initial_budget <= 0:
        initial_budget = 1.0

    budget_dollars = internal_state.budget_dollars
    remaining_steps = max(1, internal_state.max_steps - internal_state.current_step)

    providers_info = [
        ("route_to_a", internal_state.providers["A"].current_health,
         internal_state.providers["A"].cost_per_request),
        ("route_to_b", internal_state.providers["B"].current_health,
         internal_state.providers["B"].cost_per_request),
        ("route_to_c", internal_state.providers["C"].current_health,
         internal_state.providers["C"].cost_per_request),
    ]

    best_action = None
    best_ev = float("-inf")

    for action_name, health, cost in providers_info:
        # Hard feasibility: can we afford this provider for remaining steps?
        # If not, skip it entirely to avoid budget exhaustion penalty (-10)
        if cost * remaining_steps > budget_dollars:
            continue

        # Expected per-step reward matching reward.py:
        # P(success) * 1.0 + P(fail) * -2.0 - (cost/initial_budget) * BUDGET_WEIGHT
        ev = health * 1.0 + (1.0 - health) * (-2.0) - (cost / initial_budget) * BUDGET_WEIGHT

        if ev > best_ev:
            best_ev = ev
            best_action = action_name

    if best_action is None:
        # No affordable provider — pick the cheapest one we can still afford once
        for action_name, health, cost in providers_info:
            if cost <= budget_dollars:
                ev = health * 1.0 + (1.0 - health) * (-2.0) - (cost / initial_budget) * BUDGET_WEIGHT
                if ev > best_ev:
                    best_ev = ev
                    best_action = action_name

    if best_action is None or best_ev < -0.5:
        return Action(action_type=ActionType.SHED_LOAD)

    return Action(action_type=ActionType(best_action))


def always_route_a_policy(obs: Observation) -> Action:
    """Degenerate: always route to cheapest provider A."""
    return Action(action_type=ActionType.ROUTE_TO_A)


def always_route_b_policy(obs: Observation) -> Action:
    """Degenerate: always route to balanced provider B."""
    return Action(action_type=ActionType.ROUTE_TO_B)


def always_route_c_policy(obs: Observation) -> Action:
    """Degenerate: always route to most expensive/reliable provider C."""
    return Action(action_type=ActionType.ROUTE_TO_C)


def always_shed_load_policy(obs: Observation) -> Action:
    """Degenerate: always shed load (never routes)."""
    return Action(action_type=ActionType.SHED_LOAD)
