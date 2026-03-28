"""
Episode visualization for Budget Router environment.

Generates a 4-panel matplotlib figure showing:
  1. Provider health degradation over time
  2. Budget remaining curve
  3. Action distribution per step (color-coded strip)
  4. Cumulative reward trajectory

Usage:
    python visualize.py --scenario hard_multi --seed 42
    python visualize.py --scenario medium --policy oracle
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.policies import (
    debug_upper_bound_policy,
    heuristic_baseline_policy,
    random_policy,
)
from budget_router.tasks import TASK_PRESETS

ACTION_ORDER = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]
ACTION_LABELS = ["Route A", "Route B", "Route C", "Shed Load"]
ACTION_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]


def run_and_trace(
    env: BudgetRouterEnv,
    policy_fn: Any,
    seed: int,
    scenario_name: str,
    policy_name: str = "heuristic_baseline",
) -> Dict[str, List]:
    """Run an episode and collect per-step trace data for visualization."""
    scenario = TASK_PRESETS[scenario_name]
    obs = env.reset(seed=seed, scenario=scenario)
    rng = random.Random(seed + 10000) if "random" in policy_name else None

    trace: Dict[str, List] = {
        "step": [],
        "a_health": [],
        "b_health": [],
        "c_health": [],
        "budget": [],
        "budget_pct": [],
        "reward": [],
        "cumulative_reward": [],
        "action": [],
        "latency_ms": [],
        "queue_backlog": [],
    }

    cumulative = 0.0
    steps = 0
    initial_budget = scenario.initial_budget

    while not obs.done and steps < scenario.max_steps:
        if "upper_bound" in policy_name:
            action = policy_fn(obs, env._internal)
        elif "random" in policy_name:
            action = policy_fn(obs, rng=rng)
        else:
            action = policy_fn(obs)

        obs = env.step(action)
        steps += 1

        reward = obs.reward or 0.0
        cumulative += reward
        s = env._internal

        trace["step"].append(steps)
        trace["a_health"].append(s.providers["A"].current_health)
        trace["b_health"].append(s.providers["B"].current_health)
        trace["c_health"].append(s.providers["C"].current_health)
        trace["budget"].append(s.budget_dollars)
        trace["budget_pct"].append(s.budget_dollars / initial_budget if initial_budget > 0 else 0)
        trace["reward"].append(reward)
        trace["cumulative_reward"].append(cumulative)
        trace["action"].append(action.action.value)
        trace["latency_ms"].append(s.last_latency_ms)
        trace["queue_backlog"].append(s.queue_backlog_count)

    return trace


def render_episode(trace: Dict[str, List], scenario_name: str, policy_name: str, seed: int) -> plt.Figure:
    """Create a 4-panel matplotlib figure from episode trace data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Budget Router Episode — {scenario_name.upper()} / {policy_name.replace('_', ' ').title()} / seed={seed}",
        fontsize=14,
        fontweight="bold",
    )

    steps = trace["step"]

    # ── Panel 1: Provider health degradation ──
    ax1 = axes[0, 0]
    ax1.plot(steps, trace["a_health"], "o-", color="#e74c3c", label="Provider A", linewidth=2, markersize=4)
    ax1.plot(steps, trace["b_health"], "s-", color="#3498db", label="Provider B", linewidth=2, markersize=4)
    ax1.plot(steps, trace["c_health"], "^-", color="#2ecc71", label="Provider C", linewidth=2, markersize=4)
    ax1.axhline(y=0.52, color="gray", linestyle="--", alpha=0.5, label="Heuristic threshold (0.52)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Health (true)")
    ax1.set_title("Provider Health Degradation")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Budget remaining ──
    ax2 = axes[0, 1]
    ax2.plot(steps, trace["budget"], "o-", color="#f39c12", linewidth=2, markersize=4)
    ax2.fill_between(steps, trace["budget"], alpha=0.2, color="#f39c12")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Budget exhausted")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Budget Remaining ($)")
    ax2.set_title("Budget Remaining")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Action distribution (color-coded strip) ──
    ax3 = axes[1, 0]
    action_map = {name: i for i, name in enumerate(ACTION_ORDER)}
    for i, (action_val, step_val) in enumerate(zip(trace["action"], steps)):
        idx = action_map.get(action_val, 3)
        ax3.barh(idx, 0.8, left=step_val - 0.4, height=0.6, color=ACTION_COLORS[idx], edgecolor="white", linewidth=0.5)

    ax3.set_yticks(range(len(ACTION_LABELS)))
    ax3.set_yticklabels(ACTION_LABELS)
    ax3.set_xlabel("Step")
    ax3.set_title("Action per Step")
    ax3.set_xlim(0.5, max(steps) + 0.5 if steps else 20.5)
    ax3.grid(True, alpha=0.3, axis="x")

    # Add legend patches manually
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=ACTION_COLORS[i], label=ACTION_LABELS[i]) for i in range(len(ACTION_LABELS))]
    ax3.legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=2)

    # ── Panel 4: Cumulative reward ──
    ax4 = axes[1, 1]
    ax4.plot(steps, trace["cumulative_reward"], "o-", color="#9b59b6", linewidth=2, markersize=4)
    ax4.fill_between(steps, trace["cumulative_reward"], alpha=0.1, color="#9b59b6")
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_title(f"Cumulative Reward (total: {trace['cumulative_reward'][-1]:.2f})")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Budget Router episode")
    parser.add_argument("--scenario", default="hard_multi", choices=list(TASK_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        default="heuristic_baseline",
        choices=["heuristic_baseline", "oracle", "random"],
    )
    parser.add_argument("--output", default=None, help="Output file path (default: docs/{scenario}_{policy}.png)")
    args = parser.parse_args()

    policies = {
        "heuristic_baseline": (heuristic_baseline_policy, "heuristic_baseline"),
        "oracle": (debug_upper_bound_policy, "upper_bound"),
        "random": (random_policy, "random"),
    }

    policy_fn, policy_name = policies[args.policy]
    env = BudgetRouterEnv()

    print(f"Running episode: {args.scenario} / {args.policy} / seed={args.seed}")
    trace = run_and_trace(env, policy_fn, args.seed, args.scenario, policy_name=policy_name)
    print(f"Episode complete: {len(trace['step'])} steps, total reward={trace['cumulative_reward'][-1]:.2f}")

    fig = render_episode(trace, args.scenario, args.policy, args.seed)

    output = args.output or f"docs/{args.scenario}_{args.policy}_seed{args.seed}.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
