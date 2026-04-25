#!/usr/bin/env python3
"""
Trace one Budget Router episode for a chosen policy, task, and seed.

This is a debugging/evidence tool: it prints per-step actions, step rewards,
costs, success/failure, latency, cumulative reward, and final grader metrics.
It does not expose hidden provider health to the policy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# Ensure imports work when run as `uv run python eval/trace_episode.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import TASK_PRESETS
from inference import LLMRouter


app = typer.Typer(add_completion=False)

POLICIES = {"heuristic", "llm", "ppo"}
DEFAULT_PPO_MODELS = {
    "easy": Path("trained_models/ppo_easy_50k.zip"),
    "hard_multi": Path("trained_models/ppo_hard_multi_100k.zip"),
}

def _visible_observation_row(obs: Observation) -> Dict[str, float]:
    """Public observation values available to the policy before it acts."""
    return {
        "provider_a_status": round(float(obs.provider_a_status), 4),
        "provider_b_status": round(float(obs.provider_b_status), 4),
        "provider_c_status": round(float(obs.provider_c_status), 4),
        "observed_budget_remaining": round(float(obs.budget_remaining), 4),
        "queue_backlog": round(float(obs.queue_backlog), 4),
        "system_latency": round(float(obs.system_latency), 4),
        "step_count": round(float(obs.step_count), 4),
    }


def _visible_observation_row_from_array(values: Any) -> Dict[str, float]:
    """Public observation values from the Gym wrapper's 7-field observation array."""
    return {
        "provider_a_status": round(float(values[0]), 4),
        "provider_b_status": round(float(values[1]), 4),
        "provider_c_status": round(float(values[2]), 4),
        "observed_budget_remaining": round(float(values[3]), 4),
        "queue_backlog": round(float(values[4]), 4),
        "system_latency": round(float(values[5]), 4),
        "step_count": round(float(values[6]), 4),
    }


def _cumulative_step_rows(
    history: List[Dict[str, Any]],
    visible_observations: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    for item in history:
        reward = float(item.get("reward", 0.0) or 0.0)
        cost = float(item.get("cost", 0.0) or 0.0)
        initial_budget = float(item.get("initial_budget", 0.0) or 0.0)
        cumulative_reward += reward
        cumulative_cost += cost
        budget_remaining = max(0.0, initial_budget - cumulative_cost)

        obs_row = visible_observations[len(rows)] if len(rows) < len(visible_observations) else {}
        rows.append({
            "step": int(item.get("step", len(rows) + 1)),
            "action": item.get("action_type"),
            "provider": item.get("provider"),
            "success": bool(item.get("request_succeeded", False)),
            "reward": round(reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "cost": round(cost, 4),
            "budget_remaining": round(budget_remaining, 4),
            "latency_ms": float(item.get("latency_ms", 0.0) or 0.0),
            "queue_overflow": bool(item.get("queue_overflow", False)),
            "budget_exhausted": bool(item.get("budget_exhausted", False)),
            **obs_row,
        })

    return rows


def _run_heuristic(task_cfg: TaskConfig, seed: int) -> tuple[BudgetRouterEnv, List[Dict[str, float]]]:
    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=task_cfg)
    visible_observations = []
    while not obs.done:
        visible_observations.append(_visible_observation_row(obs))
        obs = env.step(heuristic_baseline_policy(obs))
    return env, visible_observations


def _run_llm(task_name: str, task_cfg: TaskConfig, seed: int) -> tuple[BudgetRouterEnv, List[Dict[str, float]]]:
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    if not api_key:
        raise RuntimeError("LLM policy requires HF_TOKEN or API_KEY.")

    policy = LLMRouter(api_base_url=api_base_url, model_name=model_name, api_key=api_key)
    policy.reset(task_name=task_name)

    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=task_cfg)
    visible_observations = []
    while not obs.done:
        visible_observations.append(_visible_observation_row(obs))
        obs = env.step(policy.choose_action(obs))
    return env, visible_observations


def _default_ppo_model_path(task_name: str) -> Path:
    if task_name not in DEFAULT_PPO_MODELS:
        raise ValueError(
            f"No default PPO model for task '{task_name}'. "
            "Pass --model-path explicitly, or use task easy/hard_multi."
        )
    return DEFAULT_PPO_MODELS[task_name]


def _run_ppo(
    task_name: str,
    task_cfg: TaskConfig,
    seed: int,
    model_path: Optional[Path],
) -> tuple[BudgetRouterEnv, List[Dict[str, float]]]:
    # Lazy import keeps heuristic/LLM tracing available without training extras.
    try:
        from stable_baselines3 import PPO
        from train.gym_wrapper import BudgetRouterGymEnv
    except ImportError as exc:
        raise RuntimeError("PPO tracing requires training dependencies: `uv sync --extra training`.") from exc

    resolved_model_path = model_path or _default_ppo_model_path(task_name)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {resolved_model_path}")

    model = PPO.load(str(resolved_model_path))
    gym_env = BudgetRouterGymEnv(scenario=task_cfg, seed=seed)
    obs, _ = gym_env.reset()
    done = False
    visible_observations = []
    while not done:
        visible_observations.append(_visible_observation_row_from_array(obs))
        action_idx, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = gym_env.step(int(action_idx))
        done = terminated or truncated

    return gym_env._env, visible_observations


def trace_episode(
    task_name: str,
    seed: int,
    policy_name: str,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one episode and return step rows plus final scorer outputs."""
    if task_name not in TASK_PRESETS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {sorted(TASK_PRESETS)}")
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_name}'. Choose from: {sorted(POLICIES)}")

    task_cfg = TASK_PRESETS[task_name]
    if policy_name == "heuristic":
        env, visible_observations = _run_heuristic(task_cfg=task_cfg, seed=seed)
    elif policy_name == "llm":
        env, visible_observations = _run_llm(task_name=task_name, task_cfg=task_cfg, seed=seed)
    else:
        env, visible_observations = _run_ppo(
            task_name=task_name,
            task_cfg=task_cfg,
            seed=seed,
            model_path=model_path,
        )

    history = env._internal.history
    steps = _cumulative_step_rows(history, visible_observations)
    grader = {k: round(float(v), 4) for k, v in grade_episode(history).items()}

    return {
        "task": task_name,
        "seed": seed,
        "policy": policy_name,
        "episode_length": len(steps),
        "total_reward": round(sum(row["reward"] for row in steps), 4),
        "grader": grader,
        "metrics": episode_metrics(history),
        "steps": steps,
    }


def _print_trace(result: Dict[str, Any]) -> None:
    typer.echo(f"Task={result['task']}  Policy={result['policy']}  Seed={result['seed']}")
    typer.echo(f"Episode length={result['episode_length']}  Total reward={result['total_reward']:+.4f}")
    typer.echo("Grader:")
    for key, value in result["grader"].items():
        typer.echo(f"  {key}: {value:.4f}")

    typer.echo("")
    typer.echo(
        "Step | A_stat | B_stat | C_stat | Action      | Provider | Success | "
        "Reward  | CumReward | Cost | Budget | Latency | Flags"
    )
    typer.echo(
        "-----|--------|--------|--------|-------------|----------|---------|"
        "---------|-----------|------|--------|---------|------"
    )
    for row in result["steps"]:
        flags = []
        if row["queue_overflow"]:
            flags.append("queue_overflow")
        if row["budget_exhausted"]:
            flags.append("budget_exhausted")
        typer.echo(
            f"{row['step']:>4} | {row.get('provider_a_status', 0.0):>6.3f} | "
            f"{row.get('provider_b_status', 0.0):>6.3f} | "
            f"{row.get('provider_c_status', 0.0):>6.3f} | "
            f"{row['action']:<11} | {str(row['provider'] or '-'):>8} | "
            f"{str(row['success']).lower():>7} | {row['reward']:>+7.2f} | "
            f"{row['cumulative_reward']:>+9.2f} | {row['cost']:>4.2f} | "
            f"{row['budget_remaining']:>6.2f} | {row['latency_ms']:>7.2f} | {','.join(flags) or '-'}"
        )


@app.command()
def main(
    task: str = typer.Option("hard_multi", help=f"Task name: {' | '.join(TASK_PRESETS)}"),
    seed: int = typer.Option(..., help="Exact episode seed."),
    policy: str = typer.Option("heuristic", help=f"Policy: {' | '.join(sorted(POLICIES))}"),
    model_path: Optional[Path] = typer.Option(None, help="PPO model path. Defaults exist for easy/hard_multi."),
    output_json: Optional[Path] = typer.Option(None, help="Optional path to save the full trace JSON."),
) -> None:
    """Run and print a single exact-seed episode trace."""
    result = trace_episode(task_name=task, seed=seed, policy_name=policy, model_path=model_path)
    _print_trace(result)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"\nSaved trace JSON: {output_json}")


if __name__ == "__main__":
    app()
