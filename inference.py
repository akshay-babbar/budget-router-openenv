import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import typer
from openai import OpenAI

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import EASY, HARD, HARD_MULTI, MEDIUM

import re

_VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]


def _parse_llm_action(response_text: str) -> str:
    """Extract a valid action from LLM output. Falls back to shed_load — never raises."""
    text = response_text.strip().lower()
    for action in _VALID_ACTIONS:
        if action in text:
            return action
    return "shed_load"  # safe fallback: always valid, never crashes


SYSTEM_PROMPT = """You are an incident-response routing agent controlling a 
    production service. At each step you observe provider health metrics and must 
    choose exactly one action.

    Valid actions — respond with ONLY one of these four strings, nothing else:
      route_to_a
      route_to_b
      route_to_c
      shed_load

    Decision rules:
    - Prefer providers with windowed success rate above 0.52 and lower cost.
    - If budget_remaining is below 0.20, prefer cheaper providers or shed_load.
    - shed_load reduces backlog but costs -0.5 reward — use it when all providers 
      are degraded, not as a default.
    - Never respond with anything other than one of the four action strings above."""

app = typer.Typer(add_completion=False)

SEED_SETS: Dict[str, List[int]] = {
    "development": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "heldout": [100, 101, 102, 103, 104],
}
TASKS: Dict[str, TaskConfig] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
    "hard_multi": HARD_MULTI,
}
VALID_ACTIONS = [action.value for action in ActionType]


class LLMRouter:
    def __init__(self, api_base_url: str, model_name: str, api_key: str) -> None:
        self._client = OpenAI(base_url=api_base_url, api_key=api_key)
        self._model_name = model_name

    def choose_action(self, observation: Observation) -> Action:
        obs = observation
        obs_text = "\n".join([
            f"provider_a_status: {obs.provider_a_status:.3f}",
            f"provider_b_status: {obs.provider_b_status:.3f}",
            f"provider_c_status: {obs.provider_c_status:.3f}",
            f"budget_remaining:  {obs.budget_remaining:.3f}",
            f"queue_backlog:     {obs.queue_backlog:.3f}",
            f"system_latency:    {obs.system_latency:.3f}",
            f"step_count:        {obs.step_count:.3f}",
        ])
        user_prompt = f"Current observation:\n{obs_text}\n\nYour action:"

        client = self._client
        model_name = self._model_name
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=20,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            action_str = _parse_llm_action(raw)
        except Exception as e:
            import sys
            print(f"[llm_policy] API error — defaulting to shed_load: {e}", file=sys.stderr)
            action_str = "shed_load"
        return Action(action=ActionType(action_str))


def select_policy(policy_name: Literal["heuristic", "llm"]) -> object:
    if policy_name == "heuristic":
        return heuristic_baseline_policy

    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("HF_TOKEN")
    if not api_base_url or not model_name or not api_key:
        raise RuntimeError(
            "LLM policy requires API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables."
        )
    return LLMRouter(api_base_url=api_base_url, model_name=model_name, api_key=api_key)


def run_episode(env: BudgetRouterEnv, scenario: TaskConfig, seed: int, policy_name: Literal["heuristic", "llm"]) -> Dict[str, float]:
    policy = select_policy(policy_name)
    observation = env.reset(seed=seed, scenario=scenario)
    total_reward = 0.0
    while not observation.done:
        if policy_name == "heuristic":
            action = policy(observation)
        else:
            action = policy.choose_action(observation)
        observation = env.step(action)
        total_reward += float(observation.reward or 0.0)
    metrics = episode_metrics(env._internal.history)
    metrics["total_reward"] = round(total_reward, 4)
    metrics["episode_length"] = env._internal.current_step
    # Add grader score (0.0-1.0) for each episode
    grader = grade_episode(env._internal.history)
    metrics["grader_score"] = grader["overall_score"]
    metrics["grader_breakdown"] = grader
    return metrics


def summarize(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    return {
        "mean_reward": round(sum(row["total_reward"] for row in rows) / len(rows), 4),
        "mean_success_rate": round(sum(row["success_rate"] for row in rows) / len(rows), 4),
        "mean_cost": round(sum(row["total_cost_spent"] for row in rows) / len(rows), 4),
        "mean_latency_ms": round(sum(row["average_latency_ms"] for row in rows) / len(rows), 2),
        "mean_grader_score": round(sum(row["grader_score"] for row in rows) / len(rows), 4),
    }


@app.command()
def main(
    policy: Literal["heuristic", "llm"] = typer.Option("heuristic"),
    seed_set: Literal["development", "heldout"] = typer.Option("development"),
    scenario: Literal["all", "easy", "medium", "hard", "hard_multi"] = typer.Option("all"),
    output_path: Path = typer.Option(Path("baseline_results.json")),
) -> None:
    env = BudgetRouterEnv()
    selected_tasks = TASKS if scenario == "all" else {scenario: TASKS[scenario]}
    results: Dict[str, Dict[str, object]] = {}

    for task_name, task_config in selected_tasks.items():
        task_metrics = [
            run_episode(env=env, scenario=task_config, seed=seed, policy_name=policy)
            for seed in SEED_SETS[seed_set]
        ]
        results[task_name] = {
            "policy": policy,
            "seed_set": seed_set,
            "summary": summarize(task_metrics),
            "episodes": task_metrics,
        }

    output_path.write_text(json.dumps(results, indent=2))
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
