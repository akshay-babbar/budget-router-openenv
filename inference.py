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

    _SYSTEM_PROMPT = """You are an incident commander managing a real-time API routing system.
You must route each incoming request to one of three providers or shed load.

PROVIDERS (fixed costs and baseline reliability):
- Provider A: $0.01/request, baseline reliability ~85%, fastest latency (~100ms). MAY DEGRADE over time.
- Provider B: $0.05/request, baseline reliability ~92%, medium latency (~150ms). MAY DEGRADE in hard scenarios.
- Provider C: $0.10/request, baseline reliability ~99%, slower latency (~200ms). Stable but expensive.

OBSERVATION (all values normalized 0.0-1.0):
- provider_a_status: recent windowed success rate for Provider A (low = degraded/failing)
- provider_b_status: recent windowed success rate for Provider B
- provider_c_status: recent windowed success rate for Provider C
- budget_remaining: fraction of episode budget left (0.0 = exhausted, causes episode end with -10 penalty)
- queue_backlog: request queue pressure (high = latency amplification cascade risk)
- system_latency: last latency relative to 500ms SLA ceiling (above 1.0 = SLA breach)
- step_count: episode progress (0.0 = start, 1.0 = end)

GOAL: Maximize cumulative reward over 20 steps.
- Success: +1.0 per routed request that succeeds
- Failure: -2.0 per routed request that fails
- Cost penalty: proportional to provider cost (A cheapest, C most expensive)
- SLA breach penalty: if latency exceeds 500ms ceiling
- Budget exhaustion: -10.0 terminal penalty if budget hits zero

STRATEGY HINTS:
- If a provider's status drops below ~0.5, switch away from it — it is degrading.
- Provider A is cheapest but may degrade early; detect this from its status reading.
- shed_load (-0.5 flat) is better than routing to a failing provider (-2.0 + cost).
- Watch budget_remaining — if it's critically low (<0.15), avoid expensive providers.
- The optimal strategy adapts routing as providers degrade; never locks onto one provider.

Respond with EXACTLY one of: route_to_a, route_to_b, route_to_c, shed_load"""

    def choose_action(self, observation: Observation) -> Action:
        prompt = self._build_prompt(observation)
        completion = self._client.chat.completions.create(
            model=self._model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = (completion.choices[0].message.content or "").strip().lower()
        for action_name in VALID_ACTIONS:
            if action_name in content:
                return Action(action=ActionType(action_name))
        return heuristic_baseline_policy(observation)

    def _build_prompt(self, observation: Observation) -> str:
        # Include both raw normalized values AND interpreted context
        context = {
            "step": round(observation.step_count * 20),
            "budget_remaining_fraction": round(observation.budget_remaining, 3),
            "budget_critically_low": observation.budget_remaining < 0.15,
            "provider_a_status": round(observation.provider_a_status, 3),
            "provider_a_healthy": observation.provider_a_status > 0.6,
            "provider_b_status": round(observation.provider_b_status, 3),
            "provider_b_healthy": observation.provider_b_status > 0.6,
            "provider_c_status": round(observation.provider_c_status, 3),
            "queue_backlog_fraction": round(observation.queue_backlog, 3),
            "latency_fraction_of_sla": round(observation.system_latency, 3),
            "sla_at_risk": observation.system_latency > 0.7,
            "valid_actions": VALID_ACTIONS,
            "action_costs": {
                "route_to_a": "$0.01",
                "route_to_b": "$0.05",
                "route_to_c": "$0.10",
                "shed_load": "$0.00 (flat -0.5 penalty)",
            },
        }
        return (
            "Current environment state:\n"
            + json.dumps(context, indent=2)
            + "\n\nChoose the best action. Reply with exactly one action name."
        )


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
