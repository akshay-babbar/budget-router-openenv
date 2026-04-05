import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal

import typer
from openai import OpenAI

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import EASY, HARD, HARD_MULTI, MEDIUM

_VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]


def _parse_llm_action(response_text: str) -> str:
    """Extract a valid action from LLM output. Falls back to shed_load — never raises."""
    text = response_text.strip().lower()
    for action in _VALID_ACTIONS:
        if action in text:
            return action
    return "shed_load"  # safe fallback: always valid, never crashes


SYSTEM_PROMPT = """You are a cost-aware LLM API routing agent managing a production system.
At each step, output EXACTLY ONE action string. Nothing else.

PROVIDERS AND COSTS:
  provider_a_status: health of Provider A — costs $0.01/request (CHEAPEST)
  provider_b_status: health of Provider B — costs $0.05/request
  provider_c_status: health of Provider C — costs $0.10/request (10x cost of A)
  budget_remaining: fraction of budget left [0=exhausted → -10 penalty, 1=full]
  queue_backlog: pending requests (higher = more urgent to route)
  system_latency: current latency pressure
  step_count: episode progress [0=start, 1=end]

VALID ACTIONS (output ONLY one):
  route_to_a
  route_to_b
  route_to_c
  shed_load

COST PRINCIPLES (reason from these, not a fixed lookup):
- Budget exhaustion (budget_remaining → 0) triggers a -10 penalty that wipes all reward.
  Protect budget: route_to_a when healthy, it costs 1/10th of route_to_c.
- A provider is "healthy" when its status > 0.52. Below that, routing there risks failure.
- shed_load is a last resort only when routing is impossible: every provider has status ≤ 0.52, OR budget cannot absorb even the cheapest available healthy provider. If any provider is healthy and affordable, shed_load is always the wrong choice.
- When budget_remaining < 0.10, avoid route_to_c entirely regardless of its health.
- Provider degradation is permanent within an episode: once a provider's status begins falling, it will not recover. Treat any provider with declining status as a permanent reliability risk for the remainder of the episode.

REASONING APPROACH:
1. Is any provider healthy (status > 0.52)?
2. If yes: pick the cheapest healthy provider that fits the budget.
3. If no: shed_load.
4. Always protect budget_remaining — it is the binding constraint.

Output only the action string."""

app = typer.Typer(add_completion=False)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

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
            print(f"[llm_error] step API_ERROR: {e}", file=sys.stdout, flush=True)
            action_str = "shed_load"
        return Action(action_type=ActionType(action_str))


def _emit_log(prefix: str, payload: Dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload)}", flush=True)


def _observation_payload(observation: Observation) -> Dict[str, float]:
    return {
        "provider_a_status": float(observation.provider_a_status),
        "provider_b_status": float(observation.provider_b_status),
        "provider_c_status": float(observation.provider_c_status),
        "budget_remaining": float(observation.budget_remaining),
        "queue_backlog": float(observation.queue_backlog),
        "system_latency": float(observation.system_latency),
        "step_count": float(observation.step_count),
    }


def log_start(task: str, seed: int, episode: int) -> None:
    _emit_log("[START]", {"task": task, "seed": seed, "episode": episode})


def log_step(step: int, action: str, reward: float, done: bool, observation: Observation) -> None:
    _emit_log(
        "[STEP]",
        {
            "step": step,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "observation": _observation_payload(observation),
        },
    )


def log_end(task: str, seed: int, episode: int, total_reward: float, grade: float) -> None:
    _emit_log(
        "[END]",
        {
            "task": task,
            "seed": seed,
            "episode": episode,
            "total_reward": float(total_reward),
            "grade": float(grade),
        },
    )


def select_policy(policy_name: Literal["heuristic", "llm"]) -> object:
    if policy_name == "heuristic":
        return heuristic_baseline_policy

    if not HF_TOKEN:
        raise RuntimeError(
            "LLM policy requires HF_TOKEN and reads API_BASE_URL and MODEL_NAME from environment variables."
        )
    return LLMRouter(api_base_url=API_BASE_URL, model_name=MODEL_NAME, api_key=HF_TOKEN)


def choose_action(policy_name: Literal["heuristic", "llm"], policy: object, observation: Observation) -> Action:
    if policy_name == "heuristic":
        return policy(observation)
    return policy.choose_action(observation)


def run_episode(
    env: BudgetRouterEnv,
    scenario: TaskConfig,
    seed: int,
    episode: int,
    policy_name: Literal["heuristic", "llm"],
    policy: object,
) -> Dict[str, Any]:
    total_reward = 0.0
    grader_score: float | None = None

    log_start(task=scenario.name, seed=seed, episode=episode)

    try:
        observation = env.reset(seed=seed, scenario=scenario)
        while not observation.done:
            action = choose_action(policy_name=policy_name, policy=policy, observation=observation)
            action_name = action.action_type.value
            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            total_reward += reward
            log_step(
                step=env._internal.current_step,
                action=action_name,
                reward=reward,
                done=bool(observation.done),
                observation=observation,
            )

        metrics = episode_metrics(env._internal.history)
        metrics["seed"] = seed
        metrics["episode"] = episode
        metrics["total_reward"] = round(total_reward, 4)
        metrics["episode_length"] = env._internal.current_step
        grader = grade_episode(env._internal.history)
        grader_score = float(grader["overall_score"])
        metrics["grader_score"] = grader_score
        metrics["grader_breakdown"] = grader
        return metrics
    finally:
        if grader_score is None:
            grader_score = float(grade_episode(env._internal.history)["overall_score"])
        log_end(
            task=scenario.name,
            seed=seed,
            episode=episode,
            total_reward=round(total_reward, 4),
            grade=grader_score,
        )


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
    selected_policy = select_policy(policy)
    selected_tasks = TASKS if scenario == "all" else {scenario: TASKS[scenario]}
    results: Dict[str, Dict[str, object]] = {}
    episode = 1

    for task_name, task_config in selected_tasks.items():
        task_metrics = []
        for seed in SEED_SETS[seed_set]:
            env = BudgetRouterEnv()
            task_metrics.append(
                run_episode(
                    env=env,
                    scenario=task_config,
                    seed=seed,
                    episode=episode,
                    policy_name=policy,
                    policy=selected_policy,
                )
            )
            episode += 1
        results[task_name] = {
            "policy": policy,
            "seed_set": seed_set,
            "summary": summarize(task_metrics),
            "episodes": task_metrics,
        }

    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
