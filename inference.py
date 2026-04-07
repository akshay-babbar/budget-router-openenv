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


SYSTEM_PROMPT = """
You are a cost-aware LLM API routing agent managing a production system.
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

Output only the action string.
You will receive observations sequentially across steps. Use the full conversation history to detect trends: if a provider's status has dropped across consecutive observations, treat it as actively degrading and switch proactively. If a provider's status is declining and the budget is insufficient to route to the most reliable remaining provider for all remaining steps, prefer the cheapest viable provider now to preserve budget optionality."""

app = typer.Typer(add_completion=False)

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS") or "10")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES") or "0")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME") or "budget_router"

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
        self._client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        self._model_name = model_name
        self._messages: List[Dict[str, str]] = []
        self.last_error: str | None = None
        self.reset()

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_error = None

    def choose_action(self, observation: Observation) -> Action:
        obs = observation
        if not self._messages or obs.step_count == 0.0:
            self.reset()
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
        self._messages.append({"role": "user", "content": user_prompt})
        try:
            response = client.with_options(timeout=LLM_TIMEOUT_SECONDS).chat.completions.create(
                model=model_name,
                messages=self._messages,
                max_tokens=30,
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            action_str = _parse_llm_action(raw)
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            action_str = "shed_load"
        self._messages.append({"role": "assistant", "content": action_str})
        return Action(action_type=ActionType(action_str))


def _single_line(value: str | None) -> str:
    if not value:
        return "null"
    return str(value).replace("\n", " ").replace("\r", " ")


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


def _reported_score(value: float) -> float:
    return min(max(float(value), 0.001), 0.999)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={_single_line(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_reported_score(score):.3f} rewards={rewards_str}",
        flush=True,
    )


def select_policy(policy_name: Literal["heuristic", "llm"]) -> object:
    if policy_name == "heuristic":
        return heuristic_baseline_policy

    if not API_KEY or not API_BASE_URL:
        raise RuntimeError(
            "LLM policy requires API_BASE_URL and API_KEY and reads MODEL_NAME from environment variables."
        )
    return LLMRouter(api_base_url=API_BASE_URL, model_name=MODEL_NAME, api_key=API_KEY)


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
    rewards: List[float] = []
    steps_taken = 0
    success = False

    if policy_name == "llm":
        policy.reset()

    log_start(task=scenario.name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        observation = env.reset(seed=seed, scenario=scenario)
        while not observation.done:
            action = choose_action(policy_name=policy_name, policy=policy, observation=observation)
            action_name = action.action_type.value
            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            total_reward += reward
            rewards.append(reward)
            steps_taken = env._internal.current_step
            step_error = getattr(policy, "last_error", None) if policy_name == "llm" else None
            log_step(
                step=env._internal.current_step,
                action=action_name,
                reward=reward,
                done=bool(observation.done),
                error=step_error,
            )

        metrics = episode_metrics(env._internal.history)
        metrics["seed"] = seed
        metrics["episode"] = episode
        metrics["total_reward"] = round(total_reward, 4)
        metrics["episode_length"] = env._internal.current_step
        grader = grade_episode(env._internal.history)
        grader_score = float(grader["overall_score"])
        success = grader_score > 0.0
        metrics["grader_score"] = grader_score
        metrics["grader_breakdown"] = grader
        return metrics
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        if grader_score is None:
            grader_score = float(grade_episode(env._internal.history)["overall_score"])
            success = grader_score > 0.0
        log_end(success=success, steps=steps_taken, score=grader_score, rewards=rewards)


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
    policy: Literal["heuristic", "llm"] = typer.Option("llm" if API_KEY and API_BASE_URL else "heuristic"),
    seed_set: Literal["development", "heldout"] = typer.Option("development"),
    scenario: Literal["all", "easy", "medium", "hard", "hard_multi"] = typer.Option("hard_multi"),
    max_seeds: int = typer.Option(1),
    output_path: Path = typer.Option(Path("baseline_results.json")),
) -> None:
    selected_policy = select_policy(policy)
    selected_tasks = TASKS if scenario == "all" else {scenario: TASKS[scenario]}
    selected_seeds = SEED_SETS[seed_set][: max(1, max_seeds)]
    results: Dict[str, Dict[str, object]] = {}
    episode = 1

    for task_name, task_config in selected_tasks.items():
        task_metrics = []
        for seed in selected_seeds:
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
