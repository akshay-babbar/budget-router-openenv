#!/usr/bin/env python3
"""
Generate SFT data for Budget Router.

Default path is deliberately zero-API-cost: distill the existing PPO hard_multi
policy into chat transcripts, then push the dataset to the Hub for HF Jobs.

Optional LLM labeling is available with --teacher llm, but it costs one large
model call per environment step (20 calls per episode).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import HARD_MULTI, TASK_PRESETS
from inference import LLMRouter, SYSTEM_PROMPT


VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]
PPO_ACTION_NAMES = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]
DEFAULT_DATASET_REPO = "akshay4/budget-router-sft-data"
DEFAULT_PPO_MODEL_PATH = "trained_models/ppo_hard_multi_100k.zip"
_PPO_POLICY_CACHE: dict[str, Callable[[Observation], str]] = {}


def _obs_to_array(obs: Observation) -> np.ndarray:
    return np.array(
        [
            obs.provider_a_status,
            obs.provider_b_status,
            obs.provider_c_status,
            obs.budget_remaining,
            obs.queue_backlog,
            obs.system_latency,
            obs.step_count,
        ],
        dtype=np.float32,
    )


def _steps_remaining(obs: Observation, max_steps: int = 20) -> int:
    elapsed = int(round(float(obs.step_count) * max_steps))
    return max(0, max_steps - elapsed)


def _trend_text(obs: Observation, previous_obs: Observation | None, previous2_obs: Observation | None) -> str:
    if previous2_obs is not None:
        ta = (obs.provider_a_status - previous2_obs.provider_a_status) / 2.0
        tb = (obs.provider_b_status - previous2_obs.provider_b_status) / 2.0
        tc = (obs.provider_c_status - previous2_obs.provider_c_status) / 2.0
        return f"trend (avg/step, 2-step): A:{ta:+.3f} B:{tb:+.3f} C:{tc:+.3f}"
    if previous_obs is not None:
        ta = obs.provider_a_status - previous_obs.provider_a_status
        tb = obs.provider_b_status - previous_obs.provider_b_status
        tc = obs.provider_c_status - previous_obs.provider_c_status
        return f"trend (1-step only, noisy): A:{ta:+.3f} B:{tb:+.3f} C:{tc:+.3f}"
    return "trend: unavailable"


def _budget_runway_text(obs: Observation, previous_obs: Observation | None) -> str:
    if previous_obs is None:
        return "budget_runway_at_current_rate: >20 steps"
    budget_spent = float(previous_obs.budget_remaining) - float(obs.budget_remaining)
    if budget_spent <= 0.001:
        return "budget_runway_at_current_rate: >20 steps"
    runway = int(float(obs.budget_remaining) / budget_spent)
    return f"budget_runway_at_current_rate: ~{runway} steps"


def _previous_step_feedback(obs: Observation) -> str:
    metadata = getattr(obs, "metadata", None) or {}
    if not metadata.get("action_type"):
        return ""

    parts = [
        "previous_step_feedback:",
        f"  previous_action: {metadata.get('action_type')}",
    ]
    if obs.reward is not None:
        parts.append(f"  previous_reward: {float(obs.reward):+.2f}")
    if metadata.get("request_succeeded") is not None:
        parts.append(f"  previous_success: {str(bool(metadata.get('request_succeeded'))).lower()}")
    if metadata.get("cost") is not None:
        parts.append(f"  previous_cost: {float(metadata.get('cost')):.2f}")
    if metadata.get("latency_ms") is not None:
        parts.append(f"  previous_latency_ms: {float(metadata.get('latency_ms')):.2f}")
    if metadata.get("budget_exhausted"):
        parts.append("  previous_budget_exhausted: true")
    return "\n".join(parts)


def format_observation_for_sft(
    *,
    obs: Observation,
    task_name: str,
    previous_obs: Observation | None,
    previous2_obs: Observation | None,
) -> str:
    """Public observation text used consistently for SFT train/eval."""
    lines = [
        f"task: {task_name}",
        f"provider_a_status: {obs.provider_a_status:.3f}",
        f"provider_b_status: {obs.provider_b_status:.3f}",
        f"provider_c_status: {obs.provider_c_status:.3f}",
        f"budget_remaining: {obs.budget_remaining:.3f}",
        f"queue_backlog: {obs.queue_backlog:.3f}",
        f"system_latency: {obs.system_latency:.3f}",
        f"step_count: {obs.step_count:.3f}",
        f"steps_remaining: {_steps_remaining(obs)}",
        _trend_text(obs, previous_obs, previous2_obs),
        _budget_runway_text(obs, previous_obs),
    ]
    feedback = _previous_step_feedback(obs)
    if feedback:
        lines.append(feedback)
    return "\n".join(lines)


def run_heuristic_episode(task_cfg: TaskConfig, seed: int) -> dict[str, Any]:
    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=task_cfg)
    total_reward = 0.0
    while not obs.done:
        obs = env.step(heuristic_baseline_policy(obs))
        total_reward += float(obs.reward or 0.0)
    grader = grade_episode(env._internal.history)
    return {
        "grader_score": float(grader["overall_score"]),
        "total_reward": total_reward,
        "grader": grader,
    }


def _load_ppo_policy(model_path: str) -> Callable[[Observation], str]:
    if model_path in _PPO_POLICY_CACHE:
        return _PPO_POLICY_CACHE[model_path]

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "PPO teacher requires training dependencies. Run `uv sync --extra training` "
            "or use --teacher heuristic/llm."
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {path}")
    model = PPO.load(str(path))

    def choose(obs: Observation) -> str:
        action_idx, _ = model.predict(_obs_to_array(obs), deterministic=True)
        idx = int(action_idx)
        return PPO_ACTION_NAMES[idx] if 0 <= idx < len(PPO_ACTION_NAMES) else "shed_load"

    _PPO_POLICY_CACHE[model_path] = choose
    return choose


def _load_llm_policy(task_name: str) -> Callable[[Observation], str]:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("LLM teacher requires HF_TOKEN or API_KEY in the environment.")
    router = LLMRouter(
        api_base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        api_key=api_key,
    )
    router.reset(task_name=task_name)

    def choose(obs: Observation) -> str:
        return router.choose_action(obs).action_type.value

    return choose


def collect_teacher_episode(
    *,
    task_name: str,
    task_cfg: TaskConfig,
    seed: int,
    teacher: str,
    ppo_model_path: str,
) -> dict[str, Any]:
    if teacher == "ppo":
        choose_action = _load_ppo_policy(ppo_model_path)
    elif teacher == "heuristic":
        choose_action = lambda obs: heuristic_baseline_policy(obs).action_type.value
    elif teacher == "llm":
        choose_action = _load_llm_policy(task_name)
    else:
        raise ValueError(f"Unknown teacher {teacher!r}")

    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=task_cfg)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    previous2_obs: Observation | None = None
    previous_obs: Observation | None = None
    actions: list[str] = []
    total_reward = 0.0

    while not obs.done:
        obs_text = format_observation_for_sft(
            obs=obs,
            task_name=task_name,
            previous_obs=previous_obs,
            previous2_obs=previous2_obs,
        )
        action_str = choose_action(obs)
        if action_str not in VALID_ACTIONS:
            action_str = "shed_load"

        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": action_str})
        actions.append(action_str)

        previous2_obs = previous_obs
        previous_obs = obs
        obs = env.step(Action(action_type=ActionType(action_str)))
        total_reward += float(obs.reward or 0.0)

    grader = grade_episode(env._internal.history)
    return {
        "seed": seed,
        "teacher": teacher,
        "messages": messages,
        "actions": actions,
        "grader_score": float(grader["overall_score"]),
        "total_reward": total_reward,
        "grader": grader,
        "metrics": episode_metrics(env._internal.history),
    }


def select_training_rows(
    episodes: list[dict[str, Any]],
    *,
    top_fraction: float,
    min_keep: int,
    min_delta: float,
) -> list[dict[str, Any]]:
    ranked = sorted(episodes, key=lambda item: float(item["delta_vs_heuristic"]), reverse=True)
    target = max(min_keep, int(math.ceil(len(ranked) * top_fraction)))
    positive = [ep for ep in ranked if float(ep["delta_vs_heuristic"]) >= min_delta]
    source = positive if len(positive) >= min_keep else ranked
    return source[: min(target, len(source))]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Budget Router SFT dataset.")
    parser.add_argument("--teacher", choices=["ppo", "heuristic", "llm"], default=os.getenv("TEACHER_POLICY", "ppo"))
    parser.add_argument("--task", default=os.getenv("TASK_NAME", "hard_multi"), choices=sorted(TASK_PRESETS))
    parser.add_argument("--start-seed", type=int, default=int(os.getenv("SFT_START_SEED", "1000")))
    parser.add_argument("--n-episodes", type=int, default=int(os.getenv("SFT_N_EPISODES", "100")))
    parser.add_argument("--top-fraction", type=float, default=float(os.getenv("SFT_TOP_FRACTION", "0.30")))
    parser.add_argument("--min-keep", type=int, default=int(os.getenv("SFT_MIN_KEEP", "20")))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("SFT_MIN_DELTA", "0.0")))
    parser.add_argument("--ppo-model-path", default=os.getenv("PPO_MODEL_PATH", DEFAULT_PPO_MODEL_PATH))
    parser.add_argument("--dataset-repo", default=os.getenv("DATASET_REPO", DEFAULT_DATASET_REPO))
    parser.add_argument("--local-jsonl", default=os.getenv("SFT_LOCAL_JSONL", "outputs/sft_dataset.jsonl"))
    parser.add_argument("--no-push", action="store_true", help="Write local JSONL only; do not push to Hub.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = TASK_PRESETS[args.task]
    seeds = list(range(args.start_seed, args.start_seed + args.n_episodes))

    if args.teacher == "llm":
        print(
            f"[sft-data] teacher=llm n_episodes={args.n_episodes}; "
            f"expected large-model calls <= {args.n_episodes * task_cfg.max_steps}",
            flush=True,
        )
    else:
        print(f"[sft-data] teacher={args.teacher} uses 0 large-LLM calls", flush=True)

    episodes: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds, start=1):
        teacher_ep = collect_teacher_episode(
            task_name=args.task,
            task_cfg=task_cfg,
            seed=seed,
            teacher=args.teacher,
            ppo_model_path=args.ppo_model_path,
        )
        heuristic_ep = run_heuristic_episode(task_cfg, seed)
        delta = teacher_ep["grader_score"] - heuristic_ep["grader_score"]
        teacher_ep["heuristic_score"] = heuristic_ep["grader_score"]
        teacher_ep["delta_vs_heuristic"] = delta
        episodes.append(teacher_ep)
        print(
            f"[sft-data] {i:03d}/{len(seeds)} seed={seed} "
            f"teacher={teacher_ep['grader_score']:.4f} heuristic={heuristic_ep['grader_score']:.4f} "
            f"delta={delta:+.4f}",
            flush=True,
        )

    kept = select_training_rows(
        episodes,
        top_fraction=args.top_fraction,
        min_keep=args.min_keep,
        min_delta=args.min_delta,
    )
    dataset_rows = [
        {
            "messages": ep["messages"],
            "seed": ep["seed"],
            "teacher": ep["teacher"],
            "teacher_score": ep["grader_score"],
            "heuristic_score": ep["heuristic_score"],
            "delta_vs_heuristic": ep["delta_vs_heuristic"],
            "actions": ep["actions"],
        }
        for ep in kept
    ]
    write_jsonl(Path(args.local_jsonl), dataset_rows)

    mean_all = sum(float(ep["grader_score"]) for ep in episodes) / len(episodes)
    mean_kept = sum(float(ep["grader_score"]) for ep in kept) / len(kept)
    mean_delta = sum(float(ep["delta_vs_heuristic"]) for ep in kept) / len(kept)
    print(
        "[sft-data] summary "
        f"generated={len(episodes)} kept={len(kept)} mean_all={mean_all:.4f} "
        f"mean_kept={mean_kept:.4f} mean_delta_kept={mean_delta:+.4f} "
        f"local_jsonl={args.local_jsonl}",
        flush=True,
    )

    if not args.no_push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be set to push the dataset. Use --no-push for local only.")
        from datasets import Dataset

        Dataset.from_list(dataset_rows).push_to_hub(args.dataset_repo, token=token)
        print(f"[sft-data] pushed dataset to https://huggingface.co/datasets/{args.dataset_repo}", flush=True)


if __name__ == "__main__":
    main()
