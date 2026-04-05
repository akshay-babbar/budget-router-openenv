"""
Generate sample episode output JSON files for easy (seed 42) and hard_multi (seed 42).

Usage:
    uv run python train/gen_outputs.py

Output:
    outputs/episode_easy_seed42.json
    outputs/episode_hard_multi_seed42.json
"""
from __future__ import annotations

import json
from pathlib import Path

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import grade_episode
from budget_router.tasks import EASY, HARD_MULTI


def capture_episode(scenario, seed: int) -> dict:
    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=scenario)
    steps = []

    while not obs.done:
        action: Action = heuristic_baseline_policy(obs)
        obs_before = {
            "provider_a_status": float(obs.provider_a_status),
            "provider_b_status": float(obs.provider_b_status),
            "provider_c_status": float(obs.provider_c_status),
            "budget_remaining": float(obs.budget_remaining),
            "queue_backlog": float(obs.queue_backlog),
            "system_latency": float(obs.system_latency),
            "step_count": float(obs.step_count),
        }
        obs = env.step(action)
        # Serialize metadata — coerce non-JSON-native types
        meta = {}
        for k, v in (obs.metadata or {}).items():
            if isinstance(v, (int, float, bool, str, type(None))):
                meta[k] = v
            else:
                meta[k] = str(v)

        steps.append({
            "step": int(meta.get("step", len(steps) + 1)),
            "action": action.action_type.value,
            "observation_before": obs_before,
            "observation_after": {
                "provider_a_status": float(obs.provider_a_status),
                "provider_b_status": float(obs.provider_b_status),
                "provider_c_status": float(obs.provider_c_status),
                "budget_remaining": float(obs.budget_remaining),
                "queue_backlog": float(obs.queue_backlog),
                "system_latency": float(obs.system_latency),
                "step_count": float(obs.step_count),
            },
            "reward": float(obs.reward),
            "done": bool(obs.done),
            "metadata": meta,
        })

    grader_raw = grade_episode(env._internal.history)
    grader = {k: float(v) for k, v in grader_raw.items()}
    return {
        "scenario": scenario.name,
        "seed": seed,
        "policy": "heuristic",
        "total_steps": len(steps),
        "grader": grader,
        "steps": steps,
    }


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("Capturing easy seed=42 ...")
    easy = capture_episode(EASY, 42)
    (output_dir / "episode_easy_seed42.json").write_text(json.dumps(easy, indent=2))
    print(f"  total_steps={easy['total_steps']}  grader={easy['grader']['overall_score']:.4f}")

    print("Capturing hard_multi seed=42 ...")
    hm = capture_episode(HARD_MULTI, 42)
    (output_dir / "episode_hard_multi_seed42.json").write_text(json.dumps(hm, indent=2))
    print(f"  total_steps={hm['total_steps']}  grader={hm['grader']['overall_score']:.4f}")

    print("✅ Done — outputs/ contains 2 valid JSON files")


if __name__ == "__main__":
    main()
