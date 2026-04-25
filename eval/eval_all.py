#!/usr/bin/env python3
"""
eval_all.py — Budget Router Consolidated Evaluator
====================================================
Runs heuristic + LLM (+ optional PPO) across all tasks and seeds.
Outputs a Markdown table + per-episode JSON to outputs/.

Usage:
    # Quick (3 seeds, heuristic + LLM):
    uv run python eval_all.py

    # Full (10 seeds, all policies):
    uv run python eval_all.py --seeds 10 --policies heuristic llm

    # Heuristic only (no API needed):
    uv run python eval_all.py --policies heuristic

    # Specific tasks:
    uv run python eval_all.py --tasks hard hard_multi --seeds 5

Prerequisites:
    export HF_TOKEN=<your_hf_token>          # required for LLM policy
    export API_BASE_URL=https://router.huggingface.co/v1  # default
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct           # default

Output:
    outputs/eval_results_<timestamp>.json    — full per-episode data
    outputs/eval_summary_<timestamp>.md      — markdown table for README
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer

# ── Add parent to path so we can import budget_router ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import EASY, HARD, HARD_MULTI, MEDIUM

from inference import LLMRouter

# ── Config ──────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskConfig] = {
    "easy":       EASY,
    "medium":     MEDIUM,
    "hard":       HARD,
    "hard_multi": HARD_MULTI,
}

SEED_SETS: Dict[str, List[int]] = {
    "dev":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "heldout": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
}

API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
LLM_LOG_RAW = (os.getenv("LLM_LOG_RAW") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
LLM_LOG_RAW_MAX_CHARS = int(os.getenv("LLM_LOG_RAW_MAX_CHARS") or "220")


def _single_line(value: str | None) -> str:
    if not value:
        return "null"
    return str(value).replace("\n", " ").replace("\r", " ")


def _truncate(value: str | None, max_chars: int) -> str:
    s = _single_line(value).strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


# ── Policies ────────────────────────────────────────────────────────────────
def _llm_choose_action(policy: LLMRouter, obs: Observation) -> str:
    action = policy.choose_action(obs)
    return action.action_type.value


def _heuristic(obs: Observation) -> str:
    return heuristic_baseline_policy(obs).action_type.value


# ── Episode runner ───────────────────────────────────────────────────────────

def run_one_episode(
    task_name: str,
    task_cfg: TaskConfig,
    seed: int,
    policy_name: str,
    policy,  # callable or LLMPolicy
) -> Dict:
    env = BudgetRouterEnv()
    if policy_name == "llm":
        policy.reset(task_name=task_name)

    obs = env.reset(seed=seed, scenario=task_cfg)
    rewards = []
    actions = []

    while not obs.done:
        if policy_name == "heuristic":
            action_str = _heuristic(obs)
        else:
            action_str = _llm_choose_action(policy, obs)

        obs = env.step(Action(action_type=ActionType(action_str)))
        reward = float(obs.reward or 0.0)
        rewards.append(reward)
        actions.append(action_str)
        if policy_name == "llm" and LLM_LOG_RAW:
            llm_raw = getattr(policy, "last_raw_output", None)
            llm_parsed = getattr(policy, "last_parsed_action", None)
            typer.echo(
                f"[LLM] step={env._internal.current_step} action={action_str} "
                f"reward={reward:+.2f} llm_raw={_truncate(llm_raw, max(20, LLM_LOG_RAW_MAX_CHARS))} "
                f"llm_parsed={_single_line(llm_parsed)}"
            )

    grader = grade_episode(env._internal.history)
    metrics = episode_metrics(env._internal.history)

    return {
        "task":          task_name,
        "seed":          seed,
        "policy":        policy_name,
        "total_reward":  round(sum(rewards), 4),
        "grader_score":  round(grader["overall_score"], 4),
        "success_score": round(grader["success_score"], 4),
        "budget_score":  round(grader["budget_score"], 4),
        "adaptation_score": round(grader["adaptation_score"], 4),
        "latency_score": round(grader["latency_score"], 4),
        "sla_score":     round(grader["sla_score"], 4),
        "success_rate":  round(metrics["success_rate"], 4),
        "steps":         len(rewards),
        "actions":       actions,
        "rewards":       rewards,
    }


# ── Summary helpers ──────────────────────────────────────────────────────────

def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def build_summary(results: List[Dict]) -> Dict:
    summary = {}
    for r in results:
        key = (r["task"], r["policy"])
        summary.setdefault(key, []).append(r)
    return {
        f"{task}|{pol}": {
            "grader_mean":   _mean([e["grader_score"] for e in eps]),
            "reward_mean":   _mean([e["total_reward"] for e in eps]),
            "success_rate":  _mean([e["success_rate"] for e in eps]),
            "adaptation":    _mean([e["adaptation_score"] for e in eps]),
            "n":             len(eps),
        }
        for (task, pol), eps in summary.items()
    }


def render_markdown_table(summary: Dict, policies: List[str], tasks: List[str]) -> str:
    task_labels = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "hard_multi": "Hard_Multi"}
    pol_headers = " | ".join(f"{p.upper()} Grader" for p in policies)
    lines = [
        f"| Task | {pol_headers} | Notes |",
        "|" + "---|" * (len(policies) + 2),
    ]
    for task in tasks:
        scores = []
        for p in policies:
            key = f"{task}|{p}"
            s = summary.get(key, {})
            if s:
                n = s["n"]
                scores.append(f"{s['grader_mean']:.4f} (n={n})")
            else:
                scores.append("—")
        note = ""
        if task == "hard_multi" and len(policies) >= 2:
            k0 = f"{task}|{policies[0]}"
            k1 = f"{task}|{policies[1]}"
            if k0 in summary and k1 in summary:
                diff = summary[k1]["grader_mean"] - summary[k0]["grader_mean"]
                if diff > 0:
                    note = f"LLM +{diff*100:.1f} points vs heuristic"
        line = f"| {task_labels.get(task, task)} | {' | '.join(scores)} | {note} |"
        lines.append(line)
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False)


@app.command()
def main(
    policies: List[str] = typer.Option(["heuristic", "llm"], help="Policies to run"),
    tasks:    List[str] = typer.Option(["easy", "medium", "hard", "hard_multi"], help="Tasks"),
    seeds:    int       = typer.Option(3, help="Number of dev seeds (1-10, costs scale with LLM)"),
    seed_set: str       = typer.Option("dev", help="Seed set: dev | heldout"),
    out_dir:  Path      = typer.Option(Path("outputs"), help="Output directory"),
) -> None:
    """Run Budget Router evaluation across policies, tasks, and seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    selected_seeds = SEED_SETS[seed_set][:max(1, min(seeds, len(SEED_SETS[seed_set])))]
    selected_tasks = {t: TASKS[t] for t in tasks if t in TASKS}

    if not selected_tasks:
        typer.echo(f"No valid tasks. Choose from: {list(TASKS)}", err=True)
        raise typer.Exit(1)

    # Build policy instances
    policy_instances = {}
    for p in policies:
        if p == "heuristic":
            policy_instances["heuristic"] = None  # uses _heuristic() directly
        elif p == "llm":
            try:
                if not API_KEY:
                    raise RuntimeError("No API key found. Set HF_TOKEN or API_KEY env var.")
                policy_instances["llm"] = LLMRouter(
                    api_base_url=API_BASE_URL, model_name=MODEL_NAME, api_key=API_KEY
                )
                typer.echo(f"LLM policy: {MODEL_NAME} via {API_BASE_URL}")
            except RuntimeError as e:
                typer.echo(f"[WARN] LLM policy unavailable: {e} — skipping", err=True)
        elif p == "ppo":
            typer.echo("[WARN] PPO eval not yet wired in this script — run your train_ppo.py separately", err=True)

    all_results = []
    total_episodes = len(policy_instances) * len(selected_tasks) * len(selected_seeds)
    done = 0

    for pol_name, pol_obj in policy_instances.items():
        for task_name, task_cfg in selected_tasks.items():
            for seed in selected_seeds:
                typer.echo(f"[{done+1}/{total_episodes}] {pol_name:10s} | {task_name:12s} | seed={seed} ...", nl=False)
                try:
                    result = run_one_episode(task_name, task_cfg, seed, pol_name, pol_obj)
                    all_results.append(result)
                    typer.echo(f" grader={result['grader_score']:.4f}  reward={result['total_reward']:+.2f}")
                except Exception as e:
                    typer.echo(f" ERROR: {e}", err=True)
                done += 1

    if not all_results:
        typer.echo("No results produced.", err=True)
        raise typer.Exit(1)

    # Save JSON
    json_path = out_dir / f"eval_results_{ts}.json"
    summary = build_summary(all_results)
    output = {"metadata": {"timestamp": ts, "policies": policies, "tasks": tasks, "seeds": selected_seeds}, "summary": summary, "episodes": all_results}
    json_path.write_text(json.dumps(output, indent=2))
    typer.echo(f"\nResults saved to {json_path}")

    # Save markdown table
    md_table = render_markdown_table(summary, list(policy_instances.keys()), list(selected_tasks.keys()))
    md_path = out_dir / f"eval_summary_{ts}.md"
    md_path.write_text(f"# Budget Router Evaluation — {ts}\n\n{md_table}\n")
    typer.echo(f"Markdown table saved to {md_path}")
    typer.echo(f"\n{md_table}")


if __name__ == "__main__":
    app()