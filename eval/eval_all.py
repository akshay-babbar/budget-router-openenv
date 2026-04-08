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

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ── Config ──────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskConfig] = {
    "easy":       EASY,
    "medium":     MEDIUM,
    "hard":       HARD,
    "hard_multi": HARD_MULTI,
}

SEED_SETS: Dict[str, List[int]] = {
    "dev":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "heldout": [100, 101, 102, 103, 104],
}

API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

_VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]

SYSTEM_PROMPT = """
You are a cost-aware LLM API routing agent managing a production system.
At each step, output EXACTLY ONE action string. Nothing else.

ENVIRONMENT:
  Three providers: A ($0.01/req, cheapest), B ($0.05/req), C ($0.10/req, most reliable).
  provider_X_status = windowed success rate [0=always fails, 1=always succeeds].
  budget_remaining: fraction of budget left. Reaching 0 = catastrophic -10 penalty.
  step_count [0→1], steps_remaining: episode progress (20 steps total).

VALID ACTIONS (output ONLY one):
  route_to_a | route_to_b | route_to_c | shed_load

GOLDEN RULE — DEFAULT STRATEGY:
  Stay on the CHEAPEST provider whose status > 0.52. Only deviate if there is CLEAR, SUSTAINED evidence of degradation (defined below). Unnecessary switching to expensive providers burns budget and reduces your score.

NOISE CALIBRATION (critical):
- Status fluctuates due to Bernoulli sampling noise. Single-step dips are not reliable signals.
- Use the provided 2-step trend (avg/step): a sustained negative trend across multiple steps
indicates real degradation; a trend near 0 means the provider is stable. Do NOT switch on noise.
- REAL degradation signal: sustained negative trend AND current status is visibly declining.
- Only when both conditions hold across consecutive observations should you consider early switching.
- On stable tasks, trends hover near zero. Switching on noise burns budget without benefit.


WHEN TO SWITCH (use your conversation history):
A → B: When trend_a is clearly and consistently negative AND status_a is approaching unreliable,
           OR status_a is already below 0.52 (failure probability exceeds success probability).
B → C: Same principle — sustained decline signals, not single-step noise.
Never switch based on a single bad observation — noise causes occasional dips.

BUDGET RUNWAY:
If budget_runway_at_current_rate < steps_remaining, switch to a cheaper provider immediately.
TASK PROFILES (the task name appears in each observation — use it):
  easy:       Stable environment. Trend fluctuations are mostly noise. Stay on the cheapest provider unless its trend is catastrophically and sustainedly negative.
  medium:     Dynamic environment. A provider may degrade mid-episode. Monitor trends and switch to the next cheapest healthy fallback if the primary fails.
  hard / hard_multi: Hostile, multi-failure environments. Multiple providers may degrade at unexpected times in unpredictable sequences.
              Your Runbook: Always map traffic to the lowest-cost healthy provider (A=$0.01, B=$0.05, C=$0.10).
              Watch your conversation history: if your currently active provider shows a clear, sustained negative trend, switch early to the next cheapest option that is healthy.
              CRITICAL: Before switching to expensive fallbacks (like C), use budget_runway to verify you can afford them to prevent budget exhaustion.

Output only the action string."""


# ── Policies ────────────────────────────────────────────────────────────────

def _parse_action(text: str) -> str:
    text = text.strip().lower()
    for a in _VALID_ACTIONS:
        if a in text:
            return a
    return "shed_load"


class LLMPolicy:
    def __init__(self) -> None:
        if not HAS_OPENAI:
            raise RuntimeError("openai package not installed: pip install openai")
        if not API_KEY:
            raise RuntimeError("No API key found. Set HF_TOKEN or API_KEY env var.")
        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=15.0, max_retries=0)
        self._messages: List[Dict] = []
        self.last_error: Optional[str] = None
        self._prev_obs: Optional[dict] = None
        self._prev2_obs: Optional[dict] = None
        self._task_name: str = ""

    def reset(self, task_name: str = "") -> None:
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_error = None
        self._prev_obs = None
        self._prev2_obs = None
        self._task_name = task_name

    def choose_action(self, obs: Observation) -> str:
        # ── Compute 2-step trend (more noise-robust than single-step delta) ──
        trend_text = ""
        budget_runway_text = ""

        if self._prev2_obs is not None:
            ta = (obs.provider_a_status - self._prev2_obs["a"]) / 2.0
            tb = (obs.provider_b_status - self._prev2_obs["b"]) / 2.0
            tc = (obs.provider_c_status - self._prev2_obs["c"]) / 2.0
            trend_text = f"\ntrend (avg/step, 2-step):  A:{ta:+.3f}  B:{tb:+.3f}  C:{tc:+.3f}"
        elif self._prev_obs is not None:
            ta = obs.provider_a_status - self._prev_obs["a"]
            tb = obs.provider_b_status - self._prev_obs["b"]
            tc = obs.provider_c_status - self._prev_obs["c"]
            trend_text = f"\ntrend (1-step only, noisy): A:{ta:+.3f}  B:{tb:+.3f}  C:{tc:+.3f}"

        if self._prev_obs is not None:
            budget_spent = self._prev_obs["budget"] - obs.budget_remaining
            if budget_spent > 0.001:
                runway = int(obs.budget_remaining / budget_spent)
                budget_runway_text = f"\nbudget_runway_at_current_rate: ~{runway} steps"
            else:
                budget_runway_text = "\nbudget_runway_at_current_rate: >20 steps"

        steps_total = 20
        steps_remaining = max(1, steps_total - int(round(obs.step_count * steps_total)))
        task_line = f"\ntask: {self._task_name}" if self._task_name else ""

        obs_text = "\n".join([
            f"provider_a_status: {obs.provider_a_status:.3f}",
            f"provider_b_status: {obs.provider_b_status:.3f}",
            f"provider_c_status: {obs.provider_c_status:.3f}",
            f"budget_remaining:  {obs.budget_remaining:.3f}",
            f"queue_backlog:     {obs.queue_backlog:.3f}",
            f"system_latency:    {obs.system_latency:.3f}",
            f"step_count:        {obs.step_count:.3f}",
            f"steps_remaining:   {steps_remaining}",
        ])
        obs_text += trend_text + budget_runway_text + task_line

        # Shift history: prev becomes prev2, current becomes prev
        self._prev2_obs = self._prev_obs
        self._prev_obs = {
            "a": obs.provider_a_status,
            "b": obs.provider_b_status,
            "c": obs.provider_c_status,
            "budget": obs.budget_remaining,
        }

        self._messages.append({"role": "user", "content": f"Observation:\n{obs_text}\n\nAction:"})
        try:
            resp = self._client.chat.completions.create(
                model=MODEL_NAME, messages=self._messages, max_tokens=30, temperature=0
            )
            raw = resp.choices[0].message.content or ""
            action = _parse_action(raw)
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            action = "shed_load"
        self._messages.append({"role": "assistant", "content": action})
        return action


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
            action_str = policy.choose_action(obs)

        obs = env.step(Action(action_type=ActionType(action_str)))
        reward = float(obs.reward or 0.0)
        rewards.append(reward)
        actions.append(action_str)

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
                    note = f"LLM +{diff*100:.1f}% vs heuristic"
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
                policy_instances["llm"] = LLMPolicy()
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