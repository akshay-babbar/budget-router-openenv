#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "huggingface_hub>=0.24.0",
#   "scipy",
#   "budget-router @ git+https://huggingface.co/spaces/akshay4/budget-router-openenv",
# ]
# ///
"""Evaluate a Budget Router SFT model against the heuristic baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType, Observation, TaskConfig
from budget_router.policies import heuristic_baseline_policy
from budget_router.reward import episode_metrics, grade_episode
from budget_router.tasks import HARD_MULTI, TASK_PRESETS
try:
    from inference import SYSTEM_PROMPT

    _SYSTEM_PROMPT_SOURCE = "inference"
except ModuleNotFoundError as exc:
    if exc.name != "inference":
        raise
    SYSTEM_PROMPT = """
You are a cost-aware LLM API routing agent managing a production system.
At each step, output EXACTLY ONE action string. Nothing else.

ENVIRONMENT:
  Three providers: A ($0.01/req, cheapest), B ($0.05/req), C ($0.10/req, most reliable).
  provider_X_status = windowed success rate [0=always fails, 1=always succeeds].
    IMPORTANT: A status of exactly 0.500 means this provider has NEVER been routed to
    in this episode — it is unobserved, not confirmed healthy. Route to it once to get
    a real reading. Do not treat 0.500 as a health signal.
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

BUDGET RUNWAY — HARD CONSTRAINT:
budget_runway_at_current_rate shows how many more steps you can afford at current spend rate.
If budget_runway_at_current_rate < steps_remaining: switch to a cheaper provider IMMEDIATELY.
If budget_remaining < 0.15 (less than 15% left): treat C as OFF-LIMITS unless A and B are
  both below 0.30 status. Prefer shed_load over routing C when budget is this low.
NEVER route to any provider if doing so would leave budget_remaining below the cost of
that provider times the steps_remaining. The -10 bankruptcy penalty destroys all episode
value accumulated so far — budget survival is non-negotiable.
TASK PROFILES (the task name appears in each observation — use it):
  easy:       Stable environment. Trend fluctuations are mostly noise. Stay on the cheapest provider unless its trend is catastrophically and sustainedly negative.
  medium:     Dynamic environment. A provider may degrade mid-episode. Monitor trends and switch to the next cheapest healthy fallback if the primary fails.
  hard / hard_multi: Hostile, multi-failure environments. Multiple providers may degrade at unexpected times in unpredictable sequences.
              Your Runbook: Always map traffic to the lowest-cost healthy provider (A=$0.01, B=$0.05, C=$0.10).
              Watch your conversation history: if your currently active provider shows a clear, sustained negative trend, switch early to the next cheapest option that is healthy.
              CRITICAL: Before switching to expensive fallbacks (like C), use budget_runway to verify you can afford them to prevent budget exhaustion.

Output only the action string."""
    _SYSTEM_PROMPT_SOURCE = "embedded_fallback"

_AGENT_DEBUG_LOG = "/Users/akshaybabbar/Desktop/work/.cursor/debug-e4cac3.log"


def _agent_debug_ndjson(payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        print(f"[agent-debug] {line}", flush=True)


VALID_ACTIONS = ["route_to_a", "route_to_b", "route_to_c", "shed_load"]
DEFAULT_MODEL_REPO = "akshay4/budget-router-sft-qwen1.5b"


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


def parse_action(text: str) -> tuple[str, bool]:
    lowered = text.strip().lower()
    for action in VALID_ACTIONS:
        if action in lowered:
            return action, True
    return "route_to_a", False


def apply_budget_safety_guard(action_str: str, observation: Observation, task_cfg: TaskConfig) -> str:
    if action_str == "shed_load":
        return action_str
    costs = {
        "route_to_a": task_cfg.cost_a,
        "route_to_b": task_cfg.cost_b,
        "route_to_c": task_cfg.cost_c,
    }
    selected_cost = costs.get(action_str, 0.0)
    budget_dollars = float(observation.budget_remaining) * float(task_cfg.initial_budget)
    if selected_cost >= budget_dollars - 1e-9:
        return "shed_load"
    return action_str


def run_heuristic_episode(task_cfg: TaskConfig, seed: int) -> dict[str, Any]:
    env = BudgetRouterEnv()
    obs = env.reset(seed=seed, scenario=task_cfg)
    total_reward = 0.0
    while not obs.done:
        obs = env.step(heuristic_baseline_policy(obs))
        total_reward += float(obs.reward or 0.0)
    grader = grade_episode(env._internal.history)
    metrics = episode_metrics(env._internal.history)
    return {
        "grader_score": float(grader["overall_score"]),
        "total_reward": total_reward,
        "episode_length": env._internal.current_step,
        "grader": grader,
        "metrics": metrics,
    }


class SFTPolicy:
    def __init__(self, model_repo: str, *, token: str | None, use_budget_guard: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=dtype, token=token)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_budget_guard = use_budget_guard
        self.messages: list[dict[str, str]] = []
        self.previous_obs: Observation | None = None
        self.previous2_obs: Observation | None = None
        self.parse_failures = 0

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.previous_obs = None
        self.previous2_obs = None
        self.parse_failures = 0

    def choose_action(self, obs: Observation, *, task_name: str, task_cfg: TaskConfig) -> str:
        import torch

        obs_text = format_observation_for_sft(
            obs=obs,
            task_name=task_name,
            previous_obs=self.previous_obs,
            previous2_obs=self.previous2_obs,
        )
        self.messages.append({"role": "user", "content": obs_text})
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        action_str, ok = parse_action(generated)
        if not ok:
            self.parse_failures += 1
        if self.use_budget_guard:
            action_str = apply_budget_safety_guard(action_str, obs, task_cfg)
        self.messages.append({"role": "assistant", "content": action_str})
        self.previous2_obs = self.previous_obs
        self.previous_obs = obs
        return action_str


def run_sft_episode(policy: SFTPolicy, task_name: str, task_cfg: TaskConfig, seed: int) -> dict[str, Any]:
    env = BudgetRouterEnv()
    policy.reset()
    obs = env.reset(seed=seed, scenario=task_cfg)
    total_reward = 0.0
    actions: list[str] = []
    while not obs.done:
        action_str = policy.choose_action(obs, task_name=task_name, task_cfg=task_cfg)
        actions.append(action_str)
        obs = env.step(Action(action_type=ActionType(action_str)))
        total_reward += float(obs.reward or 0.0)
    grader = grade_episode(env._internal.history)
    metrics = episode_metrics(env._internal.history)
    return {
        "grader_score": float(grader["overall_score"]),
        "total_reward": total_reward,
        "episode_length": env._internal.current_step,
        "grader": grader,
        "metrics": metrics,
        "actions": actions,
        "parse_failures": policy.parse_failures,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)))


def compute_paired_stats(heuristic_scores: list[float], sft_scores: list[float]) -> dict[str, Any]:
    if len(heuristic_scores) != len(sft_scores):
        raise ValueError("Paired stats require equal-length score lists.")
    if not heuristic_scores:
        raise ValueError("No scores provided.")

    diffs = [s - h for h, s in zip(heuristic_scores, sft_scores)]
    n = len(diffs)
    delta = _mean(diffs)
    std_diff = _sample_std(diffs)
    if std_diff == 0.0:
        t_stat = math.inf if delta > 0 else (-math.inf if delta < 0 else 0.0)
        p_val = 0.0 if delta > 0 else 1.0
        cohens_d = math.inf if delta > 0 else (-math.inf if delta < 0 else 0.0)
    else:
        try:
            from scipy import stats

            t_stat, p_val = stats.ttest_rel(sft_scores, heuristic_scores, alternative="greater")
            cohens_d = delta / std_diff
        except Exception:
            t_stat = delta / (std_diff / math.sqrt(n))
            p_val = float("nan")
            cohens_d = delta / std_diff

    return {
        "n_seeds": n,
        "mean_heuristic": _mean(heuristic_scores),
        "mean_sft": _mean(sft_scores),
        "std_heuristic": _sample_std(heuristic_scores),
        "std_sft": _sample_std(sft_scores),
        "delta": delta,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "cohens_d": float(cohens_d),
        "significant": bool(delta > 0 and p_val < 0.05),
        "wins": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
    }


def _ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = _mean(values)
    if n < 2:
        return mean, mean
    se = _sample_std(values) / math.sqrt(n)
    try:
        from scipy import stats

        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        return float(lo), float(hi)
    except Exception:
        return mean - 1.96 * se, mean + 1.96 * se


def _parse_seed_values(value: str | None, n_seeds: int) -> list[int]:
    if value:
        return [int(part) for part in value.replace(",", " ").split()]
    return list(range(300, 300 + n_seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT Budget Router model.")
    parser.add_argument("--model-repo", default=os.getenv("SFT_MODEL_REPO", DEFAULT_MODEL_REPO))
    parser.add_argument("--task", default=os.getenv("TASK_NAME", "hard_multi"), choices=sorted(TASK_PRESETS))
    parser.add_argument("--n-seeds", type=int, default=int(os.getenv("N_SEEDS", "10")))
    parser.add_argument("--seed-values", default=os.getenv("EVAL_SEED_VALUES"))
    parser.add_argument("--output-json", default=os.getenv("EVAL_OUTPUT_JSON", "eval_results_sft.json"))
    parser.add_argument("--no-budget-guard", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    task_cfg = TASK_PRESETS[args.task]
    seeds = _parse_seed_values(args.seed_values, args.n_seeds)
    # #region agent log
    _agent_debug_ndjson(
        {
            "sessionId": "e4cac3",
            "runId": os.environ.get("DEBUG_RUN_ID", "eval-import-fix"),
            "hypothesisId": "H1",
            "location": "eval_sft.py:main",
            "message": "eval_startup",
            "data": {
                "system_prompt_source": _SYSTEM_PROMPT_SOURCE,
                "model_repo": args.model_repo,
                "task": args.task,
                "n_seeds": len(seeds),
            },
            "timestamp": int(time.time() * 1000),
        }
    )
    # #endregion
    policy = SFTPolicy(args.model_repo, token=token, use_budget_guard=not args.no_budget_guard)

    episodes: list[dict[str, Any]] = []
    heuristic_scores: list[float] = []
    sft_scores: list[float] = []
    for seed in seeds:
        heuristic_ep = run_heuristic_episode(task_cfg, seed)
        sft_ep = run_sft_episode(policy, args.task, task_cfg, seed)
        heuristic_scores.append(float(heuristic_ep["grader_score"]))
        sft_scores.append(float(sft_ep["grader_score"]))
        episodes.append({"seed": seed, "heuristic": heuristic_ep, "sft": sft_ep})
        print(
            f"[eval-sft] seed={seed} heuristic={heuristic_ep['grader_score']:.4f} "
            f"sft={sft_ep['grader_score']:.4f} delta={sft_ep['grader_score'] - heuristic_ep['grader_score']:+.4f} "
            f"parse_failures={sft_ep['parse_failures']}",
            flush=True,
        )

    stats = compute_paired_stats(heuristic_scores, sft_scores)
    heu_ci = _ci95(heuristic_scores)
    sft_ci = _ci95(sft_scores)
    result = {
        **stats,
        "task": args.task,
        "seeds": seeds,
        "heuristic_scores": heuristic_scores,
        "sft_scores": sft_scores,
        "heuristic_ci95": heu_ci,
        "sft_ci95": sft_ci,
        "budget_guard": not args.no_budget_guard,
        "episodes": episodes,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("| Policy | Mean | Std | 95% CI | vs Heuristic |")
    print("|---|---:|---:|---|---:|")
    print(
        f"| Heuristic | {stats['mean_heuristic']:.3f} | {stats['std_heuristic']:.3f} | "
        f"[{heu_ci[0]:.3f}, {heu_ci[1]:.3f}] | baseline |"
    )
    print(
        f"| SFT | {stats['mean_sft']:.3f} | {stats['std_sft']:.3f} | "
        f"[{sft_ci[0]:.3f}, {sft_ci[1]:.3f}] | {stats['delta']:+.3f} |"
    )
    verdict = "SIGNIFICANT" if stats["significant"] else "NOT SIGNIFICANT"
    print(
        f"SFT: {stats['mean_sft']:.3f} vs Heuristic: {stats['mean_heuristic']:.3f} | "
        f"delta={stats['delta']:+.3f} | t({stats['n_seeds'] - 1})={stats['t_stat']:.2f}, "
        f"p={stats['p_val']:.4f} | {verdict} | Cohen's d={stats['cohens_d']:.2f} | "
        f"wins/ties/losses={stats['wins']}/{stats['ties']}/{stats['losses']}"
    )

    if not args.no_upload:
        if not token:
            raise RuntimeError("HF_TOKEN must be set to upload eval JSON. Use --no-upload to skip.")
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=args.output_json,
            path_in_repo=Path(args.output_json).name,
            repo_id=args.model_repo,
            repo_type="model",
            token=token,
        )
        print(f"[eval-sft] uploaded {args.output_json} to {args.model_repo}", flush=True)


if __name__ == "__main__":
    main()
