from __future__ import annotations

from typing import Any, Dict

from budget_router.models import Observation


def fresh_side_state() -> Dict:
    return {
        "env": None,
        "policy_name": "",
        "policy_runner": None,
        "obs": {},
        "history": [],
        "cumulative_reward": 0.0,
        "step": 0,
        "done": False,
        "status": "",
    }


def _observation_to_dict(observation: Observation) -> Dict[str, Any]:
    return {
        "provider_a_status": float(observation.provider_a_status),
        "provider_b_status": float(observation.provider_b_status),
        "provider_c_status": float(observation.provider_c_status),
        "budget_remaining": float(observation.budget_remaining),
        "queue_backlog": float(observation.queue_backlog),
        "system_latency": float(observation.system_latency),
        "step_count": float(observation.step_count),
        "done": bool(getattr(observation, "done", False)),
        "reward": float(getattr(observation, "reward", 0.0) or 0.0),
        "metadata": dict(getattr(observation, "metadata", {}) or {}),
    }


def record_step(step: int, action: str, obs: Dict, reward: float, meta: Dict) -> Dict:
    return {
        "step": step,
        "action": action,
        "health_a": obs.get("provider_a_status", 0),
        "health_b": obs.get("provider_b_status", 0),
        "health_c": obs.get("provider_c_status", 0),
        "budget": obs.get("budget_remaining", 0),
        "reward": reward,
        "meta_raw": dict(meta or {}),
        "succeeded": meta.get("request_succeeded", False),
        "cost": meta.get("cost", 0.0),
        "latency_ms": meta.get("latency_ms", 0.0),
        "sla_ceiling_ms": meta.get("sla_ceiling_ms", 500.0),
        "initial_budget": meta.get("initial_budget", 1.0),
        "degradation_start_step": meta.get("degradation_start_step", 999),
        "secondary_degradation_start_step": meta.get("secondary_degradation_start_step"),
    }
