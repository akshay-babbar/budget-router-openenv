from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

from budget_router.models import Action, ActionType, Observation
from inference import select_policy

from .config import PPO_MODEL_PATH

_PPO_MODEL: Any = None


class Policy(Protocol):
    def choose_action(self, obs: Dict) -> str:
        ...


class PolicyRunnerAdapter:
    def __init__(self, policy_name: str, policy_impl: object) -> None:
        self._policy_name = policy_name
        self._policy_impl = policy_impl

    def reset(self, scenario_name: str = "") -> None:
        reset_fn = getattr(self._policy_impl, "reset", None)
        if not callable(reset_fn):
            return
        try:
            reset_fn(task_name=scenario_name)
        except TypeError:
            reset_fn()

    def choose_action(self, obs: Dict) -> str:
        observation = _obs_to_observation(obs)
        if self._policy_name == "heuristic":
            action = self._policy_impl(observation)
        else:
            action = self._policy_impl.choose_action(observation)
        return action.action_type.value


def _obs_to_observation(obs: Dict) -> Observation:
    return Observation(
        provider_a_status=float(obs.get("provider_a_status", 0.0) or 0.0),
        provider_b_status=float(obs.get("provider_b_status", 0.0) or 0.0),
        provider_c_status=float(obs.get("provider_c_status", 0.0) or 0.0),
        budget_remaining=float(obs.get("budget_remaining", 0.0) or 0.0),
        queue_backlog=float(obs.get("queue_backlog", 0.0) or 0.0),
        system_latency=float(obs.get("system_latency", 0.0) or 0.0),
        step_count=float(obs.get("step_count", 0.0) or 0.0),
        done=bool(obs.get("done", False)),
        reward=float(obs.get("reward", 0.0) or 0.0),
        metadata=obs.get("metadata", {}) or {},
    )


def _format_policy_error(policy_name: str, error: str) -> str:
    if policy_name == "llm" and "API_BASE_URL" in error and "API_KEY" in error:
        return (
            "LLM auto-play is not enabled in this hosted Space. "
            "To try it, duplicate this Space to your own account and set "
            "API_BASE_URL, API_KEY, and optionally MODEL_NAME in that copy's Space settings."
        )
    return error


def get_policy_runner(policy_name: str) -> Tuple[Optional[Policy], Optional[str]]:
    if policy_name == "ppo":
        model, err = _load_ppo_model()
        if err:
            return None, err
        return PolicyRunnerAdapter(policy_name, PPOPolicy(model)), None
    try:
        return PolicyRunnerAdapter(policy_name, select_policy(policy_name)), None
    except Exception as exc:
        return None, _format_policy_error(policy_name, str(exc))


def _load_ppo_model() -> Tuple[Optional[Any], Optional[str]]:
    global _PPO_MODEL
    if _PPO_MODEL is not None:
        return _PPO_MODEL, None
    try:
        from pathlib import Path

        if not Path(PPO_MODEL_PATH).exists():
            return None, f"PPO model not found at {PPO_MODEL_PATH}."
        from stable_baselines3 import PPO  # type: ignore

        _PPO_MODEL = PPO.load(PPO_MODEL_PATH)
        return _PPO_MODEL, None
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "stable_baselines3":
            return None, "PPO policy requires stable_baselines3. Install the 'training' extra to enable it."
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)


class PPOPolicy:
    def __init__(self, model: Any):
        self._model = model

    def choose_action(self, observation: Observation) -> Action:
        try:
            import numpy as np  # type: ignore

            obs_arr = np.array(
                [
                    observation.provider_a_status,
                    observation.provider_b_status,
                    observation.provider_c_status,
                    observation.budget_remaining,
                    observation.queue_backlog,
                    observation.system_latency,
                    observation.step_count,
                ],
                dtype=np.float32,
            )
        except Exception:
            obs_arr = [
                observation.provider_a_status,
                observation.provider_b_status,
                observation.provider_c_status,
                observation.budget_remaining,
                observation.queue_backlog,
                observation.system_latency,
                observation.step_count,
            ]
        action_idx, _ = self._model.predict(obs_arr, deterministic=True)
        idx = int(action_idx)
        if idx == 0:
            at = ActionType.ROUTE_TO_A
        elif idx == 1:
            at = ActionType.ROUTE_TO_B
        elif idx == 2:
            at = ActionType.ROUTE_TO_C
        else:
            at = ActionType.SHED_LOAD
        return Action(action_type=at)
