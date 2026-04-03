"""
Budget Router Environment — Core RL environment.

Extends openenv-core Environment base class with the standard
reset(), step(), state interface. Processes one request per step
through 3 providers under budget, latency, reliability, and
degradation constraints.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from typing import Any, Dict, Optional, Tuple

from openenv_core.env_server import Environment
from openenv_core.env_server.types import Action as OpenEnvAction

from .models import (
    Action,
    ActionType,
    EnvState,
    InternalState,
    Observation,
    ProviderState,
    TaskConfig,
)
from .reward import step_reward
from .tasks import EASY

BACKLOG_LATENCY_PER_ITEM_MS = 8.0


class BudgetRouterEnv(Environment):
    """
    Incident Commander for Budgeted Tool/API Reliability.

    An agent routes incoming requests to one of 3 providers (A, B, C)
    or sheds load, under budget, latency, and reliability constraints.

    Extends OpenEnv Environment base class with proper type parameters.

    Interface:
        reset(seed, scenario) -> Observation
        step(action) -> Observation  (reward in obs.reward, done in obs.done)
        state -> EnvState
    """

    def __init__(self, emit_structured_logs: bool = False) -> None:
        super().__init__()
        self._internal: InternalState = InternalState()
        self._config: TaskConfig = EASY
        self._rng: random.Random = random.Random()
        self._episode_id: str = ""
        self._cumulative_reward: float = 0.0
        self._emit_structured_logs = emit_structured_logs
        self._episode_number = 0
        self._current_seed: Optional[int] = None

    def _emit_log(self, prefix: str, payload: Dict[str, Any]) -> None:
        if self._emit_structured_logs:
            print(f"{prefix} {json.dumps(payload)}", flush=True)

    def _observation_payload(self, observation: Observation) -> Dict[str, float]:
        return {
            "provider_a_status": float(observation.provider_a_status),
            "provider_b_status": float(observation.provider_b_status),
            "provider_c_status": float(observation.provider_c_status),
            "budget_remaining": float(observation.budget_remaining),
            "queue_backlog": float(observation.queue_backlog),
            "system_latency": float(observation.system_latency),
            "step_count": float(observation.step_count),
        }

    # ─── OpenEnv interface ──────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: Optional[TaskConfig] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment to initial state."""
        config = scenario or kwargs.get("scenario", EASY)
        if isinstance(config, str):
            from .tasks import TASK_PRESETS

            config = TASK_PRESETS.get(config, EASY)
        self._config = config

        # Seed the RNG
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        self._episode_id = episode_id or str(uuid.uuid4())
        self._episode_number += 1
        self._current_seed = seed
        self._cumulative_reward = 0.0

        # Initialize providers
        providers = {
            "A": ProviderState(
                name="A",
                base_reliability=config.reliability_a,
                current_health=config.reliability_a,
                cost_per_request=config.cost_a,
                base_latency_ms=config.latency_a,
            ),
            "B": ProviderState(
                name="B",
                base_reliability=config.reliability_b,
                current_health=config.reliability_b,
                cost_per_request=config.cost_b,
                base_latency_ms=config.latency_b,
            ),
            "C": ProviderState(
                name="C",
                base_reliability=config.reliability_c,
                current_health=config.reliability_c,
                cost_per_request=config.cost_c,
                base_latency_ms=config.latency_c,
            ),
        }

        self._internal = InternalState(
            providers=providers,
            budget_dollars=config.initial_budget,
            initial_budget_dollars=config.initial_budget,
            queue_backlog_count=0,
            max_queue_backlog=config.max_queue_backlog,
            last_latency_ms=config.latency_a,  # initial non-zero latency
            sla_ceiling_ms=config.sla_ceiling_ms,
            current_step=0,
            max_steps=config.max_steps,
            episode_done=False,
            history=[],
            provider_window={"A": [], "B": [], "C": []},
            window_size=5,
        )

        observation = self._get_obs()
        self._emit_log(
            "[START]",
            {
                "task": self._config.name,
                "seed": int(seed) if seed is not None else -1,
                "episode": self._episode_number,
            },
        )
        return observation

    def step(
        self,
        action: OpenEnvAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute one step: route a request or shed load.

        Returns:
            Observation with reward set, done flag, and metadata dict.
        """
        if self._internal.episode_done:
            # Already done — return terminal observation
            obs = self._get_obs()
            obs.done = True
            obs.reward = 0.0
            return obs

        if not isinstance(action, Action):
            action = Action(
                action_type=getattr(action, "action_type"),
                metadata=getattr(action, "metadata", {}),
            )

        self._internal.current_step += 1
        action_type = action.action_type.value

        # ── Apply degradation BEFORE processing the request ──
        self._degrade()

        # ── Process the action ──
        step_info: Dict[str, Any] = {
            "step": self._internal.current_step,
            "action_type": action_type,
            "sla_ceiling_ms": self._config.sla_ceiling_ms,
            "initial_budget": self._internal.initial_budget_dollars,
            "degradation_start_step": self._config.degradation_start_step,
        }

        if action_type == "shed_load":
            # Shed load: no routing, flat penalty
            reward = step_reward(
                action_type="shed_load",
                request_succeeded=False,
                provider_cost=0.0,
                initial_budget=self._internal.initial_budget_dollars,
                latency_ms=0.0,
                sla_ceiling_ms=self._config.sla_ceiling_ms,
            )
            # Queue pressure decreases slightly when shedding
            self._internal.queue_backlog_count = max(
                0, self._internal.queue_backlog_count - 1
            )
            # Latency set to 0 for shed (no request processed)
            self._internal.last_latency_ms = 0.0

            step_info.update(
                {
                    "request_succeeded": False,
                    "cost": 0.0,
                    "latency_ms": 0.0,
                    "reward": reward,
                    "provider": None,
                    "queue_overflow": False,
                }
            )

        else:
            # Route to a provider
            provider_name = {"route_to_a": "A", "route_to_b": "B", "route_to_c": "C"}[
                action_type
            ]
            provider = self._internal.providers[provider_name]

            # Deduct cost
            cost = provider.cost_per_request
            self._internal.budget_dollars -= cost

            # Check budget exhaustion
            if self._internal.budget_dollars <= 0:
                self._internal.budget_dollars = max(0.0, self._internal.budget_dollars)
                # Terminal penalty
                reward = -10.0
                self._internal.episode_done = True
                self._internal.last_latency_ms = 0.0

                step_info.update(
                    {
                        "request_succeeded": False,
                        "cost": cost,
                        "latency_ms": 0.0,
                        "reward": reward,
                        "provider": provider_name,
                        "queue_overflow": False,
                        "budget_exhausted": True,
                    }
                )

                self._internal.history.append(step_info)
                self._cumulative_reward += reward

                obs = self._get_obs()
                obs.done = True
                obs.reward = reward
                obs.metadata = step_info
                self._emit_log(
                    "[STEP]",
                    {
                        "step": self._internal.current_step,
                        "action": action_type,
                        "reward": float(reward),
                        "done": bool(obs.done),
                        "observation": self._observation_payload(obs),
                    },
                )
                self._emit_log(
                    "[END]",
                    {
                        "task": self._config.name,
                        "seed": int(self._current_seed) if self._current_seed is not None else -1,
                        "episode": self._episode_number,
                        "total_reward": round(float(self._cumulative_reward), 4),
                        "grade": 0.0,
                    },
                )
                return obs

            # Determine if request succeeds (based on current_health)
            request_succeeded = self._rng.random() < provider.current_health
            provider.total_requests += 1

            # Update windowed tracking
            window = self._internal.provider_window[provider_name]
            window.append(request_succeeded)
            if len(window) > self._internal.window_size:
                window.pop(0)

            if request_succeeded:
                provider.successful_requests += 1

            # Compute latency
            base_lat = provider.base_latency_ms
            noise = self._rng.gauss(0, self._config.latency_noise_std)
            # Queue backlog amplifies latency multiplicatively.
            # At max backlog (norm=1.0), latency increases by 50%.
            # This makes queue_backlog a causally relevant observation
            # by indirectly coupling it to reward via SLA breaches.
            queue_norm = (
                self._internal.queue_backlog_count / self._internal.max_queue_backlog
                if self._internal.max_queue_backlog > 0 else 0.0
            )
            backlog_amplifier = 1.0 + 0.5 * queue_norm
            # Failed requests have higher latency (timeout-like behavior)
            if not request_succeeded:
                actual_latency = (base_lat + abs(noise) + 200.0) * backlog_amplifier
            else:
                actual_latency = max(10.0, (base_lat + noise) * backlog_amplifier)
            self._internal.last_latency_ms = actual_latency

            # Queue backlog: failures increase pressure
            queue_overflow = False
            if not request_succeeded:
                self._internal.queue_backlog_count = min(
                    self._internal.max_queue_backlog,
                    self._internal.queue_backlog_count + 2,
                )
                if (
                    self._internal.queue_backlog_count
                    >= self._internal.max_queue_backlog
                ):
                    queue_overflow = True
            else:
                # Successful request drains queue slightly
                self._internal.queue_backlog_count = max(
                    0, self._internal.queue_backlog_count - 1
                )

            # Compute reward
            reward = step_reward(
                action_type=action_type,
                request_succeeded=request_succeeded,
                provider_cost=cost,
                initial_budget=self._internal.initial_budget_dollars,
                latency_ms=actual_latency,
                sla_ceiling_ms=self._config.sla_ceiling_ms,
            )

            step_info.update(
                {
                    "request_succeeded": request_succeeded,
                    "cost": cost,
                    "latency_ms": round(actual_latency, 2),
                    "reward": reward,
                    "provider": provider_name,
                    "queue_overflow": queue_overflow,
                }
            )

        # ── Record history ──
        self._internal.history.append(step_info)
        self._cumulative_reward += reward

        # ── Check episode end ──
        if self._internal.current_step >= self._internal.max_steps:
            self._internal.episode_done = True

        # ── Build observation ──
        obs = self._get_obs()
        obs.done = self._internal.episode_done
        obs.reward = reward
        obs.metadata = step_info

        self._emit_log(
            "[STEP]",
            {
                "step": self._internal.current_step,
                "action": action_type,
                "reward": float(reward),
                "done": bool(obs.done),
                "observation": self._observation_payload(obs),
            },
        )

        if obs.done:
            self._emit_log(
                "[END]",
                {
                    "task": self._config.name,
                    "seed": int(self._current_seed) if self._current_seed is not None else -1,
                    "episode": self._episode_number,
                    "total_reward": round(float(self._cumulative_reward), 4),
                    "grade": 0.0,
                },
            )

        return obs

    @property
    def state(self) -> EnvState:
        """OpenEnv-compatible state property."""
        return EnvState(
            episode_id=self._episode_id,
            step_count=self._internal.current_step,
            scenario_name=self._config.name,
            is_done=self._internal.episode_done,
        )

    # ─── Internal methods ──────────────────────────────────────────────

    def _get_obs(self) -> Observation:
        """Convert internal state to normalized [0,1] observation."""
        s = self._internal

        # Provider status: windowed success rate
        a_status = s.get_windowed_success_rate("A")
        b_status = s.get_windowed_success_rate("B")
        c_status = s.get_windowed_success_rate("C")

        # Budget: fraction remaining
        if s.initial_budget_dollars > 0:
            budget_frac = max(0.0, s.budget_dollars / s.initial_budget_dollars)
        else:
            budget_frac = 0.0

        # Queue backlog: normalized
        if s.max_queue_backlog > 0:
            queue_norm = s.queue_backlog_count / s.max_queue_backlog
        else:
            queue_norm = 0.0

        # Latency: normalized to SLA ceiling
        if s.sla_ceiling_ms > 0:
            latency_norm = s.last_latency_ms / s.sla_ceiling_ms
        else:
            latency_norm = 0.0

        # Step progress
        if s.max_steps > 0:
            step_norm = s.current_step / s.max_steps
        else:
            step_norm = 0.0

        return Observation(
            provider_a_status=max(0.0, min(1.0, a_status)),
            provider_b_status=max(0.0, min(1.0, b_status)),
            provider_c_status=max(0.0, min(1.0, c_status)),
            budget_remaining=max(0.0, min(1.0, budget_frac)),
            queue_backlog=max(0.0, min(1.0, queue_norm)),
            system_latency=max(0.0, min(1.0, latency_norm)),
            step_count=max(0.0, min(1.0, step_norm)),
        )

    def _degrade(self) -> None:
        """
        Apply stochastic degradation to configured provider(s).

        The target provider's health decreases based on:
        - degradation_rate from the TaskConfig
        - A small random perturbation
        - Only triggers after degradation_start_step
        Supports secondary degradation for multi-provider scenarios.
        """
        config = self._config
        step = self._internal.current_step

        # Primary degradation
        if step >= config.degradation_start_step:
            target = config.degradation_target
            provider = self._internal.providers.get(target)
            if provider is not None:
                noise = self._rng.gauss(0, 0.02)
                health_reduction = config.degradation_rate + noise
                provider.current_health = max(
                    0.05,
                    provider.current_health - health_reduction,
                )

        # Secondary degradation (for multi-provider scenarios)
        if (
            config.secondary_degradation_target
            and step >= config.secondary_degradation_start_step
        ):
            target = config.secondary_degradation_target
            provider = self._internal.providers.get(target)
            if provider is not None:
                noise = self._rng.gauss(0, 0.02)
                health_reduction = config.secondary_degradation_rate + noise
                provider.current_health = max(
                    0.05,
                    provider.current_health - health_reduction,
                )
