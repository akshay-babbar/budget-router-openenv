from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openenv_core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


# =============================================================================
# Action — extends OpenEnv Action
# =============================================================================


class ActionType(str, Enum):
    """The four possible routing actions."""

    ROUTE_TO_A = "route_to_a"
    ROUTE_TO_B = "route_to_b"
    ROUTE_TO_C = "route_to_c"
    SHED_LOAD = "shed_load"


@dataclass(kw_only=True)
class Action(BaseAction):
    """
    Agent action: route a request to a provider or shed load.

    Extends OpenEnv Action (which provides `metadata` field).
    """

    action_type: Literal["route_to_a", "route_to_b", "route_to_c", "shed_load"]

    def __post_init__(self) -> None:
        if isinstance(self.action_type, str):
            self.action_type = ActionType(self.action_type)


# =============================================================================
# Observation — extends OpenEnv Observation
# =============================================================================


@dataclass(kw_only=True)
class Observation(BaseObservation):
    """
    Agent-visible observation. ALL numeric fields are normalized to [0.0, 1.0].

    Extends OpenEnv Observation (which provides `done`, `reward`, `metadata` fields).
    """

    # Provider health (recent success rates)
    provider_a_status: float
    provider_b_status: float
    provider_c_status: float

    # Resource state
    budget_remaining: float
    queue_backlog: float
    system_latency: float

    # Episode progress
    step_count: float

    def __post_init__(self) -> None:
        for field_name in (
            "provider_a_status",
            "provider_b_status",
            "provider_c_status",
            "budget_remaining",
            "queue_backlog",
            "system_latency",
            "step_count",
        ):
            setattr(self, field_name, max(0.0, min(1.0, getattr(self, field_name))))


# =============================================================================
# Internal State (raw units, for debugging / trace only)
# =============================================================================


@dataclass
class ProviderState:
    """Internal state of a single provider in raw units."""

    name: str
    base_reliability: float  # initial reliability [0, 1]
    current_health: float  # current health [0, 1]
    cost_per_request: float  # dollars
    base_latency_ms: float  # base latency in ms
    total_requests: int = 0
    successful_requests: int = 0

    @property
    def observed_success_rate(self) -> float:
        """Success rate from agent's perspective (windowed)."""
        if self.total_requests == 0:
            return self.base_reliability
        return self.successful_requests / self.total_requests


@dataclass
class InternalState:
    """
    Full internal state in raw units. NOT exposed to the agent.
    Used for manual trace, debugging, and the oracle policy.
    """

    providers: Dict[str, ProviderState] = field(default_factory=dict)
    budget_dollars: float = 0.0
    initial_budget_dollars: float = 0.0
    queue_backlog_count: int = 0
    max_queue_backlog: int = 10
    last_latency_ms: float = 0.0
    sla_ceiling_ms: float = 500.0
    current_step: int = 0
    max_steps: int = 20
    episode_done: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Windowed success tracking (last N requests per provider)
    provider_window: Dict[str, List[bool]] = field(default_factory=dict)
    window_size: int = 5

    def get_windowed_success_rate(self, provider_name: str) -> float:
        """Get success rate over the last `window_size` requests for a provider."""
        window = self.provider_window.get(provider_name, [])
        if not window:
            return self.providers[provider_name].base_reliability
        return sum(window) / len(window)


# =============================================================================
# Task Configuration
# =============================================================================


@dataclass
class TaskConfig:
    """
    Configuration for a task scenario. Passed to reset(scenario=config).
    NOT a subclass — just a data container.
    """

    name: str
    description: str

    # Budget
    initial_budget: float = 5.0  # dollars

    # Provider costs (per request, dollars)
    cost_a: float = 0.01
    cost_b: float = 0.05
    cost_c: float = 0.10

    # Provider base reliability
    reliability_a: float = 0.70
    reliability_b: float = 0.90
    reliability_c: float = 0.99

    # Provider base latency (ms)
    latency_a: float = 100.0
    latency_b: float = 150.0
    latency_c: float = 200.0

    # SLA
    sla_ceiling_ms: float = 500.0

    # Degradation config (primary)
    degradation_start_step: int = 0  # step at which degradation begins
    degradation_rate: float = 0.0  # health reduction per step for provider A
    degradation_target: str = "A"  # which provider degrades

    # Secondary degradation (for multi-provider scenarios)
    secondary_degradation_start_step: int = 999  # 999 = no secondary degradation
    secondary_degradation_rate: float = 0.0
    secondary_degradation_target: str = ""  # empty = no secondary degradation

    # Episode
    max_steps: int = 20
    max_queue_backlog: int = 10

    # Stochastic noise
    latency_noise_std: float = 30.0  # ms std dev added to base latency


# =============================================================================
# OpenEnv State — extends BaseState
# =============================================================================


@dataclass
class EnvState(BaseState):
    """
    OpenEnv-compatible state object returned by the `state` property.
    Extends BaseState (which provides `episode_id`, `step_count` fields).
    """

    scenario_name: str = ""
    is_done: bool = False
