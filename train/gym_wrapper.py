"""
Gymnasium wrapper for BudgetRouterEnv.

Wraps a single scenario (default: Easy) into a standard Gymnasium interface
compatible with stable-baselines3 and other RL libraries.

Observation space: Box(7,) float32 — all fields normalized to [0, 1]
    [provider_a_status, provider_b_status, provider_c_status,
     budget_remaining, queue_backlog, system_latency, step_count]

Action space: Discrete(4)
    0 → route_to_a
    1 → route_to_b
    2 → route_to_c
    3 → shed_load
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.tasks import EASY, TaskConfig


_ACTION_MAP = [
    ActionType.ROUTE_TO_A,
    ActionType.ROUTE_TO_B,
    ActionType.ROUTE_TO_C,
    ActionType.SHED_LOAD,
]


class BudgetRouterGymEnv(gym.Env):
    """Gymnasium-compatible wrapper around BudgetRouterEnv.

    Args:
        scenario: TaskConfig to use. Defaults to EASY.
        seed: Fixed seed for reproducible resets. If None, a random seed is
              drawn from the Gymnasium RNG on each reset().
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario: TaskConfig = EASY, seed: int | None = None) -> None:
        super().__init__()
        self._env = BudgetRouterEnv()
        self._scenario = scenario
        self._fixed_seed = seed
        self._episode_seed: int | None = seed

        # 7-dimensional normalized observation
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # 4 discrete actions: route_to_a, route_to_b, route_to_c, shed_load
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Use fixed seed if provided at construction, otherwise use Gymnasium's RNG
        if self._fixed_seed is not None:
            episode_seed = self._fixed_seed
        elif seed is not None:
            episode_seed = seed
        else:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))

        self._episode_seed = episode_seed
        obs = self._env.reset(seed=episode_seed, scenario=self._scenario)
        return self._obs_to_array(obs), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_type = _ACTION_MAP[int(action)]
        obs = self._env.step(Action(action_type=action_type))
        reward = float(obs.reward or 0.0)
        terminated = bool(obs.done)
        truncated = False  # termination handled by env's max_steps
        info = dict(obs.metadata or {})
        return self._obs_to_array(obs), reward, terminated, truncated, info

    def render(self) -> None:
        pass  # no rendering required

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_to_array(obs) -> np.ndarray:
        """Convert Observation dataclass to a (7,) float32 numpy array."""
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
