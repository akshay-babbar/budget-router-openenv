"""
BudgetRouterGRPOEnv — TRL environment_factory-compatible class for GRPO training.

Usage with TRL GRPOTrainer:
    from datasets import Dataset
    from train.grpo_env import BudgetRouterGRPOEnv
    from budget_router.reward import grade_episode

    # Dataset: columns become **kwargs in reset(). "prompt" drives the model's initial message.
    dataset = Dataset.from_list([
        {"prompt": [[{"role": "user", "content": "Route requests using the available tools."}]],
         "scenario": "hard_multi", "seed": i}
        for i in range(200)
    ])

    # reward_funcs with an `environments` parameter is the CORRECT TRL pattern when using
    # environment_factory. TRL inspects the signature and passes env instances (not completions).
    # This is explicitly documented in the official TRL/OpenEnv integration guide.
    # Alternatively, env.reward is set on the instance and TRL reads it directly if
    # reward_funcs is omitted — but the explicit function gives more control.
    def reward_func(environments, **kwargs):
        return [float(grade_episode(env._env._internal.history)["overall_score"])
                for env in environments]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=GRPOConfig(num_generations=8, max_completion_length=2048),
        environment_factory=BudgetRouterGRPOEnv,
    )

Design Constraints (do NOT violate):
  - Tool methods MUST call self._env.step() and never construct custom step_info dicts.
    environment.py writes actual_degradation_start (jittered per-episode) into step_info.
    grade_episode() reads degradation_start_step from that dict to compute adaptation windows.
    Custom dicts would write the config constant (e.g. 0) instead of the jittered value,
    silently corrupting adaptation scores with no crash.
  - History is authoritative at self._env._internal.history — never maintain a separate copy.
  - Reward is computed once at episode end via grade_episode()["overall_score"] (float in [0,1]).
  - Raise ValueError("Episode complete.") when done — TRL catches this and ends the rollout.

Mac / MPS Notes:
  - Unsloth does NOT support Mac for training (CUDA-only as of Apr 2026).
  - Use TRL + PyTorch MPS: no load_in_4bit, no vLLM, no paged_adamw_8bit.
  - PYTORCH_ENABLE_MPS_FALLBACK=1 required for ops not yet on Metal.
  - Recommended models for Mac: Qwen2.5-1.5B (fits 8GB+), Qwen2.5-3B (fits 16GB+).
  - For Colab/cloud: Unsloth + vLLM work normally on NVIDIA T4/A100.

Reward Variance Note:
  - GRPO gradient = 0 when group_std ≈ 0. Use hard_multi scenario (not easy).
  - hard_multi has jitter + dual degradation → wider inter-rollout score spread.
  - num_generations=8 (not 4) recommended to get better group variance estimates.
"""

from __future__ import annotations

from typing import Optional

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.reward import grade_episode
from budget_router.tasks import HARD_MULTI, TASK_PRESETS


class BudgetRouterGRPOEnv:
    """
    TRL environment_factory-compatible wrapper around BudgetRouterEnv.

    Exposes four named tool methods: route_to_a, route_to_b, route_to_c, shed_load.
    The LLM calls these via function-calling. TRL discovers them automatically.

    Episode lifecycle:
        1. reset(**kwargs) → returns rich text observation (initial state).
        2. Model calls tool methods N times until episode ends.
        3. Tool method raises ValueError("Episode complete.") when obs.done is True.
        4. TRL reads self.reward from the reward_func after the episode.
    """

    def __init__(self) -> None:
        self._env = BudgetRouterEnv()
        self.reward: float = 0.0

    # ─── TRL lifecycle ──────────────────────────────────────────────────

    def reset(self, **kwargs) -> str:
        """
        Reset the environment. Called by TRL at the start of each episode.

        Accepts dataset columns as kwargs:
            scenario (str): one of "easy", "medium", "hard", "hard_multi" (default).
            seed (int): optional fixed seed for reproducibility.

        Returns:
            str: Initial observation text including provider status, budget, and task brief.
        """
        scenario_name = str(kwargs.get("scenario", "hard_multi"))
        scenario = TASK_PRESETS.get(scenario_name, HARD_MULTI)
        seed: Optional[int] = kwargs.get("seed", None)
        if seed is not None:
            seed = int(seed)

        self._env.reset(seed=seed, scenario=scenario)
        self.reward = 0.0

        return self._format_observation(is_initial=True)

    # ─── Tool methods (TRL discovers all public non-reset methods) ───────

    def route_to_a(self) -> str:
        """
        Route the current request to Provider A ($0.01/req, cheapest, lowest base reliability).

        Args:
            (none)

        Returns:
            Outcome feedback: success/failure, latency, budget remaining, provider health update.
        """
        return self._step(ActionType.ROUTE_TO_A)

    def route_to_b(self) -> str:
        """
        Route the current request to Provider B ($0.05/req, balanced cost and reliability).

        Args:
            (none)

        Returns:
            Outcome feedback: success/failure, latency, budget remaining, provider health update.
        """
        return self._step(ActionType.ROUTE_TO_B)

    def route_to_c(self) -> str:
        """
        Route the current request to Provider C ($0.10/req, most expensive, highest base reliability).

        Args:
            (none)

        Returns:
            Outcome feedback: success/failure, latency, budget remaining, provider health update.
        """
        return self._step(ActionType.ROUTE_TO_C)

    def shed_load(self) -> str:
        """
        Shed the current request — reject it without routing to any provider.
        Use when all providers appear degraded or budget is critically low.
        Penalty: -0.5 step reward. Slightly reduces queue backlog.

        Args:
            (none)

        Returns:
            Outcome feedback: load shed confirmation, budget remaining, current state.
        """
        return self._step(ActionType.SHED_LOAD)

    # ─── Internal step dispatch ──────────────────────────────────────────

    def _step(self, action_type: ActionType) -> str:
        """
        Execute one environment step.

        CRITICAL: Delegates entirely to self._env.step(). Never constructs a custom
        step_info dict. environment.py writes actual_degradation_start (the jittered
        per-episode onset) into step_info; grade_episode() reads this to compute
        adaptation scores. A custom dict would use the config constant instead,
        silently breaking adaptation scoring.
        """
        if self._env._internal.episode_done:
            # Guard: called after done — reuse last reward, signal TRL to stop
            raise ValueError(
                f"Episode already complete. Final score: {self.reward:.3f}"
            )

        action = Action(action_type=action_type)
        obs = self._env.step(action)  # step_info written to self._env._internal.history

        # Format response text BEFORE checking done (obs fields still valid)
        response = self._format_step_result(obs)

        if obs.done:
            # History is authoritative at self._env._internal.history
            self.reward = float(
                grade_episode(self._env._internal.history)["overall_score"]
            )
            raise ValueError(
                f"Episode complete. Score: {self.reward:.3f}. {response}"
            )

        return response

    # ─── Observation / response formatters ──────────────────────────────

    def _format_observation(self, is_initial: bool = False) -> str:
        """Format current env state as a rich text observation string."""
        obs = self._env._get_obs()
        s = self._env._internal
        config = self._env._config

        steps_remaining = max(0, s.max_steps - s.current_step)
        budget_dollars = s.budget_dollars
        budget_pct = obs.budget_remaining * 100.0

        lines = []
        if is_initial:
            lines.append(
                f"=== Budget Router — {config.name.upper()} ===\n"
                f"Budget: ${budget_dollars:.3f} ({budget_pct:.1f}% remaining) | "
                f"Steps remaining: {steps_remaining}/{s.max_steps}\n"
                f"Providers: A=$0.01/req (cheapest), B=$0.05/req, C=$0.10/req (most reliable)\n"
                f"Goal: Maximize successful routed requests. Budget exhaustion = heavy penalty.\n"
            )
        else:
            lines.append(
                f"Budget: ${budget_dollars:.3f} ({budget_pct:.1f}%) | "
                f"Steps remaining: {steps_remaining}"
            )

        lines.append(
            f"Provider health (windowed success rate; 0.5 = unobserved):\n"
            f"  A: {obs.provider_a_status:.3f} | B: {obs.provider_b_status:.3f} | C: {obs.provider_c_status:.3f}\n"
            f"Queue backlog: {obs.queue_backlog:.3f} (normalized) | "
            f"System latency: {obs.system_latency:.3f} (normalized to SLA)\n"
        )

        if is_initial:
            lines.append(
                "Choose a routing action: route_to_a / route_to_b / route_to_c / shed_load"
            )

        return "\n".join(lines)

    def _format_step_result(self, obs) -> str:
        """Format step outcome as text returned to the model."""
        s = self._env._internal
        history = s.history
        if not history:
            return self._format_observation()

        last = history[-1]
        action_type = last.get("action_type", "unknown")
        succeeded = last.get("request_succeeded", False)
        provider = last.get("provider")
        latency = last.get("latency_ms", 0.0)
        cost = last.get("cost", 0.0)
        budget_exhausted = last.get("budget_exhausted", False)
        queue_overflow = last.get("queue_overflow", False)

        if action_type == "shed_load":
            result = "shed"
        elif budget_exhausted:
            result = "budget_exhausted"
        elif succeeded:
            result = "ok"
        else:
            result = "fail"

        overflow_note = " overflow=1" if queue_overflow else ""
        step_num = last.get("step", s.current_step)

        obs_obj = self._env._get_obs()
        budget_pct = obs_obj.budget_remaining * 100.0
        steps_remaining = max(0, s.max_steps - s.current_step)

        return (
            f"step={step_num} action={action_type} result={result} p={provider or '-'} "
            f"lat={latency:.0f} cost={cost:.3f} budget={budget_pct:.1f}% "
            f"steps_left={steps_remaining} health=A{obs_obj.provider_a_status:.2f}/"
            f"B{obs_obj.provider_b_status:.2f}/C{obs_obj.provider_c_status:.2f} "
            f"queue={obs_obj.queue_backlog:.2f}{overflow_note}"
        )
