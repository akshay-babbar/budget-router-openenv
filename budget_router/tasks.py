"""
Task preset configurations: EASY, MEDIUM, HARD.

Each is a TaskConfig instance passed to reset(scenario=config).
"""

from .models import TaskConfig


EASY = TaskConfig(
    name="easy",
    description="Stable providers. Cheapest valid routing wins.",
    initial_budget=1.0,
    cost_a=0.01,
    cost_b=0.05,
    cost_c=0.10,
    reliability_a=0.85,           # bumped so cheapest path is viable
    reliability_b=0.92,
    reliability_c=0.99,
    latency_a=100.0,
    latency_b=150.0,
    latency_c=200.0,
    sla_ceiling_ms=500.0,
    degradation_start_step=999,   # effectively no degradation
    degradation_rate=0.0,
    degradation_target="A",
    max_steps=20,
    max_queue_backlog=10,
    latency_noise_std=30.0,
)


MEDIUM = TaskConfig(
    name="medium",
    description="Provider A degrades sharply after step 5. Must adapt routing.",
    initial_budget=0.95,
    cost_a=0.01,
    cost_b=0.05,
    cost_c=0.10,
    reliability_a=0.85,
    reliability_b=0.92,
    reliability_c=0.99,
    latency_a=100.0,
    latency_b=150.0,
    latency_c=200.0,
    sla_ceiling_ms=500.0,
    degradation_start_step=5,
    degradation_rate=0.15,        # sharp drop after step 5
    degradation_target="A",
    max_steps=20,
    max_queue_backlog=10,
    latency_noise_std=30.0,
)


HARD = TaskConfig(
    name="hard",
    description="Provider A degrades every step. Tight budget. Must diversify early.",
    initial_budget=0.9,
    cost_a=0.01,
    cost_b=0.05,
    cost_c=0.10,
    reliability_a=0.85,
    reliability_b=0.92,
    reliability_c=0.99,
    latency_a=100.0,
    latency_b=150.0,
    latency_c=200.0,
    sla_ceiling_ms=500.0,
    degradation_start_step=0,     # degrades from the start
    degradation_rate=0.08,        # faster degradation for consistency
    degradation_target="A",
    max_steps=20,
    max_queue_backlog=10,
    latency_noise_std=40.0,       # more noise
)


TASK_PRESETS = {"easy": EASY, "medium": MEDIUM, "hard": HARD}
