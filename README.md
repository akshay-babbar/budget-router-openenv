# Budget Router — Incident Commander for Budgeted Tool/API Reliability

An OpenEnv RL environment where an agent keeps a service healthy by routing incoming requests across three providers (A, B, C) under budget, latency, reliability, and degradation constraints. Each step processes one request: the agent chooses a provider or sheds load. Provider A is cheap but unreliable (and degrades over time), while Provider C is expensive but rock-solid. The agent must balance cost-efficiency against reliability to maximize service quality within a finite budget.

## Why This Is RL-Shaped (Not Just a Contextual Bandit)

This environment exhibits sequential dynamics that make it genuinely RL-shaped: Provider A's health degrades over time (creating non-stationary transition dynamics), budget depletes cumulatively (actions have long-term consequences), and queue backlog accumulates across steps (failures compound). An optimal policy must reason about future budget exhaustion and provider degradation — not just react to current observations. A contextual bandit would miss the temporal structure entirely.

## Observation Space

All values normalized to `[0.0, 1.0]`. Internal raw units available only in manual trace mode.

| Field              | Range  | Description                                 |
|--------------------|--------|---------------------------------------------|
| provider_a_status  | [0, 1] | Provider A windowed success rate            |
| provider_b_status  | [0, 1] | Provider B windowed success rate            |
| provider_c_status  | [0, 1] | Provider C windowed success rate            |
| budget_remaining   | [0, 1] | Fraction of initial budget remaining        |
| queue_backlog      | [0, 1] | Normalized failed/pending request pressure  |
| system_latency     | [0, 1] | Last step latency / SLA ceiling             |
| step_count         | [0, 1] | Normalized episode progress                 |

## Action Space

| Action      | Effect                                               |
|-------------|------------------------------------------------------|
| route_to_a  | Route to Provider A (cheapest, lowest reliability)   |
| route_to_b  | Route to Provider B (medium cost, medium reliability)|
| route_to_c  | Route to Provider C (most expensive, highest reliability) |
| shed_load   | Drop the request (prevents catastrophe, incurs -0.5) |

## Reward Logic

| Condition                      | Reward                            |
|--------------------------------|-----------------------------------|
| Successful request served      | +1.0                              |
| Failed request                 | -2.0                              |
| Routing cost                   | -(provider_cost / initial_budget) |
| Latency breach (above SLA)     | -(excess_latency / sla_ceiling)   |
| shed_load action               | -0.5 (replaces routing terms)     |
| Budget exhaustion (terminal)   | -10.0, then done=True             |

## Task Scenarios

| Task   | Description |
|--------|-------------|
| EASY   | All providers stable. Cheapest valid routing wins. Budget = $2.00. |
| MEDIUM | Provider A degrades sharply after step 5. Agent must detect and switch. Budget = $2.00. |
| HARD   | Provider A degrades every step from start. Must diversify routing proactively. Budget = $2.00. |

## How to Run

```bash
# Install with uv (Python 3.12)
uv venv --python 3.12
uv pip install -e ".[dev]"

# Run full validation suite
uv run python -m budget_router.validation

# Run tests
uv run pytest

# Run manual trace for debugging
uv run python -c "from budget_router.validation import run_manual_trace; run_manual_trace(seed=42, scenario_name='medium')"
```

## What "Working" Looks Like

All 29 hard assertions must pass. Key criteria:

- **Policy ordering**: `oracle >= baseline > random` on all tasks and both seed sets
- **Non-triviality**: baseline-random gap > 20% on at least one task
- **Solvability**: oracle achieves positive reward and >50% success rate on easy
- **Anti-gaming**: `always_route_a`, `always_shed_load` do NOT dominate the baseline
- **Held-out robustness**: baseline results stable within 30% margin across seed sets
- **Safety**: no NaN rewards, no episodes exceeding 20 steps, no budget explosions

## Why Degenerate Policies Don't Solve the Environment

- **always_route_a**: Cheap but A has only 85% base reliability and degrades on medium/hard. Suffers -2.0 per failure, which quickly overwhelms the cost savings. Catastrophic on medium/hard.
- **always_route_c**: Most reliable but costs $0.10/step. With $2.00 budget, can afford exactly 20 steps — but the terminal budget exhaustion penalty (-10.0) makes this barely viable. The heuristic baseline avoids this by routing cheaper providers when they're healthy.
- **always_shed_load**: Gets -0.5 × 20 = -10.0 every episode. Never earns the +1.0 success reward. Universally terrible.

The heuristic baseline succeeds by adapting: it routes cheaply when possible and escalates to more expensive providers only when observing degradation. This is the strategic advantage an RL agent should exploit.

## Architecture

Built on `openenv-core` v0.2 base classes:
- `Action` extends `openenv.core.env_server.types.Action`
- `Observation` extends `openenv.core.env_server.types.Observation`
- `EnvState` extends `openenv.core.env_server.types.State`
- `BudgetRouterEnv` extends `openenv.core.env_server.Environment[Action, Observation, EnvState]`

## Project Structure

```
budget_router/
├── __init__.py          # Package exports
├── models.py            # Pydantic v2: Action, Observation, State, TaskConfig
├── environment.py       # Core env: reset(), step(), state, _get_obs(), _degrade()
├── tasks.py             # TaskConfig presets: EASY, MEDIUM, HARD
├── reward.py            # step_reward(), episode_metrics()
├── policies.py          # random, heuristic, upper_bound, degenerate policies
├── validation.py        # run_validation(), run_manual_trace(), assert_all_checks()
├── requirements.txt     # Minimal deps
└── tests/
    ├── test_environment.py
    └── test_validation.py
```
