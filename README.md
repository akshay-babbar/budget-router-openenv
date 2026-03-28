# Budget Router — Incident Commander for Budgeted Tool/API Reliability

An OpenEnv environment that models a very common real-world incident response problem:

You operate a service that must route requests across multiple third-party providers (A/B/C). Providers differ in:

- **Cost** per request
- **Reliability** (success probability)
- **Latency**
- **Non-stationarity**: a provider can **degrade over time**

At each step you either route a request to a provider or shed load. The goal is to maximize service quality under budget and SLA constraints.

This repo is designed to be:

- **Deployable** (FastAPI + Docker + `openenv.yaml`)
- **Deterministic enough to benchmark** (seeded RNG, fixed tasks)
- **Hard to “win” with trivial degenerate policies** (explicit anti-gaming checks)

## Executive Intuition (for Staff/VP)

This is not a toy “pick the best arm” bandit. It is an incident-control problem with compounding dynamics:

- **Budget** is a hard, cumulative constraint.
- **Provider health degrades** (task-dependent), making the transition dynamics non-stationary.
- **Backlog is causally meaningful**: failures increase backlog, and backlog increases latency (tail-latency pressure).

The intended learning problem is: *detect degradation early, route to stabilize the system, and spend budget where it matters.*

```mermaid
flowchart LR
  A[Agent / Policy] -->|Action: route_to_a/b/c or shed_load| S[OpenEnv Server
FastAPI: server/app.py]
  S --> E[BudgetRouterEnv
budget_router/environment.py]
  E -->|Observation + reward + done| S
  S --> O[Agent / Policy]

  subgraph Env
    E --> P[Providers A/B/C
cost, reliability, latency]
    E --> B[Budget ($)]
    E --> Q[Queue backlog]
    E --> D[Degradation process]
  end
```

## OpenEnv Packaging (ground truth)

The OpenEnv manifest is:

- `openenv.yaml`:
  - `runtime: fastapi`
  - `app: server.app:app`
  - `port: 8000`

The FastAPI entrypoint is `server/app.py`, built via `openenv_core.env_server.create_app()`.

Docker is defined at `server/Dockerfile` (this layout is what `openenv validate` checks).

## API Surface (what exists in code)

The environment class is `BudgetRouterEnv` in `budget_router/environment.py`.

- **reset**: `reset(seed: Optional[int], scenario: Optional[TaskConfig], ...) -> Observation`
- **step**: `step(action: Action, ...) -> Observation` (reward in `obs.reward`, termination in `obs.done`)
- **state**: `state -> EnvState`

## Observation Space (agent-visible)

All observation values are normalized to `[0.0, 1.0]` in `Observation.__post_init__`.

| Field | Meaning (code-grounded) |
|------|--------------------------|
| `provider_a_status` | Windowed success rate for provider A (last `window_size=5` attempts) |
| `provider_b_status` | Windowed success rate for provider B |
| `provider_c_status` | Windowed success rate for provider C |
| `budget_remaining` | `budget_dollars / initial_budget_dollars` |
| `queue_backlog` | `queue_backlog_count / max_queue_backlog` |
| `system_latency` | `last_latency_ms / sla_ceiling_ms` |
| `step_count` | `current_step / max_steps` |

## Action Space

`ActionType` is an enum with exactly 4 actions:

- `route_to_a`
- `route_to_b`
- `route_to_c`
- `shed_load`

## POMDP Framing

The agent operates under **partial observability**: it sees windowed success rates (last 5 attempts per provider), not true health values. This is intentional — real incident responders also see symptoms (error rates), not root cause. The agent must infer degradation from the observation stream and adapt routing accordingly.

## Environment Dynamics (what actually happens per step)

```mermaid
sequenceDiagram
  participant Agent
  participant Env as BudgetRouterEnv.step()
  participant Deg as _degrade()
  participant Prov as Provider(A/B/C)

  Agent->>Env: Action(route_to_*) or shed_load
  Env->>Deg: Apply degradation (task-defined, supports multi-target)
  alt shed_load
    Env->>Env: backlog = max(0, backlog-1)
    Env->>Env: reward = -0.5
  else route_to_provider
    Env->>Env: budget -= provider.cost
    Env->>Prov: success ~ Bernoulli(current_health)
    Env->>Env: update windowed success tracking
    Env->>Env: backlog += 2 on failure; backlog -= 1 on success
    Env->>Env: latency amplified by backlog (multiplicative)
    Env->>Env: reward = step_reward(...)
  end
  Env-->>Agent: Observation + reward + done
```

### Backlog → Latency causality (multiplicative)

Backlog amplifies latency via multiplicative coupling, not just additive:

- `queue_norm = queue_backlog_count / max_queue_backlog`
- `backlog_amplifier = 1.0 + 0.5 * queue_norm`
- Latency is multiplied by `backlog_amplifier` (up to 1.5x when queue is full)
- On **failure**, an additional 200ms penalty is applied (representing retry overhead)

This creates a compounding feedback loop: failures → backlog → higher latency → more failures. The agent must break this cycle by routing to healthy providers.

## Reward (per-step, code-grounded)

Implemented in `budget_router/reward.py::step_reward()`:

- **Route actions**
  - `+1.0` on success
  - `-2.0` on failure
  - `-(provider_cost / initial_budget) * BUDGET_WEIGHT` cost penalty (BUDGET_WEIGHT=5.0)
  - if `latency_ms > sla_ceiling_ms`: `-(excess_latency / sla_ceiling_ms)`
- **shed_load**: fixed `-0.5` (replaces routing terms)
- **Budget exhaustion**: terminal `-10.0` and `done=True`

The 5x cost scaling ensures the cost signal (~0.05–0.50 per step) is comparable to the success/failure signal (±1.0–2.0), creating meaningful budget tradeoffs.

## Episode Metrics & Grading

Two deterministic evaluation utilities in `budget_router/reward.py`:

- `episode_metrics(history)` — total reward, success rate, total cost, average latency, SLA met, queue overflow events
- `grade_episode(history)` — returns `overall_score` in `[0,1]` with weighted breakdown:
  - 0.30 × success_score (routing success rate)
  - 0.20 × latency_score (latency relative to SLA ceiling)
  - 0.15 × budget_score (remaining budget fraction)
  - 0.15 × sla_score (per-request SLA compliance rate)
  - 0.20 × adaptation_score (post-degradation success rate — directly measures whether the agent detected and adapted to degradation)

The adaptation score is the key differentiator: it measures performance *after* degradation begins, rewarding agents that detect and respond to non-stationarity.

### Grader example output

```python
from budget_router.reward import grade_episode
# After running an episode, pass env._internal.history:
scores = grade_episode(history)
# Returns: {"overall_score": 0.72, "success_score": 0.85, "latency_score": 0.68,
#           "budget_score": 0.55, "sla_score": 0.90, "adaptation_score": 0.60}
# All scores clamped to [0.0, 1.0]
```

## Task Presets (exact values in code)

Defined in `budget_router/tasks.py`:

| Task | Budget | Degradation | Key challenge |
|------|--------|-------------|---------------|
| `easy` | `1.00` | None | Routing quality matters; always-A fails on heldout seeds |
| `medium` | `0.95` | A degrades after step 5 (rate 0.15) | Must detect A's degradation and switch routing |
| `hard` | `0.85` | A degrades from step 0 (rate 0.15), high noise (σ=50ms) | Tight budget + immediate degradation + noise |
| `hard_multi` | `1.00` | A degrades from step 0 (rate 0.12), B degrades from step 10 (rate 0.10) | Multi-provider failure cascade |

All tasks use `max_steps=20`, `max_queue_backlog=10`, `sla_ceiling_ms=500.0`.

### Validation Benchmarks (development seeds, mean reward)

| Task | Oracle | Baseline | Random | Baseline advantage |
|------|--------|----------|--------|--------------------|
| `easy` | 10.10 | 7.88 | 3.15 | +150% over random |
| `medium` | 9.49 | 3.72 | -6.75 | Baseline positive, random negative |
| `hard` | 6.57 | 0.01 | -13.15 | 656× gap |
| `hard_multi` | 2.83 | -3.43 | -11.30 | 182% gap (oracle-only solvable) |

HARD_MULTI is designed so the heuristic baseline fails by design — only an adaptive agent can achieve positive reward.

## Baselines (what exists + how we prevent gaming)

Policies are in `budget_router/policies.py`:

- `heuristic_baseline_policy`: cheapest provider whose windowed status > `0.52`, else `shed_load`
- `debug_upper_bound_policy`: oracle policy (has access to internal state; used only for validation)
- Degenerate policies:
  - `always_route_a_policy`
  - `always_route_b_policy`
  - `always_route_c_policy`
  - `always_shed_load_policy`

The validation harness (`budget_router/validation.py`) runs these policies across:

- **Development seeds**: 10 episodes per task
- **Held-out seeds**: 5 episodes per task

…and enforces policy ordering + stability + safety invariants.

## Reproducible Inference Script (hackathon requirement)

`inference.py` is a Typer CLI that can run either:

- `policy=heuristic` (no external calls)
- `policy=llm` using the OpenAI Python SDK with these environment variables:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`

It runs all 4 tasks (easy/medium/hard/hard_multi) across seeded episodes and outputs both raw metrics and **grader scores in [0.0, 1.0]**. Results are written to `baseline_results.json`.

```bash
# Heuristic baseline (no API needed)
uv run python inference.py --policy heuristic

# LLM policy (requires env vars)
API_BASE_URL=https://... MODEL_NAME=... HF_TOKEN=... uv run python inference.py --policy llm
```

## How to Run (local)

This repo uses `pyproject.toml` + `uv.lock`.

```bash
# Create venv
uv venv --python 3.12

# Install (including dev deps for tests)
uv pip install -e ".[dev]"

# Tests
uv run pytest

# Validation harness (prints results + runs hard assertions)
uv run python -m budget_router.validation

# OpenEnv multi-mode validation
openenv validate --verbose .
```

## Episode Visualization

4-panel matplotlib plots showing provider health, budget remaining, action distribution, and cumulative reward per step.

```bash
# Generate visualizations for all scenarios
uv run python visualize.py --scenario medium --seed 42
uv run python visualize.py --scenario hard_multi --seed 42 --policy oracle
```

Output goes to `docs/` as PNGs. These demonstrate the environment dynamics and policy behavior differences for judge presentation.

## Known Limitations / Potential Loopholes (grounded, not hypothetical)

- **Multi-target degradation is task-specific**: only `hard_multi` has secondary degradation (B from step 10). Other tasks degrade only provider A. This is realistic for single-vendor incidents but less realistic for correlated outages.
- **Shed-load can dodge SLA penalties**: `shed_load` produces no request latency (latency=0.0) and reduces backlog by 1, but it still incurs `-0.5`. The benchmark prevents “always shed” from winning via explicit degenerate-policy checks.

## Project Structure (as in this repo)

```
budget_router/
  models.py        # dataclasses: Action, Observation, EnvState, TaskConfig (incl. secondary degradation)
  environment.py   # BudgetRouterEnv (reset/step/state), degradation + backlog dynamics
  reward.py        # step_reward(), episode_metrics(), grade_episode() (incl. adaptation_score)
  tasks.py         # EASY/MEDIUM/HARD/HARD_MULTI presets
  policies.py      # heuristic baseline + oracle + degenerate baselines
  validation.py    # run_validation(), assert_all_checks(), run_manual_trace()
  client.py        # BudgetRouterClient (HTTPEnvClient)
  tests/
server/
  app.py           # FastAPI wrapper (create_app)
  Dockerfile       # OpenEnv base image + uv sync + uvicorn
docs/              # Episode visualization PNGs
openenv.yaml       # OpenEnv manifest
visualize.py       # Episode visualization script (4-panel matplotlib)
inference.py       # baseline reproduction script
pyproject.toml     # dependencies + server entrypoint
uv.lock            # resolved deps
```
