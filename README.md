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

## Environment Dynamics (what actually happens per step)

```mermaid
sequenceDiagram
  participant Agent
  participant Env as BudgetRouterEnv.step()
  participant Deg as _degrade()
  participant Prov as Provider(A/B/C)

  Agent->>Env: Action(route_to_*) or shed_load
  Env->>Deg: Apply degradation (task-defined)
  alt shed_load
    Env->>Env: backlog = max(0, backlog-1)
    Env->>Env: reward = -0.5
  else route_to_provider
    Env->>Env: budget -= provider.cost
    Env->>Prov: success ~ Bernoulli(current_health)
    Env->>Env: update windowed success tracking
    Env->>Env: backlog += 2 on failure; backlog -= 1 on success
    Env->>Env: latency = base_latency + noise + (8ms * backlog)
    Env->>Env: reward = step_reward(...)
  end
  Env-->>Agent: Observation + reward + done
```

### Backlog causality (deterministic)

Backlog increases latency via a simple linear coupling:

- `BACKLOG_LATENCY_PER_ITEM_MS = 8.0`
- `backlog_latency = 8.0 * queue_backlog_count`
- Added to the computed request latency in `BudgetRouterEnv.step()`.

This makes backlog a leading indicator for SLA risk without adding new actions or external infrastructure.

## Reward (per-step, code-grounded)

Implemented in `budget_router/reward.py::step_reward()`:

- **Route actions**
  - `+1.0` on success
  - `-2.0` on failure
  - `-(provider_cost / initial_budget)` cost penalty
  - if `latency_ms > sla_ceiling_ms`: `-(excess_latency / sla_ceiling_ms)`
- **shed_load**: fixed `-0.5` (replaces routing terms)
- **Budget exhaustion**: terminal `-10.0` and `done=True`

## Episode Metrics & Grading

Two deterministic evaluation utilities exist in `budget_router/reward.py`:

- `episode_metrics(history)`
  - total reward, success rate, total cost, average latency, SLA met, queue overflow events
- `grade_episode(history)`
  - returns `overall_score` in `[0,1]` plus a breakdown

The current validation harness uses `episode_metrics`.

## Task Presets (exact values in code)

Defined in `budget_router/tasks.py`:

| Task | Initial budget | Degradation |
|------|----------------|-------------|
| `easy` | `1.0` | No degradation (`degradation_start_step=999`, `rate=0.0`) |
| `medium` | `0.95` | Provider A degrades after step 5 (`rate=0.15`, target `A`) |
| `hard` | `0.9` | Provider A degrades from step 0 (`rate=0.08`, target `A`) |

All tasks use `max_steps=20`, `max_queue_backlog=10`, `sla_ceiling_ms=500.0`.

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

It writes results to `baseline_results.json`.

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

## Known Limitations / Potential Loopholes (grounded, not hypothetical)

- **Single degradation target**: tasks currently degrade only provider A (see `TaskConfig.degradation_target` in `budget_router/tasks.py`). This is realistic for “one vendor incident” but less realistic for correlated outages.
- **Backlog model is intentionally simple**: latency coupling is linear and small (8ms per backlog item) in `budget_router/environment.py`. This captures tail-latency pressure but not full queueing theory.
- **Cost signal is smaller than failure signal**: cost penalty is `-(cost / initial_budget)` (see `step_reward`). This is deliberate to keep correctness dominant, but it can reduce gradient signal for fine-grained cost optimization.
- **Shed-load can dodge SLA penalties**: `shed_load` produces no request latency (latency=0.0) and reduces backlog by 1, but it still incurs `-0.5`. The benchmark prevents “always shed” from winning via explicit degenerate-policy checks.

## Project Structure (as in this repo)

```
budget_router/
  models.py        # dataclasses: Action, Observation, EnvState, TaskConfig, InternalState
  environment.py   # BudgetRouterEnv (reset/step/state), degradation + backlog dynamics
  reward.py        # step_reward(), episode_metrics(), grade_episode()
  tasks.py         # EASY/MEDIUM/HARD presets
  policies.py      # heuristic baseline + oracle + degenerate baselines
  validation.py    # run_validation(), assert_all_checks(), run_manual_trace()
  client.py        # BudgetRouterClient (HTTPEnvClient)
  tests/
server/
  app.py           # FastAPI wrapper (create_app)
  Dockerfile       # OpenEnv base image + uv sync + uvicorn
openenv.yaml       # OpenEnv manifest
inference.py       # baseline reproduction script
pyproject.toml     # dependencies + server entrypoint
uv.lock            # resolved deps
```
