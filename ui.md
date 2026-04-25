---
description: Policy comparison UI plan (Gradio)
---

# Budget Router — Side-by-Side Policy Comparison UI (Plan)

## 0) Goal
Build a **professional, clean, responsive** comparison dashboard that runs **two policies side-by-side** on the same scenario/seed, with:
- Gated start (both policies selected)
- Step-by-step and fast-forward episode execution
- Per-policy history tables + per-step KPIs
- Policy options: **heuristic**, **llm**, **ppo**

Constraint: **No backend changes**.

## 1) References / design cues
- W&B Run Comparer (side-by-side run comparison)
  - https://docs.wandb.ai/models/app/features/panels/run-comparer
- TensorBoard scalar comparison mindset
  - https://www.tensorflow.org/tensorboard/scalars_and_keras
- Grafana dashboard layout best practices (panel consistency, density management)
  - https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/best-practices/

## 2) Architecture decision (data strategy)
### 2.1 Why Local Sim
The current OpenEnv server hosts a **single global env** (no session IDs), so true parallel stepping for Policy A + Policy B is not possible via `/reset` + `/step`.

Therefore, the comparison UI will run **two independent `BudgetRouterEnv` instances inside the Gradio app**.

### 2.2 What stays unchanged
- No changes to FastAPI/OpenEnv server.
- No changes to environment dynamics (`BudgetRouterEnv`).

### 2.3 Dependencies
- `ppo` requires `stable-baselines3` and `gymnasium`.
  - In this repo they exist under optional extras: `[project.optional-dependencies].training` in `pyproject.toml`.
- If these are not installed, the UI must **gracefully degrade**:
  - show a clear error message in the PPO panel when selected
  - keep start gated if a selected policy can’t be initialized

## 3) PPO policy loading plan
### 3.1 Model artifact
- Load from: `trained_models/ppo_hard_multi_100k.zip` (exists in repo)

### 3.2 Loading behavior
- Load the PPO model **once** (cached) and reuse for both panels.
- Predict deterministically:
  - `action_idx, _ = model.predict(obs_array, deterministic=True)`

### 3.3 Observation mapping (critical)
PPO expects a `(7,)` float array:
`[provider_a_status, provider_b_status, provider_c_status, budget_remaining, queue_backlog, system_latency, step_count]`

We will implement a mapping layer:
- `Observation(dataclass) -> np.ndarray` (same order as `train/gym_wrapper.py`)

### 3.4 Action mapping
PPO outputs a discrete index `0..3`, map to action types:
- `0 -> route_to_a`
- `1 -> route_to_b`
- `2 -> route_to_c`
- `3 -> shed_load`

### 3.5 Scenario mismatch handling
The PPO model was trained for `hard_multi`.
- If user selects a different scenario, allow running but show a status warning:
  - “PPO trained on hard_multi; results may be unstable on <scenario>.”

## 4) UI blueprint
### 4.1 Layout (top to bottom)
1. Header
   - Title: “Budget Router — Policy Comparison”
   - Subtitle: scenario + seed + max steps (20)

2. Comparison row (two equal columns)
   - **Policy Panel A** (left)
   - **Policy Panel B** (right)

3. Shared controls row
   - Scenario selector
   - Seed input
   - Start Comparison (gated)
   - Step (advance both)
   - Fast-forward (run both to done)
   - Finish episode (optional: run remaining steps)

4. History row (two equal columns)
   - History table A
   - History table B

5. Grade row (two equal columns)
   - Grade summary A
   - Grade summary B

### 4.2 Policy panel contents (A and B)
- Policy selector (radio/dropdown)
  - `heuristic`, `llm`, `ppo`
- Status (single line)
- Provider health (3 progress bars)
- Budget bar
- Compact KPIs
  - Step
  - Last action
  - Last reward
  - Cumulative reward

### 4.3 Gating rules
- Start button enabled only if:
  - both policy selectors are set AND
  - both policies successfully initialize (LLM has keys; PPO has deps + model file)

## 5) Interaction flows
### 5.1 Start
- Create state for left and right:
  - `env = BudgetRouterEnv()`
  - `env.reset(seed=<seed>, scenario=<scenario>)`
  - store initial observation, empty history

### 5.2 Step (synchronized stepping)
On each click:
- If a side is already `done`, it doesn’t advance
- Otherwise:
  - policy chooses action from current observation
  - env steps
  - append a history row
  - update that side’s panel + table + grade (only if done)

### 5.3 Fast-forward
- Run each side in a loop until `done` or max steps
- Update UI at the end (Phase 1)
  - (Optional later) stream intermediate states for a smoother animation

## 6) State model (Gradio)
We will store two separate `gr.State` dicts:
- `left_state`
- `right_state`

Each state:
- `env`: BudgetRouterEnv
- `obs`: latest observation (dict)
- `history`: list[dict]
- `step`: int
- `done`: bool
- `cumulative_reward`: float
- `policy_name`: str

## 7) Mapping layer (UI-friendly models)
Implement dedicated mapping helpers:
- `obs_to_ui(obs) -> dict` for provider/budget/KPIs
- `record_step(step, action, obs, reward, meta) -> dict` for tables
- `ppo_obs_array(obs) -> np.ndarray` (SB3 input)

## 8) Error handling / graceful degradation
- LLM missing config -> show actionable error in that panel
- PPO missing deps/model -> show actionable error
- Any policy exception on step -> freeze that side, show error status; other side continues

## 9) Phase 1 acceptance criteria (verifiable)
- You can select (heuristic/llm/ppo) for each side and **Start** is gated correctly.
- You can **Step** and both panels update independently.
- You can **Fast-forward** and both complete.
- Both history tables populate correctly.
- Grades appear when each side finishes.

## 10) Implementation notes (minimal-change approach)
- Reuse existing HTML renderers from `app_gradio.py` for provider/budget/history/grade.
- Refactor only enough to support two panels and local env execution.

---

# Approval request
If you approve this plan, I’ll implement it in `app_gradio.py` with minimal refactor and keep the styling consistent with your current light theme.
