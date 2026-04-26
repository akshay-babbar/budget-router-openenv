# Budget Router Reproducibility Guide

This guide is a Pareto-optimal falsification checklist for Budget Router. Its goal is not to run every possible experiment; it is to quickly answer the questions most likely to invalidate the project claims:

- Does the environment still behave like the source describes?
- Does the grader still resist reward gaming and abstention exploits?
- Does the heuristic remain a real baseline rather than a degenerate trick?
- Does the LLM policy beat the heuristic for the right reasons, not just prompt or seed overfitting?
- Does PPO still demonstrate learnability beyond reactive heuristics on `hard_multi`?

Use the active `README.md` only as a claim surface and intuition source. The source of truth is the code: `budget_router/environment.py`, `budget_router/reward.py`, `budget_router/policies.py`, `budget_router/tasks.py`, `inference.py`, `eval/eval_all.py`, `eval/trace_episode.py`, `budget_router/validation.py`, and the tests under `budget_router/tests/`. Do not use archived README files for this analysis.

## Mental Model

Budget Router is a partially observable routing environment. A policy chooses one of:

- `route_to_a`
- `route_to_b`
- `route_to_c`
- `shed_load`

The policy sees normalized public observations only: provider rolling success estimates, remaining budget, queue backlog, latency, and progress. It does not see true provider health. Provider status `0.5` means unprobed/unknown, not healthy.

The environment has two scoring layers:

- Step reward in `budget_router/reward.py::step_reward`: dense learning signal with success/failure, cost, SLA penalty, and a catastrophic budget-exhaustion path in `BudgetRouterEnv.step`.
- Episode grader in `budget_router/reward.py::grade_episode`: semantic benchmark score in `[0, 1]` using success, latency, budget, SLA, and adaptation.

This distinction matters. Reward hacking usually appears when a policy optimizes a shaped reward or loophole that does not match the semantic grader. The most important checks below are designed to catch that quickly.

## The 20-30% Command Ladder

Run these in order when you want high confidence fast. Stop at the first failure and inspect before spending tokens or API calls on larger experiments.

### 1. Install the Base Environment

```bash
uv sync
```

Why: this is the minimal dependency set for unit tests, heuristic policy checks, environment validation, and non-LLM traces. It does not require API keys or training dependencies.

Red flags:

- dependency resolution fails
- imports fail for `openenv_core`, `typer`, or local `budget_router`
- tests below require hidden setup not documented in code

### 2. Run the Unit and Regression Tests

```bash
uv run pytest budget_router/tests
```

Why: this is the fastest broad guardrail. It covers deterministic resets, observation bounds, reward sanity, anti-abstention grader semantics, `hard_multi` adaptation windows, seed selection, LLM prompt structure, trace output shape, and GRPO reward behavior.

Highest-value test areas:

- `test_environment.py::TestGraderSemantics`: catches reward gaming by always shedding or partially abstaining.
- `test_environment.py::TestBehavioralGuards`: catches heuristic budget-exhaustion regressions on `hard_multi`.
- `test_eval_all_seed_selection.py`: catches seed-bucket drift and explicit fresh-seed parsing regressions.
- `test_inference_prompt.py`: catches LLM prompt regressions around budget runway, noise calibration, task name, and bankruptcy warnings.
- `test_grpo_training_reward.py`: catches GRPO reward mistakes where incomplete episodes get full grader credit.

Red flags:

- pure abstention scores too high
- partial abstention beats full service
- `hard_multi` adaptation ignores the secondary degradation window
- explicit seeds no longer override named seed sets
- LLM prompt loses `0.500 = unobserved`, budget runway, or bankruptcy constraints
- GRPO partial episodes get the full episode grader

### 3. Run No-API Environment Validation

```bash
uv run python -m budget_router.validation
```

Why: this compares random, heuristic, oracle, and degenerate policies across tasks and seed sets without calling an LLM. It is the best single command for environment validity, reward-gaming resistance, and oracle-vs-baseline headroom.

What it checks from source:

- `random_policy`: lower-bound behavior.
- `heuristic_baseline_policy`: public-observation, cheapest-viable baseline.
- `debug_upper_bound_policy`: oracle/debug policy with privileged internal health access.
- degenerate policies: always A, always B, always C, always shed.
- hard assertions: baseline beats random on core tasks, oracle beats baseline, degenerate policies do not all dominate, heldout behavior is stable, rewards are not NaN, episodes do not exceed 20 steps.

How to interpret:

- Oracle above heuristic means the environment has exploitable headroom.
- Heuristic above random means the benchmark is not noise.
- Degenerate policies failing to dominate means the grader is not trivially gameable.
- Heldout stability means basic environment behavior is not seed-fragile.

Red flags:

- oracle no longer beats heuristic on any meaningful task
- random beats heuristic broadly outside the intentionally hard `hard_multi` caveat
- always shed or always C dominates the heuristic
- validation passes only because assertions were weakened
- reward means shift sharply without a corresponding intentional source change in `tasks.py`, `environment.py`, or `reward.py`

### 4. Inspect Exact-Seed Behavior With Traces

Use traces when aggregate numbers move or when you suspect reward hacking. Start with heuristic because it is deterministic and no-API.

**Progress while the episode runs:** By default, `eval/trace_episode.py` prints nothing until the episode completes (then it prints the full table and optional JSON). For **~20 sequential LLM calls**, that can look “stuck.” Pass `**--verbose`** or `**-v**` to print one `**[trace]**` line per environment step as it happens (`step`, `action`, step `reward`, cumulative reward, `done`). For `**--policy llm**`, you also get a `**[trace] begin …**` line before the first network call, and `**llm_error=…**` when a step falls back after an API error.

```bash
uv run python eval/trace_episode.py \
  --task hard_multi \
  --seed 3 \
  --policy heuristic \
  --verbose \
  --output-json outputs/trace_heuristic_hard_multi_seed3.json
```

If training extras are installed: use the bundled `trained_models/ppo_hard_multi_100k.zip`, or train from scratch first (overwrites the default save path used by the trace script):

```bash
uv sync --extra training

# Recreate the checkpoint from scratch (optional if zip already present and trusted)
uv run python train/train_ppo_hard_multi.py

uv run python eval/trace_episode.py \
  --task hard_multi \
  --seed 3 \
  --policy ppo \
  --verbose \
  --output-json outputs/trace_ppo_hard_multi_seed3.json
```

If API credentials are configured:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-token>"

uv run python eval/trace_episode.py \
  --task hard_multi \
  --seed 3 \
  --policy llm \
  --verbose \
  --output-json outputs/trace_llm_hard_multi_seed3.json
```

Why: After the episode, `eval/trace_episode.py` prints the public observation before each action plus action, provider, success, reward, cumulative reward, cost, budget, latency, and final grader breakdown. With `**--verbose**`, you also see **per-step progress during** the run (recommended for LLM). This is the fastest way to see whether a policy is actually adapting or merely exploiting a scoring artifact.

Red flags:

- policy sheds many steps but grader remains high
- policy burns budget early and still scores well
- policy never probes unknown providers but appears to infer hidden health
- LLM repeatedly switches on one noisy failure despite the prompt's noise calibration
- PPO repeatedly chooses a degenerate sequence such as always C or always shed
- traces expose hidden provider health to the acting policy; the trace may display evidence after the fact, but policy inputs should remain public observations

### 5. Reproduce Heuristic vs LLM Claims by Seed Bucket

Set credentials only for LLM runs:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-token>"
```

Dev full-suite check:

```bash
uv run python eval/eval_all.py \
  --tasks easy --tasks medium --tasks hard --tasks hard_multi \
  --policies heuristic --policies llm \
  --seeds 10 \
  --seed-set dev \
  --out-dir outputs/repro_dev_alltasks
```

Heldout `hard_multi` check:

```bash
uv run python eval/eval_all.py \
  --tasks hard_multi \
  --policies heuristic --policies llm \
  --seeds 10 \
  --seed-set heldout \
  --out-dir outputs/repro_heldout_hard_multi
```

Fresh arbitrary-seed check:

```bash
uv run python eval/eval_all.py \
  --tasks hard_multi \
  --policies heuristic --policies llm \
  --seed-values "200,201,202,203,204,205,206,207,208,209" \
  --out-dir outputs/repro_fresh_200_209_hard_multi
```

Why: `eval/eval_all.py` writes timestamped JSON and Markdown summaries. Its seed logic has explicit named buckets for `dev` and `heldout`, plus `--seed-values` for arbitrary fresh seeds. Fresh seeds are the main defense against "tuned on heldout" critiques.

How to interpret:

- Dev is useful for smoke and comparison with existing README claims.
- Heldout is the first real overfitting check.
- Fresh seeds are the strongest quick falsifier of prompt/guard overfitting.
- Compare paired seeds, not just aggregate means; LLM and heuristic should be evaluated on the same seeds.

Red flags:

- LLM only wins on dev and collapses on heldout/fresh
- LLM improvement comes mostly from one outlier seed
- LLM loses the `hard_multi` adaptation sub-score while gaining budget score via excessive shedding
- LLM invalid outputs are silently converted to `shed_load` too often
- API/model changes make results incomparable without recording `MODEL_NAME`, endpoint, date, and prompt mode

Optional raw LLM audit:

```bash
LLM_LOG_RAW=1 LLM_LOG_RAW_MAX_CHARS=400 \
uv run python eval/eval_all.py \
  --tasks hard_multi \
  --policies heuristic --policies llm \
  --seed-values "200,201,202" \
  --out-dir outputs/repro_llm_raw_audit
```

Why: this helps distinguish real policy behavior from parser/guard artifacts. The parser in `inference.py` extracts a valid action string when present and falls back to `shed_load` when parsing fails.

### 6. Evaluate the Included PPO Hard_Multi Policy

```bash
uv sync --extra training

uv run python train/eval_hard_multi.py
```

Why: this is the source-backed PPO comparison path for `hard_multi`. It loads `trained_models/ppo_hard_multi_100k.zip`, evaluates deterministic PPO on seeds `0-9`, evaluates the heuristic on the same seeds, reports mean/std/95% CI/win rate/subscores, and writes `outputs/ppo_hard_multi_eval.json`.

Red flags:

- model file is missing
- PPO no longer beats heuristic on most paired seeds
- PPO wins only by budget preservation while success/adaptation collapse
- PPO traces reveal degenerate always-action behavior
- PPO results are compared against a different seed set than heuristic

Important limitation: `eval/eval_all.py` accepts `--policies ppo` but currently only warns that PPO is not wired there. Use `train/eval_hard_multi.py` or `eval/trace_episode.py --policy ppo` (optional `--verbose`) for PPO evidence.

### 7. Retrain PPO Only When You Need to Revalidate Learnability

```bash
uv sync --extra training

uv run python train/train_ppo_hard_multi.py

uv run python train/eval_hard_multi.py
```

Why: training is expensive relative to the other checks. Run it when source changes touch `environment.py`, `reward.py`, `tasks.py`, `train/gym_wrapper.py`, or PPO hyperparameters. The current training script uses Stable-Baselines3 PPO, `MlpPolicy`, 4 parallel envs, 100k steps, and saves `trained_models/ppo_hard_multi_100k.zip`.

Red flags:

- PPO cannot improve after training
- training reward improves but grader does not
- policy learns to terminate early or exploit budget scoring
- learned behavior is strong on dev seeds but weak on exact fresh traces

### 8. GRPO/Tool-Calling Smoke Checks

Use this only if you are touching GRPO/training-wrapper code:

```bash
## blocked for now till we fix GRPO
#uv sync --extra grpo

#PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train/smoke_test.py
```

Why: this validates model-to-tool-to-environment-to-reward plumbing. It is not evidence of learning. The unit tests around `train/grpo_env.py` and `train/learn_experiment.py` are more important for reward correctness.

Red flags:

- model makes no tool calls and receives nonzero reward
- incomplete episodes receive full grader score
- tool wrapper constructs custom history instead of delegating to `BudgetRouterEnv.step`
- action-sequence diversity collapses before learning is expected

## Policy Definitions

Heuristic policy:

- Defined in `budget_router/policies.py::heuristic_baseline_policy`.
- Uses only public `Observation`.
- Chooses the cheapest provider with status above `0.52` or unprobed `0.5`.
- Applies a simple low-budget guard that excludes expensive C below `0.10` budget fraction.
- This is a reactive baseline, not an oracle.

Oracle/debug upper-bound policy:

- Defined in `budget_router/policies.py::debug_upper_bound_policy`.
- Uses privileged `InternalState`, including true provider health and remaining budget.
- It is validation-only and should never be presented as a deployable policy.
- Its purpose is to prove there is headroom above the public-observation heuristic.

LLM policy:

- Defined in `inference.py::LLMRouter`.
- Uses an OpenAI-compatible chat API.
- Prompt requires exactly one action string.
- Adds trend text, budget runway, task name, and optional previous-step feedback.
- Applies `_apply_budget_safety_guard`, which vetoes actions that would immediately exhaust public remaining budget.
- Parser fallback is `shed_load`; frequent fallback is a red flag, not a win.

PPO policy:

- Training path: `train/train_ppo_hard_multi.py`.
- Evaluation path: `train/eval_hard_multi.py`.
- Trace path: `eval/trace_episode.py --policy ppo` (optional `--verbose` / `-v` for per-step lines during the run).
- Gym wrapper: `train/gym_wrapper.py`.
- Current headline PPO scope is `hard_multi`, not all tasks.

Degenerate policies:

- Defined in `budget_router/policies.py`.
- Always A, always B, always C, always shed.
- These are not competitors; they are exploit detectors.

## What Counts as "Results Still Stand"

The README claims are still credible only if the following all hold:

1. Unit tests pass, especially grader semantics and seed-selection tests.
2. `budget_router/validation.py` still shows non-triviality, oracle headroom, degenerate-policy resistance, heldout stability, no NaNs, and no >20-step episodes.
3. Exact traces show plausible adaptation rather than abstention, parser fallback, or hidden-state leakage.
4. LLM vs heuristic remains positive on paired heldout and fresh `hard_multi` seeds, not just dev.
5. PPO evaluation through `train/eval_hard_multi.py` still beats heuristic on paired dev seeds if PPO claims are retained.
6. Any material drift is reflected in `README.md`; do not preserve old claims if the source-backed commands contradict them.

## Fast Failure Triage

If unit tests fail:

- Inspect `reward.py` first for grader regressions.
- Inspect `environment.py` next for step history, budget exhaustion, termination, observation bounds, and degradation timing.
- Inspect `tasks.py` if task difficulty or seed outcomes moved unexpectedly.

If validation fails:

- Compare random, heuristic, oracle, and degenerate rows.
- If degenerate policies dominate, the grader or task economics are probably gameable.
- If oracle has no headroom, the task is too easy or the oracle/health dynamics changed.
- If heuristic is unstable across seed sets, check degradation jitter and stochastic success paths.

If LLM results fail:

- Confirm `MODEL_NAME`, endpoint, prompt mode, and credentials.
- Run a one-seed trace with `--policy llm` (add `--verbose` so each step logs while waiting on the API).
- Enable `LLM_LOG_RAW=1` for a small seed slice.
- Check whether failures are reasoning failures, parser failures, safety-guard interventions, or API/model drift.

If PPO results fail:

- Confirm `trained_models/ppo_hard_multi_100k.zip` exists.
- Run `eval/trace_episode.py --policy ppo` (optionally `--verbose`) on a winning and losing seed.
- Check whether `train/gym_wrapper.py` observation/action mapping still matches `BudgetRouterEnv`.
- Retrain only after source-level checks pass.

## Minimum Evidence Bundle for a PR or Submission

For a fast but serious evidence package, save outputs from:

```bash
uv run pytest budget_router/tests

uv run python -m budget_router.validation

uv run python eval/trace_episode.py \
  --task hard_multi \
  --seed 3 \
  --policy heuristic \
  --verbose \
  --output-json outputs/evidence_trace_heuristic_hard_multi_seed3.json

uv run python eval/eval_all.py \
  --tasks hard_multi \
  --policies heuristic --policies llm \
  --seeds 10 \
  --seed-set heldout \
  --out-dir outputs/evidence_heldout_hard_multi

uv run python eval/eval_all.py \
  --tasks hard_multi \
  --policies heuristic --policies llm \
  --seed-values "200,201,202,203,204,205,206,207,208,209" \
  --out-dir outputs/evidence_fresh_hard_multi

uv sync --extra training
uv run python train/eval_hard_multi.py
```

This bundle covers correctness, anti-gaming, environment validity, exact behavior, heldout/fresh LLM comparison, and PPO learnability. It is small enough to run before a merge, but broad enough to catch most ways the published claims could become false.