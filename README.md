---
title: "Budget Router"
emoji: "⚙️"
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8000
base_path: /web
pinned: false
---

# Budget-Router-OpenEnv: An RL Environment for LLM Reliability

### The Problem: The Reliability Trilemma
Managing high-volume LLM workloads requires balancing three competing objectives: **Budget Efficiency**, **SLA Compliance**, and **Latency Minimization**. Traditional heuristics fail when providers exhibit non-stationary degradation. 

**Budget-Router-OpenEnv** is an OpenEnv-compatible RL environment exposing a Gymnasium-style `reset`/`step` API, designed to train agents that can navigate these trade-offs in real-time.

### Environment Dynamics
The agent acts as a high-frequency router, observing provider health signals and remaining budget to make optimal routing decisions.
* **State Space:** Windowed success rates, provider latency profiles, and relative budget health.
* **Observability:** Best framed as a noisy-observable constrained routing MDP: the agent sees 5-step rolling success rates, not true provider health. This is partial observability in a practical sense, but not deep hidden-state inference.
* **Action Space:** Discrete routing to available providers or shedding load to preserve SLA.
* **Scenarios:** Includes 4 difficulty tiers, ranging from stationary "Easy" to "Hard_Multi" (simulating cascading provider failures).

### Benchmarks

Our baseline implementation uses a standard health-threshold heuristic. The LLM Auto-Play policy uses an OpenAI-compatible endpoint (configurable via `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` environment variables — defaults to `https://router.huggingface.co/v1` with `Qwen/Qwen3-8B`). For validation, the repo also includes a privileged upper bound that uses internal health state unavailable to the agent, so it should be treated as a debugging-only ceiling rather than a fair baseline.

**Provenance**: numbers below are development-seed runs (seeds 0–9), current grader version.

| Scenario | Heuristic Mean Reward | Success Rate | Mean Latency (ms) | Heuristic Grader | LLM Grader | PPO Agent (50–100k steps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Easy** | 7.88 | 85.0% | 163.9 | 0.7958 | 0.7958 | 0.7497 |
| **Medium** | 3.72 | 80.5% | 188.7 | 0.7071 | 0.7071 | — |
| **Hard** | 0.01 | 81.4% | 188.2 | 0.6778 | 0.7091 | — |
| **Hard_Multi** | -2.38 | 72.8% | 243.2 | 0.6094 | 0.7157 | **0.6911** (+13.4%) |

The stronger structural signal in this repo is that **Hard_Multi** separates privileged upper bound, heuristic, and random baselines rather than collapsing them—the task is strictly harder than **Hard** and is designed to expose policies that fail to anticipate the secondary cascade.

**Why Hard_Multi is Hard:** Provider A degrades from step 0, and Provider B begins degrading at step 10. After that point, Provider C at `$0.10/request` is the only consistently reliable option, and with a `1.10` starting budget, the final 10 steps alone can consume `$1.00` if the policy must lean on C. That makes the task fundamentally anticipatory: the agent has to conserve budget before B degrades, not merely react after the fact. This is exactly where short-horizon heuristics break down.

The episode grader includes an `adaptation_score` term. On no-degradation tasks such as **Easy**, `adaptation_score = 1.0` by design because no adaptation penalty is warranted; the signal becomes discriminative on **Medium**, **Hard**, and **Hard_Multi**, where degradation is actually present. For **Hard_Multi**, adaptation is measured as a blended 0.5/0.5 score across the primary window (steps between A's and B's degradation events) and the secondary window (steps after B degrades), directly rewarding policies that detect and respond to both cascade events.

**Grader note**: `success_score` is computed over all episode steps (not just routed steps), so a policy that sheds load is penalised in proportion to the fraction of steps it chose not to serve.

### Policy Differentiation

The headline result is the **+17.4% LLM vs. heuristic gap on Hard_Multi** (0.7157 vs. 0.6094). This gap is structurally motivated: Provider B degrades at step 10, forcing a policy that can anticipate budget conservation before the cascade — not merely react after it. The heuristic routes greedily to the cheapest viable provider at each step and has no mechanism to pre-commit budget reserves. The LLM infers this constraint from the observation schema alone (specifically `step_count` and `budget_remaining`), recognising that late-episode reliance on Provider C ($0.10/request) will exhaust a budget that reactive routing did not conserve.

**PPO evidence (Hard_Multi, 100k steps, dev seeds 0–9):** A PPO agent trained directly on Hard_Multi achieves grader **0.6911 ± 0.031**, vs heuristic **0.6094 ± 0.028** — a **+13.4% improvement with non-overlapping 95% confidence intervals and a 10/10 seed win rate**. The structural source of the gain is the adaptation sub-score: PPO 0.940 vs heuristic 0.736 (Δ +0.204), confirming the agent learned to adjust routing *before* the B-provider cascade rather than reacting after it. This provides unambiguous evidence that the environment encodes a learnable signal unavailable to reactive baselines — establishing Hard_Multi as the key discriminative benchmark.

### Why RL?
As shown above, the heuristic's performance collapses in **Hard_Multi** scenarios, where multiple providers degrade simultaneously. An RL agent trained on Hard_Multi achieves +13.4% over the heuristic with statistical confidence (non-overlapping 95% CIs), driven by a +0.20 adaptation score improvement. This demonstrates the environment is trainable, the reward signal is dense enough for credit assignment across 20 steps, and the structural challenge — anticipatory budget conservation across a provider cascade — is learnable from the observation schema alone.

