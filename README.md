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

Our baseline implementation uses a standard health-threshold heuristic. While effective in stationary environments, it exhibits significant failure modes in complex scenarios. For validation, the repo also includes a privileged upper bound that uses internal health state unavailable to the agent, so it should be treated as a debugging-only ceiling rather than a fair baseline.

**Provenance**: numbers below are development-seed runs (seeds 0–9), heuristic policy, current grader version.

| Scenario | Heuristic Mean Reward | Success Rate | Mean Latency (ms) | Heuristic Grader |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | 7.88 | 85.0% | 163.9 | 0.7958 |
| **Medium** | 3.72 | 80.5% | 188.7 | 0.7071 |
| **Hard** | 0.01 | 81.4% | 188.2 | 0.6778 |
| **Hard_Multi** | -2.38 | 72.8% | 243.2 | 0.6094 |

The stronger structural signal in this repo is that **Hard_Multi** separates privileged upper bound, heuristic, and random baselines rather than collapsing them—the task is strictly harder than **Hard** and is designed to expose policies that fail to anticipate the secondary cascade.

**Why Hard_Multi is Hard:** Provider A degrades from step 0, and Provider B begins degrading at step 10. After that point, Provider C at `$0.10/request` is the only consistently reliable option, and with a `1.10` starting budget, the final 10 steps alone can consume `$1.00` if the policy must lean on C. That makes the task fundamentally anticipatory: the agent has to conserve budget before B degrades, not merely react after the fact. This is exactly where short-horizon heuristics break down.

The episode grader includes an `adaptation_score` term. On no-degradation tasks such as **Easy**, `adaptation_score = 1.0` by design because no adaptation penalty is warranted; the signal becomes discriminative on **Medium**, **Hard**, and **Hard_Multi**, where degradation is actually present. For **Hard_Multi**, adaptation is measured as a blended 0.5/0.5 score across the primary window (steps between A's and B's degradation events) and the secondary window (steps after B degrades), directly rewarding policies that detect and respond to both cascade events.

**Grader note**: `success_score` is computed over all episode steps (not just routed steps), so a policy that sheds load is penalised in proportion to the fraction of steps it chose not to serve.

### Why RL?
As shown above, the heuristic's performance collapses in **Hard_Multi** scenarios, where multiple providers degrade simultaneously. An RL agent is required to anticipate budget depletion and shift routing strategies before SLA violations occur.

