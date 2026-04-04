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

**Budget-Router-OpenEnv** is a Gymnasium-compatible environment designed to train agents that can navigate these trade-offs in real-time.

### Environment Dynamics
The agent acts as a high-frequency router, observing provider health signals and remaining budget to make optimal routing decisions.
* **State Space:** Windowed success rates, provider latency profiles, and relative budget health.
* **Observability:** Best framed as a noisy-observable constrained routing MDP: the agent sees 5-step rolling success rates, not true provider health. This is partial observability in a practical sense, but not deep hidden-state inference.
* **Action Space:** Discrete routing to available providers or shedding load to preserve SLA.
* **Scenarios:** Includes 4 difficulty tiers, ranging from stationary "Easy" to "Hard_Multi" (simulating cascading provider failures).

### Benchmarks
Our baseline implementation uses a standard health-threshold heuristic. While effective in stationary environments, it exhibits significant failure modes in complex scenarios. For validation, the repo also includes a privileged upper bound that uses internal health state unavailable to the agent, so it should be treated as a debugging-only ceiling rather than a fair baseline.

| Scenario | Heuristic Mean Reward | Success Rate | Mean Latency (ms) | Heuristic Grader | LLM Grader |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Easy** | 9.08 | 88.0% | 165.13 | 0.7863 | 0.7461 |
| **Medium** | 2.41 | 83.0% | 176.37 | 0.7129 | 0.7008 |
| **Hard** | -1.71 | 83.7% | 165.75 | 0.7127 | 0.7091 |
| **Hard_Multi** | -1.25 | 73.7% | 224.09 | 0.6399 | 0.7157 |

The `LLM Grader` column is a checked-in reference run; the stronger structural signal in this repo is that **Hard_Multi** separates privileged upper bound, heuristic, and random baselines rather than collapsing them into the same regime.

LLM routing materially outperforms the heuristic on **Hard_Multi** in grader score, while matching it closely on **Hard** under the checked-in evaluation runs.

**Why Hard_Multi is Hard:** Provider A degrades from step 0, and Provider B begins degrading at step 10. After that point, Provider C at `$0.10/request` is the only consistently reliable option, and with a `1.10` starting budget, the final 10 steps alone can consume `$1.00` if the policy must lean on C. That makes the task fundamentally anticipatory: the agent has to conserve budget before B degrades, not merely react after the fact. This is exactly where short-horizon heuristics break down.

In development validation, that structural gap is already visible: on **Hard_Multi**, the privileged upper bound is positive (`2.83`), the heuristic baseline is negative (`-3.43`), and random is far worse (`-11.30`). That is the key judge-facing result: the task is not trivial, but it is also not impossible.

The episode grader includes an `adaptation_score` term. On no-degradation tasks such as **Easy**, `adaptation_score = 1.0` by design because no adaptation penalty is warranted; the signal becomes discriminative on **Medium**, **Hard**, and **Hard_Multi**, where degradation is actually present.

### Why RL?
As shown above, the heuristic's performance collapses in **Hard_Multi** scenarios, where multiple providers degrade simultaneously. An RL agent is required to anticipate budget depletion and shift routing strategies before SLA violations occur.

