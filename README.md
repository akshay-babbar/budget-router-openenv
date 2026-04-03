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
* **Action Space:** Discrete routing to available providers or shedding load to preserve SLA.
* **Scenarios:** Includes 4 difficulty tiers, ranging from stationary "Easy" to "Hard_Multi" (simulating cascading provider failures).

### Benchmarks (Heuristic Baseline)
Our baseline implementation uses a standard health-threshold heuristic. While effective in stationary environments, it exhibits significant failure modes in complex scenarios.

| Scenario | Mean Reward | Success Rate | Mean Latency (ms) | Grader Score |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | 9.08 | 88.0% | 165.13 | 0.7863 |
| **Medium** | 2.41 | 83.0% | 176.37 | 0.7129 |
| **Hard** | -1.71 | 83.7% | 165.75 | 0.7127 |
| **Hard_Multi** | -1.25 | 73.7% | 224.09 | 0.6399 |

### Why RL?
As shown above, the heuristic's performance collapses in **Hard_Multi** scenarios, where multiple providers degrade simultaneously. An RL agent is required to anticipate budget depletion and shift routing strategies before SLA violations occur.

---
*Final/Detailed RL architecture and training logs will be released upon final submission.*
