# Budget Router: Teaching RL Agents to Make Non-Myopic Decisions Under Compounding Constraints

Every production LLM system already does some version of routing. Requests are sent across providers that differ in price, latency, and reliability, and the routing logic is usually a thin policy layer sitting between a queue and a budget dashboard. The problem is that this layer is almost always myopic. On the `hard_multi` setting in Budget Router, that myopia is the difference between a positive and a negative system trajectory: the oracle achieves `2.83`, the heuristic baseline gets `-3.43`, and random falls to `-11.30`. That `6.26`-point oracle–heuristic gap is the headline result of this environment. It is not a cosmetic benchmark spread. It is the structural cost of making locally reasonable routing decisions under a global budget constraint.

Budget Router models a concrete production failure mode: an agent routes requests across three providers under simultaneous budget, SLA, and reliability constraints while provider quality degrades non-stationarily. Provider A is cheapest at `$0.01/request` but least reliable, B costs `$0.05/request` and is the balanced middle option, and C costs `$0.10/request` and is the most reliable. When one provider degrades, and then another degrades later, the right policy is no longer “pick the cheapest provider above threshold.” The correct policy has to reason about future feasibility. Spending budget now changes what actions remain available later.

This is why the environment is framed as incident response rather than as a static routing heuristic. Failures increase backlog, backlog amplifies latency, and latency pushes the system toward further failure. The objective is to stabilize a partially observed system before reliability, queueing, and budget produce a cascade.

[IMAGE: hard_multi_oracle]

## 1. The Production Problem

Production LLM orchestration is now a control problem. Teams are balancing three quantities on every request: whether the request succeeds, whether the response stays under the SLA ceiling, and whether routing preserves enough budget to survive the rest of the traffic horizon. In Budget Router’s `easy` task, the heuristic reaches `7.88` while random gets `3.15`. But the moment reliability becomes non-stationary, that apparent adequacy disappears.

The `hard_multi` scenario is designed to reflect the failure pattern practitioners actually face. Provider A degrades from step `0` at rate `0.12`; provider B then degrades from step `10` at rate `0.10`; the episode budget is `1.10`; and the agent must decide among `route_to_a`, `route_to_b`, `route_to_c`, or `shed_load`. A stateless cheapest-viable router keeps buying B until the system is already committed to an unfavorable budget position. That policy ends at `-3.43`. The oracle, by contrast, stays positive at `2.83` on the benchmark table and reaches `10.73` on the seed-`42` hero trace by explicitly conserving budget through the cascade window.

That is the production motivation for Budget Router: AI infrastructure increasingly needs policies that act like operators, not threshold scripts. If one provider is cheap, another is stable, and the budget is finite, the key decision is not only *which route is best now*. It is *which route keeps the future feasible*.

## 2. Environment Design: Why It Is Hard

Budget Router is a finite-horizon POMDP with a compact but causally entangled state interface. The agent observes a `7`-dimensional vector, normalized to `[0,1]`: `provider_a_status`, `provider_b_status`, `provider_c_status`, `budget_remaining`, `queue_backlog`, `system_latency`, and `step_count`. The first three are windowed success rates over the last `5` requests, not latent health values. The agent never sees provider health directly; it sees only a lagging behavioral signature of degradation.

The environment difficulty comes from five interacting properties.

First, it is partially observed. In `medium`, A begins degrading only after step `5`, but the agent sees only the delayed effect of that degradation in the recent success window. In the seed-`42` trace, the heuristic continues routing A through the onset of failure, suffers a reward drawdown, then pivots to B around steps `11–12`, after which cumulative reward recovers to `8.05`. That is why `medium` remains positive for the heuristic at `3.72` while random is `-6.75`: the agent can recover, but only after inferring that the world has changed.

[IMAGE: medium_heuristic]

Second, the transition dynamics are non-stationary. `easy` has no degradation and budget `1.00`; `medium` adds a delayed regime shift with budget `0.95`; `hard` starts degradation immediately with tighter budget `0.85` and latency noise `σ=50ms`; `hard_multi` adds a second degrading provider under budget `1.10`. The same observation semantics mean something different across tasks. A policy has to infer both current provider quality and where it is in the incident timeline.

Third, budget is not just a penalty term. It is the mechanism that couples present and future action sets. In `hard`, the heuristic learns the obvious local response—route B instead of A—yet still collapses because it cannot manage remaining budget across the full horizon. The benchmark mean is `0.01` for the heuristic versus `6.57` for the oracle and `-13.15` for random. In the seed-`42` visualization, the heuristic climbs steadily and then hits a sharp end-of-episode cliff when budget reaches zero and the `-10.0` terminal penalty fires.

[IMAGE: hard_heuristic]

Fourth, failures create compounding dynamics rather than isolated losses. Backlog enters latency multiplicatively through `1 + 0.5 * queue_norm`, and failed requests incur an additional `200ms` penalty. This makes the queue a causal state variable: failures raise backlog, backlog increases latency, and elevated latency worsens the effective operating condition of the system.

Fifth, the agent is evaluated on a Pareto surface, not a single objective. The grader combines `0.30 × success + 0.20 × latency + 0.15 × budget + 0.15 × SLA + 0.20 × adaptation_score`. A policy cannot maximize one term without paying somewhere else. That is what makes the environment research-grade rather than a hand-tuned routing puzzle.

## 3. The Key Insight: Budget as Shadow Price

The central idea behind Budget Router is that budget should be treated as a shadow price over future timesteps, not as a scalar penalty on current actions. This is most visible in the `hard_multi` oracle trace for seed `42`. The oracle routes B from steps `1–13`, sheds load on steps `14–16` during the B-cascade window, then pivots to C for steps `17–20`. The cumulative reward ends at `10.73`. The action sequence looks conservative in the middle of the episode, but it is exactly that restraint that preserves enough budget to exploit C when reliability has become scarce.

A myopic router sees something simpler. It observes that C is expensive, B is still above threshold, and shed-load incurs an immediate `-0.5` penalty, so it keeps routing B. That logic is locally plausible and globally wrong. Once B’s health falls through the cascade window, the system is forced into failures with elevated latency while the remaining budget is too thin to use C aggressively. That is how the benchmark lands at `-3.43` for the heuristic while the oracle remains positive at `2.83`.

This interpretation aligns with the standard constrained-MDP view of resource limits. In the Lagrangian formulation of constrained control, the multiplier on a budget or cost constraint acts as the marginal value of preserving one more unit of that resource; later budgeted-RL formulations make the remaining budget an explicit part of the decision state. Budget Router is not solving the dual analytically, but it instantiates the same economic logic operationally: the value of spending `$0.10` on C is not fixed. It depends on how many steps remain, how degraded the other providers are, and what future failures would cost if the budget is exhausted.

In that sense, the `hard_multi` gap is the environment’s research hypothesis. Non-myopic routing is not just “use the best model.” It is sequential resource allocation under partial observability. The correct policy must learn when an immediate success is worth less than preserving future optionality.

## 4. Reward Design: Incentive Compatibility

The reward function is designed to make the right behavior learnable and the wrong shortcuts unprofitable. For route actions, the agent receives `+1.0` on success, `-2.0` on failure, a cost penalty normalized by the initial episode budget and scaled by `5.0`, and an SLA breach penalty when latency exceeds the `500ms` ceiling. `shed_load` gets `-0.5`, and budget exhaustion ends the episode with `-10.0`.

Those constants matter. If `shed_load` were cheaper than `-0.5`, the environment would invite evasive policies that avoid latency and failure risk without providing service. If it were much more expensive, the agent would never learn to use it strategically during a cascade. In `hard_multi`, the oracle demonstrates why the current calibration is correct: accepting three `-0.5` penalties is rational when it preserves enough budget to route four high-value requests through C later.

The episode-level grader closes the remaining loopholes. The key term is `0.20 × adaptation_score`, which measures post-degradation success rate directly rather than letting an agent hide behind aggregate reward. In practice, this forces the learned policy to care about the interval where the system is actually under incident stress. A policy that performs well before step `5` in `medium` or before step `10` in `hard_multi` but fails after degradation will lose on the grader even if raw reward remains superficially acceptable.

Degenerate policies are also explicitly checked. The validation harness compares the heuristic against `always_route_a`, `always_route_b`, `always_route_c`, and `always_shed_load`, and it enforces policy-ordering, stability, and NaN-safety invariants. On development seeds, for example, `always_route_a` collapses to `-22.02` on `medium` and `-31.48` on `hard`, while `always_shed_load` stays pinned at `-10.00`. The reward structure is therefore not merely implemented; it is incentive-audited.

## 5. Difficulty Calibration

The four tasks form an explicit curriculum rather than a set of unrelated presets.

`easy` teaches the basic cost-quality tradeoff. There is no degradation, budget is `1.00`, and the heuristic reaches `7.88` while the oracle reaches `10.10`. The agent can learn that cheap routing is viable but not always optimal.

[IMAGE: easy_heuristic]

`medium` introduces temporal detection. A degrades only after step `5`, budget drops to `0.95`, and the heuristic remains positive at `3.72` because adaptation is possible once the observation stream reveals the regime shift.

`hard` forces immediate degradation and budget pressure simultaneously. A degrades from step `0`, latency noise rises to `σ=50ms`, budget tightens to `0.85`, and the heuristic mean collapses to `0.01` while the oracle stays at `6.57`.

`hard_multi` adds the final missing skill: non-myopic budget allocation during a cascade. A degrades from step `0`, B degrades from step `10`, budget widens slightly to `1.10`, and the benchmark becomes oracle `2.83`, heuristic `-3.43`, random `-11.30`. The wider budget is not a kindness. It is what creates a solvable but nontrivial planning problem where conserving budget can dominate greedy routing.

The progression is the point. An agent trained across `easy → medium → hard → hard_multi` can first learn routing quality, then change-point response, then budget pressure, and finally the coupled setting where all of those competencies are required at once.

## 6. What an RL Agent Must Learn

Budget Router is designed to teach four concrete competencies.

First, passive degradation detection from noisy, windowed observations. The agent only sees the last-`5` success rates, so it must infer latent provider deterioration from delayed evidence.

Second, budget-aware routing. In `hard`, the difference between the oracle’s `6.57` and the heuristic’s `0.01` is not provider selection alone; it is whether the policy internalizes that expensive routes early can make later recovery impossible.

Third, temporal reasoning. `medium` asks when A started failing. `hard_multi` asks a harder question: whether preserving budget now is worth more than spending it before B’s second degradation fully materializes.

Fourth, load shedding as a strategic control action. In many production systems, shedding is treated as pure failure. In Budget Router it is a legitimate stabilization primitive. The `-0.5` penalty makes it undesirable in steady state and optimal only when it prevents a more damaging sequence of failures or preserves budget for higher-value future routing.

These are exactly the skills that production LLM orchestration agents will need as model routing, API brokerage, and reliability management converge.

## 7. OpenEnv Integration + Reproducibility

Budget Router is packaged as an OpenEnv FastAPI environment via `openenv.yaml` with `runtime: fastapi`, `app: server.app:app`, and `port: 8000`, exposed through `openenv_core.env_server.create_app()`. The repo includes a validation harness, seeded development and held-out evaluation sets, visualization scripts, and a reproducible inference entry point.

Reproducibility is enforced rather than implied. The validation harness runs `10` development seeds and `5` held-out seeds per task, compares oracle, heuristic, random, and degenerate policies, and checks policy ordering, stability margins, NaN safety, and episode termination bounds. The current validation output passes `36/36` hard assertions. For a submission artifact, that matters: the environment is conceptually motivated and operationally testable.

## Closing

Budget Router is a training ground for the class of agents that will increasingly govern production AI infrastructure. The core claim is simple: non-myopic budget reasoning under partial observability is learnable, and it matters. The `hard_multi` oracle–heuristic gap of `6.26` is not just a benchmark delta. It is evidence that when reliability degrades, backlog compounds, and budget is finite, locally sensible routing can still be systemically wrong. This environment was designed to provide the signal needed to learn that distinction.

---

**Theory references used for Section 3**

- Eitan Altman, *Constrained Markov Decision Processes* — standard CMDP treatment using Lagrangian methods for constrained control.
- Nicolas Carrara et al., *Budgeted Reinforcement Learning in Continuous State Space* (arXiv:1903.01004) — BMDP formulation where budget is explicit in the sequential decision problem.
