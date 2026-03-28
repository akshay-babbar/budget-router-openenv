# Meta × Hugging Face × Scaler OpenEnv Hackathon 2026: Strategic Topic Selection

## 1. EXECUTIVE ANSWER

*   **Primary Recommendation:** Build **EconAPI: The Dynamically Priced Tool-Use Environment**. It is an agentic environment where LLMs must satisfy user requests by navigating a network of stateful API tools that feature dynamic pricing, token-bucket rate limits, and latency priority queues.
*   **Why it maximizes win probability:** Tool-use is the #1 focus of the OpenEnv Hub right now, but existing benchmarks (like WebArena or GAIA) treat tools as free, instant, and infinite. EconAPI introduces *economic friction* to tool-use, creating a highly rigorous, non-trivial RL challenge.
*   **Asymmetric Advantage (Economist):** Will design the dynamic pricing algorithms (bonding curves) and non-gameable budget reward mechanics, ensuring the environment tests true strategic trade-offs (cheap/slow API vs. expensive/fast API).
*   **Asymmetric Advantage (Backend Engineer):** Will architect a robust, Dockerized microservice mesh simulating real-world rate limits, priority queues, and stateful API token tracking.
*   **Automated Verifiability:** 100% deterministic grading based on programmatic verification of the final data state and the mathematical validation of remaining budget/SLA metrics. No subjective LLM-as-a-judge required for the core loop.

## 2. CONTEXT INTELLIGENCE SUMMARY

**Official Hackathon Parameters:** [FACT] The "Meta PyTorch OpenEnv Hackathon x SST" registration closes April 3, 2026. Round 1 requires building a Mini-RL environment with defined tasks, graders, and reward logic by April 8. Evaluation utilizes programmatic checks and LLM scoring. The finale is a 48-hour in-person build at Scaler School of Technology in Bangalore (April 25-26) with $30,000 in prizes and direct interview opportunities with Meta/Hugging Face. [Source: Scaler Hackathon Official Page]

**OpenEnv Technical Structure:** [FACT] OpenEnv is a collaborative framework by Meta and Hugging Face for agentic execution. It uses Gymnasium-style APIs (`reset()`, `step()`) and relies heavily on Docker for containerized, isolated execution. Environments are deployed via the OpenEnv CLI to Hugging Face Spaces. [Source: Hugging Face Blog / OpenEnv GitHub]

**Strategic Sponsor Priorities:** [INFERENCE] Meta's Superintelligence Labs and FAIR are heavily indexing on post-training for agentic RL, specifically focusing on long-horizon reasoning, self-correction, and tool orchestration in complex execution environments. [Source: Meta AI Blog / LinkedIn job postings]

**Benchmark Ecosystem Gaps:** [FACT] Current leading benchmarks (SWE-bench, WebArena, GAIA, OSWorld) focus predominantly on static or isolated tasks (e.g., fixing a specific GitHub issue or navigating a static DOM). [INFERENCE] There is a massive whitespace for environments that simulate the operational, economic, and systemic constraints of the real world—such as rate limits, cost optimization, and multi-agent resource contention. [Source: Holistic Agent Leaderboard (HAL) 2025 Review]

## 3. CANDIDATE LANDSCAPE

| Domain | RL fit | Verifiability | Feasibility | Strategic relevance | Novelty | Initial verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. CI/CD Pipeline Healing | High | High | High | High | High | Keep |
| 2. DB Query Optimization | Med | High | High | Med | Low | Elim |
| 3. Cloud Spot Market Auction | High | High | Med | High | High | Keep |
| 4. Ad-Bidding Simulator | High | High | Low | Low | Low | Elim |
| 5. Tool-Use API Quota Economy | High | High | High | High | High | Keep |
| 6. E-commerce Dynamic Pricing | High | High | Med | Low | Low | Elim |
| 7. Multi-Agent Meeting Scheduler | Low | Low | High | Low | Low | Elim |
| 8. Distributed Consensus Repair | High | High | Low | High | High | Keep |
| 9. SLA-Driven ProfitOps (SRE) | High | High | High | High | Med | Keep |
| 10. Cache Space Bidding | High | High | Med | Med | Med | Keep |
| 11. LLM Prompt Attack/Defense | High | Low | Med | High | High | Elim |
| 12. Medical Triage Allocation | Med | Low | High | Low | Med | Elim |
| 13. Grid Energy Trading | High | High | Low | Low | High | Elim |
| 14. Network Traffic QoS Routing | High | High | Low | Med | Med | Elim |
| 15. Supply Chain Procurement | High | High | Med | Med | High | Elim |
| 16. Web Scraping Resiliency | Low | Med | Med | Med | Low | Elim |
| 17. Scientific Equipment Sched. | High | Med | Low | Med | Med | Elim |
| 18. Code Vulnerability Patching | Med | Med | Med | High | Low | Elim |
| 19. Decentralized Storage Bids | High | High | Low | Med | High | Elim |
| 20. Smart Contract Audit/Exploit | High | High | Low | Med | High | Elim |

## 4. FILTERED SHORTLIST

**Eliminated Candidates:**
*   *DB Query Optimization:* Eliminated (Constraint 6: Too generic, easy for shallow teams to copy).
*   *Ad-Bidding Simulator:* Eliminated (Constraint 4: Low strategic alignment with Meta's open-source agentic priorities).
*   *E-commerce Dynamic Pricing:* Eliminated (Constraint 4: Lacks relevance to developer/infra agents).
*   *Multi-Agent Meeting Scheduler:* Eliminated (Constraint 1: Prompts-in/prompts-out, not genuine sequential RL).
*   *LLM Prompt Attack/Defense:* Eliminated (Constraint 2: Grading requires highly subjective LLM-as-a-judge).
*   *Medical Triage Allocation:* Eliminated (Constraint 2: Cannot be safely/deterministically auto-graded without human-in-the-loop).
*   *Grid Energy Trading:* Eliminated (Constraint 5: Too much custom physics/domain simulation required for Round 1).
*   *Network Traffic QoS Routing:* Eliminated (Constraint 5: High infra burden to simulate a realistic packet network).
*   *Supply Chain Procurement:* Eliminated (Constraint 4: Does not align with PyTorch/OpenEnv technical focus).
*   *Web Scraping Resiliency:* Eliminated (Constraint 1: Mostly reactive DOM parsing, lacking deep sequential state transitions).
*   *Scientific Equipment Sched.:* Eliminated (Constraint 5: Custom dataset requirements are too heavy).
*   *Code Vulnerability Patching:* Eliminated (Constraint 3: Hard to systematically decompose difficulty without relying on external repositories).
*   *Decentralized Storage Bids:* Eliminated (Constraint 5: Unstable simulation mechanics for a 2-week sprint).
*   *Smart Contract Audit/Exploit:* Eliminated (Constraint 4: Web3 focus misaligned with Meta/HF core roadmap).

**Surviving Candidates:**
1.  Tool-Use API Quota Economy (EconAPI)
2.  Cloud Spot Market Auction (InfraAuction)
3.  SLA-Driven ProfitOps (SRE Profit Maximizer)
4.  Distributed Consensus Repair (Byzantine Hunt)
5.  Cache Space Bidding (Stateful Tiering)

## 5. DEEP DIVES

### Candidate 1: Tool-Use API Quota Economy (EconAPI)
1. **Problem Statement:** Tool-using agents navigate a network of stateful APIs that utilize dynamic pricing, latency priority queues, and token-bucket rate limits. Agents must fulfill complex user queries while strictly managing an economic API budget.
2. **RL Shape:** The agent faces a classic MDP: spending budget on a fast/expensive API now reduces available budget for subsequent required actions, forcing long-horizon economic planning and tool substitution.
3. **State/Action Space Sketch:**
   *State:* `[remaining_budget, api_price_vector, rate_limit_status, task_graph_completion]`
   *Action:* `[select_tool(api_id), payload, priority_bid_amount]`
4. **Three Tasks:**
   *   *Easy:* Static pricing. Agent executes 3 sequential API calls within budget. Grader: `return 1.0 if verify_db_state(target) and state.budget >= 0 else 0.0`
   *   *Medium:* Dynamic pricing active. Agent must switch from high-cost API to low-cost (but slower) alternative mid-task to survive. Grader: `return 1.0 if verify_db_state() and max_latency < SLA_limit`
   *   *Hard:* Priority queues active. Agent must dynamically bid high for blocking critical-path tasks and low for async tasks. Grader: `return normalized_score(remaining_budget, total_latency)`
5. **Reward Design:** `(Task_Success_Value * 100) - Sum(API_Costs) - (Total_Latency * Penalty_Rate)`. The mechanism design ensures no single static path is optimally profitable across different randomized network states.
6. **Exploit Mode:** Agent hallucinates the final answer without calling APIs to preserve 100% of its budget. *Mitigation:* The grader cryptographically verifies that the required state mutations actually occurred in the backend Docker containers; hallucinations score 0.0.
7. **Economist Exploitability:** Designs the bonding curves for dynamic API pricing and ensures the VCG-style bidding mechanics for priority queues are mathematically sound and non-gameable.
8. **Backend Exploitability:** Builds the FastAPI/Docker mock microservices with authentic token-bucket rate limiters and physical state management.
9. **Benchmark Gap:** Extends GAIA/WebArena by introducing *economic friction* and *systemic constraints*, accurately modeling how agents operate in real enterprise architectures.
10. **Meta/HF Judge:** *Pro:* Directly addresses the challenge of deploying agentic swarms at scale without bankrupting compute budgets. *Con:* Might be superficially viewed as "just another tool-use wrapper" if the economic mechanics aren't clearly highlighted.
11. **Feasibility Verdict:** Yes. Highly achievable.

### Candidate 2: Cloud Spot Market Auction (InfraAuction)
1. **Problem Statement:** Multiple AI agents (representing tenant workloads) compete in a continuous spot-auction for shared Docker cluster resources. They must optimally bid to acquire CPU/RAM allocations to finish their jobs before SLA timeouts.
2. **RL Shape:** Sequential bidding under uncertainty. Acquiring resources depletes budget; failing to acquire resources increases the SLA penalty risk.
3. **State/Action Space Sketch:**
   *State:* `[budget, workload_remaining, current_spot_price, active_allocations]`
   *Action:* `[bid_price, requested_units]`
4. **Three Tasks:**
   *   *Easy:* Single agent bidding against a fixed-heuristic baseline bot. Grader: `return 1.0 if workload_done and budget >= 0 else 0.0`
   *   *Medium:* Multi-agent competition with varying workload priorities. Grader: `return agent_priority_score / max_theoretical`
   *   *Hard:* Non-stationary resource supply (simulated node failures during auction). Grader: `return normalized_profit_margin`
5. **Reward Design:** `(Workload_Completion_Value - Bid_Cost) - Time_Penalty`.
6. **Exploit Mode:** Agent bids zero and waits for demand to drop to zero indefinitely. *Mitigation:* Strict Time-To-Live (TTL) on workloads with exponentially compounding SLA penalties forces action.
7. **Economist Exploitability:** Required to design the generalized second-price auction mechanics to ensure truthful bidding is the dominant strategy.
8. **Backend Exploitability:** Must build a high-throughput stateful auction engine that accurately tracks and allocates virtualized Docker resources.
9. **Benchmark Gap:** Introduces multi-agent economic resource contention, moving beyond single-agent puzzle-solving.
10. **Meta/HF Judge:** *Pro:* Meta runs massive shared compute clusters; AI scheduling AI is a real-world priority. *Con:* Too abstract; abstracts away from generative NLP and reasoning, focusing purely on numerical optimization.
11. **Feasibility Verdict:** Marginal. Building a stable, synchronous auction engine for an RL interface by April 8 is high-risk.

### Candidate 3: SLA-Driven ProfitOps (SRE Profit Maximizer)
1. **Problem Statement:** An SRE agent manages a fleet of web servers processing volatile traffic. It must scale infrastructure up/down to maximize profit (revenue per request minus infra costs and strict SLA violation penalties).
2. **RL Shape:** Decisions have delayed consequences. Spinning up a server takes 3 timesteps; the agent must anticipate traffic spikes to avoid catastrophic SLA penalties.
3. **State/Action Space Sketch:**
   *State:* `[current_qps, latency_p99, active_nodes, boot_queue_status]`
   *Action:* `[scale_up(N), scale_down(N), drop_traffic(pct)]`
4. **Three Tasks:**
   *   *Easy:* Predictable sine-wave traffic. Grader: `return normalized_profit / theoretical_max`
   *   *Medium:* Spiky, Poisson-distributed traffic. Grader: `return normalized_profit`
   *   *Hard:* Stateful nodes where scaling down abruptly drops active user sessions, incurring massive penalties. Grader: `return normalized_profit`
5. **Reward Design:** `(Requests_Served * Revenue) - (Nodes * Node_Cost) - (SLA_Violations * Exponential_Penalty)`.
6. **Exploit Mode:** Agent never scales down, eating the infra cost but guaranteeing zero SLA penalties, achieving a safe sub-optimal plateau. *Mitigation:* Revenue margins are tight; if utilization drops below 60%, the agent bleeds money and fails.
7. **Economist Exploitability:** Designs the non-linear SLA penalty curves to create a complex, non-convex optimization surface that requires genuine strategic risk management.
8. **Backend Exploitability:** Builds the traffic generator and stateful node simulator that physically enforces boot delays and connection draining.
9. **Benchmark Gap:** Shifts SRE benchmarks from "debugging code" (SWE-bench) to "operational site reliability and financial risk."
10. **Meta/HF Judge:** *Pro:* Highly relevant to Meta's infrastructure teams. *Con:* Leans heavily into Operations Research rather than agentic text reasoning.
11. **Feasibility Verdict:** Yes.

### Candidate 4: Distributed Consensus Repair (Byzantine Hunt)
1. **Problem Statement:** An agent must maintain the uptime of a distributed Dockerized database cluster where nodes randomly become Byzantine (corrupt data or delay responses).
2. **RL Shape:** Agent issues queries, observes conflicting responses, updates reputation scores, and makes sequential decisions to incur the cost of isolating bad nodes.
3. **State/Action Space Sketch:**
   *State:* `[node_reputations, consensus_health_metric, recent_latencies]`
   *Action:* `[query_node(id), isolate_node(id), restart_node(id)]`
4. **Three Tasks:**
   *   *Easy:* One simple crash-fault node. Grader: `return 1.0 if bad_node_isolated else 0.0`
   *   *Medium:* One Byzantine node returning subtly wrong data. Grader: `return 1.0 if uptime > 0.99 and bad_node_isolated else 0.0`
   *   *Hard:* Colluding Byzantine nodes. Grader: `return 1.0 if consensus_maintained else 0.0`
5. **Reward Design:** `+1 per successful consensus commit, -100 for data corruption, -10 for an unnecessary restart`.
6. **Exploit Mode:** Agent restarts all nodes constantly to clear faults. *Mitigation:* Restarting costs a massive availability penalty, guaranteeing a negative final score.
7. **Economist Exploitability:** Implements game-theoretic slashing conditions and proof-of-stake style reputation tracking.
8. **Backend Exploitability:** Builds a Dockerized Raft/Paxos consensus simulation with injected fault states.
9. **Benchmark Gap:** Marries infrastructure management with adversarial robustness and safety.
10. **Meta/HF Judge:** *Pro:* Highly novel, excellent for robust multi-agent systems research. *Con:* Extremely complex to evaluate deterministically without race conditions.
11. **Feasibility Verdict:** No. Too complex to build a stable distributed system simulator in 2 weeks.

### Candidate 5: Cache Space Bidding (Stateful Tiering)
1. **Problem Statement:** Agent manages caching for a stateful web app, bidding for space in a fast/expensive Redis tier vs a slow/cheap DB tier.
2. **RL Shape:** Cache eviction and placement are sequential; placing an item now affects latency/budget later.
3. **State/Action Space Sketch:**
   *State:* `[cache_contents, incoming_query_freq, available_budget]`
   *Action:* `[promote_to_redis(key), demote_to_db(key)]`
4. **Three Tasks:**
   *   *Easy:* Static query distribution. Grader: `return baseline_latency / agent_latency`
   *   *Medium:* Drifting query hot-spots. Grader: `return normalized_score`
   *   *Hard:* Adversarial query spikes. Grader: `return normalized_score`
5. **Reward Design:** `Latency_Saved_Value - Cost_of_Cache_Tier`.
6. **Exploit Mode:** Agent caches nothing to save budget. *Mitigation:* SLA penalty for DB latency far exceeds the cache cost.
7. **Economist Exploitability:** Structures the cost vs. latency reward function.
8. **Backend Exploitability:** Builds the Redis/DB mock architecture.
9. **Benchmark Gap:** Fills a niche in data-engineering RL environments.
10. **Meta/HF Judge:** *Pro:* Good infra use-case. *Con:* Slightly niche and less visually impressive than tool orchestration.
11. **Feasibility Verdict:** Yes.

## 6. ADVERSARIAL STRESS TEST

**1. EconAPI (Tool-Use Quota Economy)**
*   **Attacker A (Skeptical Meta Lead):** "This is just WebArena with a 'cost' variable appended to the prompt. It's too generic."
    *   *Mitigation:* "It is not a prompt trick. The backend is a real, stateful Docker network with actual token-bucket rate limiters and priority queues. API latency physically increases when the budget is low, forcing the agent to manage real asynchronous execution and physical network limits, not simulated text."
*   **Attacker B (PyTorch Engineer):** "The action space is under-constrained; the agent can just spam APIs until it hits the answer."
    *   *Mitigation:* "The grading is strictly profit/budget-based. Spamming APIs depletes the strict token budget, resulting in the backend physically returning 429 Too Many Requests errors and an automatic environment failure state."
*   **Attacker C (Competition Strategist):** "Other teams will also build tool-use environments. How does this stand out?"
    *   *Mitigation:* "Other teams will build static 'fetch the weather' wrappers. We are building an *economy*. Our economist's dynamic pricing algorithms (bonding curves) make the environment highly dynamic and mathematically defensible, directly preventing memorization and shallow heuristics."

**2. ProfitOps (SLA-Driven SRE)**
*   **Attacker A:** "Scaling servers is an Operations Research problem, not a Generative AI / Agent problem."
    *   *Mitigation:* "Modern autonomous agents need to operate in environments with delayed consequences. ProfitOps requires the agent to parse natural language SLA contracts, translate them into a risk model, and execute temporal scaling commands—perfectly bridging LLM reasoning with OR."
*   **Attacker B:** "The verifier is brittle because traffic generation is random."
    *   *Mitigation:* "We use a fixed-seed pseudo-random traffic generator for evaluation, guaranteeing 100% deterministic, reproducible profit scores for the identical sequence of agent actions."
*   **Attacker C:** "The demo is risky because simulating Docker boot times in a 48-hour hackathon will timeout or crash."
    *   *Mitigation:* "For Round 1 and the live demo, we abstract the physical Docker boots into a stateful Python tick-engine that mathematically mimics boot delays without the heavy container overhead, ensuring instant, flawless grading."

**3. InfraAuction (Cloud Spot Market)**
*   **Attacker A:** "Bidding for resources is too abstract. It doesn't test the agent's ability to actually do real work."
    *   *Mitigation:* "The environment marries bidding with execution. The agent must acquire the resource AND submit the correct execution payload. Bidding is the gating mechanism for the work."
*   **Attacker B:** "Multi-agent RL environments are impossible to grade consistently because policies co-adapt and diverge."
    *   *Mitigation:* "We evaluate the agent against a suite of frozen, heuristic-driven baseline bots (e.g., greedy bidder, random bidder) to ensure a mathematically stable, objective evaluation landscape."
*   **Attacker C:** "It's too mathematically complex for judges to understand quickly in a 3-minute pitch."
    *   *Mitigation:* "The visualization dashboard maps the auction to a live 'stock ticker' of cluster resources, making the game-theoretic dynamics instantly visceral and visually impressive."

## 7. SCORING TABLE

| Candidate | Verifiability (22) | Round 1 Fit (18) | RL Auth. (15) | Feasibility (12) | Finale Depth (10) | Team Adv. (10) | Novelty (8) | Sponsor Res. (5) | Total (100) | Trade-off Commentary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. EconAPI** | 22 | 18 | 14 | 12 | 10 | 10 | 8 | 5 | **99** | Perfect alignment. Trade-off: Requires tight scoping of APIs for Round 1 to stay on schedule. |
| **2. ProfitOps** | 22 | 16 | 15 | 10 | 8 | 8 | 7 | 4 | **90** | Highly rigorous, but trades off generative AI "flavor" for heavy Operations Research focus. |
| **3. InfraAuction**| 18 | 14 | 15 | 8 | 10 | 10 | 8 | 4 | **87** | Deeply exploitable by the team, but high implementation risk and slightly noisy evaluation. |

## 8. FINAL RECOMMENDATION

**Primary Choice: EconAPI (The Dynamically Priced Tool-Use Environment)**
*   **Competitive Pitch:** We are taking the hottest topic in AI (agentic tool use) and solving its biggest benchmarking flaw: the assumption that tools are free. By introducing economic friction, dynamic rate limits, and priority bidding into a Dockerized API mesh, we force agents to demonstrate genuine strategic planning and resource efficiency. It is perfectly verifiable, highly relevant to Meta's swarm orchestration goals, and custom-tailored to our team's dual strengths in backend infra and mechanism design.
*   **Round 1 Scope:** A single agent interacting with 3 mock stateful APIs (Search, Compute, DB) built in FastAPI, backed by a fixed token budget and static pricing. Grader simply checks final DB state and remaining budget.
*   **Finale Expansion Direction:** Introduce dynamic bonding-curve pricing based on network load, multi-agent competition for API rate limits, and priority queue bidding.
*   **Specific Team Match:** The backend engineer builds the resilient API mesh and token buckets; the economist designs the non-gameable pricing curves and reward functions.
*   **Top Risk & Mitigation:** *Risk:* Judges view it as "just another WebArena." *Mitigation:* Over-index the presentation and UI on the *economic dashboard*—showing the agent's real-time financial decision-making and the physical network latency it experiences.

**Runner-up:** ProfitOps (SLA-Driven SRE). Extremely solid, highly verifiable, but slightly less aligned with mainstream "tool-use" hype.
**High-risk / High-upside:** Distributed Consensus Repair. If you can pull off the distributed systems simulation, it wins on novelty, but the implementation risk for April 8 is severe.
**Safest strong option:** Cache Space Bidding. A heavily scoped-down version of ProfitOps that guarantees a working prototype but risks being too niche.

## 9. RESEARCH GAPS
*   **Granular Judging Rubric:** While the core hackathon structure, timeline, and broad evaluation criteria (programmatic checks + LLM scoring) were successfully extracted from the official Scaler landing page, the exact weighted point distribution for Round 1 grading (e.g., % weight given to code quality vs. RL complexity) is not publicly visible on the unauthenticated web. **Action:** This must be verified manually from the participant dashboard immediately upon registration to ensure the weights in Phase 6 perfectly align with the judges' internal scorecards.