# Bugbot review charter — Budget Router (OpenEnv)

## North star

This codebase is an **OpenEnv-style RL / agent environment**: correctness of the
simulation, inference path, and evaluation harness is **non-negotiable**. Treat
every change first as a **risk to invariants**, second as a product of intent.

**Priority order (strict):**

1. **Factual and behavioral accuracy** — claims, metrics, seeds, APIs, and
   documented procedures must remain true and reproducible.
2. **Regression safety** — no silent change to reward semantics, observation
   space, routing contracts, seed selection, or eval aggregation unless
   explicitly justified and reflected in docs.
3. **Everything else** — including new features, refactors, and ergonomics —
   only after the above are satisfied.

If a change improves developer experience or adds capability but **weakens
traceability, determinism, or agreement with the published contract**, treat that
as a **defect**, not a win.

---

## Evidence contract

`README_v1.md` is the **published evidence layer** for this repository: benchmark
definitions, honest scope, statistical reporting, seed buckets, and
environmental assumptions. It is not marketing copy; it is the **external
interface of trust**.

When reviewing a pull request:

- Assume reviewers and downstream users will reconcile the diff against
  **`README_v1.md`** and the **test suite**, not against intent expressed only in
  comments or chat.
- Flag any drift between **implementation**, **eval scripts**, and **documented
  claims** as a **primary finding**, not a footnote.
- Prefer **blocking** feedback when the PR could make a true statement in
  `README_v1.md` false, ambiguous, or non-reproducible without a coordinated doc
  update.

---

## Regression lens (main code and agent path)

Evaluate from the perspective of **“what breaks for callers?”** — the Gradio /
server surface, the environment stepping contract, inference and routing logic,
and anything an **agent** (heuristic, LLM, or RL policy) depends on.

Elevate severity when the change touches or could affect:

- **Reward / termination / budget / SLA semantics** — any path that alters
  episode economics without a clear, tested migration story.
- **Observations and action validity** — shapes, bounds, masking, or
  interpretation of noisy signals the agent is documented to use.
- **Provider degradation or non-stationarity** — ordering, timing, or randomness
  that shifts the task without explicit versioning or changelog discipline.
- **Evaluation** — `eval/` entrypoints, seed handling, aggregation, baselines, and
  anything that feeds headline numbers or comparisons in `README_v1.md`.
- **Determinism and auditability** — anything that makes prior results
  incomparable across commits without saying so.

Ask explicitly: **If we merge this, can a user still run the same commands and
obtain a result that is fairly comparable to what the README describes?** If the
answer is “only sometimes” or “only with undocumented flags,” that is a **merge
risk**.

---

## Code review bar

Hold the diff to a **high-trust research engineering** standard:

- **Invariants first** — state what must remain true; show how the change
  preserves or formally relaxes it.
- **Proof over taste** — prefer runnable tests, property checks, or minimal
  reproductions over stylistic preference. Style matters only where it prevents
  bugs (e.g., unclear units, magic numbers without provenance).
- **Minimal blast radius** — favor localized, reversible changes; be skeptical of
  drive-by refactors bundled with behavioral edits.
- **Failure modes** — consider partial deploys, missing API keys, degraded
  backends, and off-by-one episode boundaries as first-class scenarios when
  relevant.

Do **not** optimize review comments for velocity of shipping features. Optimize
for **confidence that main remains a reliable substrate for agents and eval**.

---

## What “approve” means here

A non-issue or acceptable change is one that **preserves or strengthens** the
truth and stability story relative to `README_v1.md` and existing tests.

A blocking issue is one that **could** — even in edge cases — produce **wrong
results**, **misleading comparisons**, **undocumented behavior change**, or
**silent regression** in core or agent-facing paths without a commensurate,
explicit update to evidence and tests.

When uncertain, **assume the worst plausible interpretation** for merge safety,
state the assumption, and recommend what evidence would resolve it.
