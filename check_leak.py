"""
check_leak.py — Validates BudgetRouterGRPOEnv before GRPO training.

Checks:
  1. Tool methods return strings (not crash).
  2. Episode ends gracefully via ValueError (TRL-idiomatic done signal).
  3. Reward is a float in [0, 1] — not a dict, not NaN.
  4. History uses actual_degradation_start (jittered) — NOT the config constant.
     This proves grade_episode() will compute correct adaptation windows.
  5. 10-step reward trajectory printed: verify no explosion/vanishing.
  6. Provider status IS present in tool responses (intentional — text interface needs it).

Run:
    uv run python check_leak.py
"""

import sys


def main() -> None:
    try:
        from train.grpo_env import BudgetRouterGRPOEnv
        from budget_router.reward import grade_episode
        from budget_router.tasks import HARD_MULTI
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        sys.exit(1)

    print("=" * 60)
    print("BudgetRouterGRPOEnv — Pre-training Validation")
    print("=" * 60)

    # ── Check 0: transformers version (soft warning — required for environment_factory) ──
    print("\n[CHECK 0] transformers version (required for environment_factory)...")
    try:
        import transformers
        ver_str = transformers.__version__
        # TRL's environment_factory requires transformers >= 4.47.0 (confirmed shipping in
        # stable builds as of Apr 2026). Exact minimum threshold is version-specific to TRL.
        # If not installed, training will fail at import time — caught here early.
        print(f"  ✅ transformers=={ver_str} installed.")
        # Soft check: warn if below 4.47 (minimum known to ship environment_factory support)
        major, minor = int(ver_str.split(".")[0]), int(ver_str.split(".")[1])
        if major < 4 or (major == 4 and minor < 47):
            print(
                f"  ⚠️  WARNING: transformers {ver_str} may be too old for environment_factory.\n"
                f"     Recommended: pip install 'transformers>=4.47.0' or install from main."
            )
    except ImportError:
        print(
            "  ⚠️  WARNING: transformers is NOT installed in this venv.\n"
            "     Install before GRPO training: pip install 'transformers>=4.47.0' trl accelerate peft"
        )

    # ── Check 1: reset() returns a non-empty string ─────────────────────
    print("\n[CHECK 1] reset() returns rich text observation...")

    env = BudgetRouterGRPOEnv()
    obs_text = env.reset(scenario="hard_multi", seed=42)
    assert isinstance(obs_text, str) and len(obs_text) > 10, \
        f"reset() should return non-empty string, got: {obs_text!r}"
    assert "Budget" in obs_text, "reset() should mention Budget"
    assert "Provider" in obs_text, "reset() should include provider status (text interface, not sanitized)"
    print(f"  ✅ reset() returned {len(obs_text)} chars. Provider status PRESENT (correct for text interface).")
    print(f"  Preview: {obs_text[:120].replace(chr(10), ' ')}...")

    # ── Check 2: Tool methods return strings step-by-step ───────────────
    print("\n[CHECK 2] Tool methods return strings and accumulate history...")
    env2 = BudgetRouterGRPOEnv()
    env2.reset(scenario="hard_multi", seed=42)

    step_results = []
    episode_done = False
    for step in range(25):  # more than max_steps to test guard
        action_fn = [env2.route_to_a, env2.route_to_b, env2.shed_load, env2.route_to_b][step % 4]
        try:
            result = action_fn()
            assert isinstance(result, str), f"Tool method should return str, got {type(result)}"
            step_results.append(result)
            print(f"  Step {step + 1:02d}: ✅ {result[:80].replace(chr(10), ' ')}...")
        except ValueError as e:
            episode_done = True
            print(f"  Step {step + 1:02d}: ✅ Episode ended via ValueError (TRL-idiomatic): {str(e)[:80]}...")
            break

    assert episode_done, "Episode should end with ValueError before step 25"
    assert len(step_results) > 0, "At least one tool step should complete"
    print(f"  ✅ Episode ended correctly after {len(step_results)} tool calls.")

    # ── Check 3: Reward is float in [0, 1] ──────────────────────────────
    print("\n[CHECK 3] Reward is float in [0, 1]...")
    assert isinstance(env2.reward, float), \
        f"env.reward should be float, got {type(env2.reward)}: {env2.reward!r}"
    assert 0.0 <= env2.reward <= 1.0, \
        f"env.reward should be in [0, 1], got {env2.reward}"
    import math
    assert not math.isnan(env2.reward), "env.reward is NaN — grade_episode bug"
    print(f"  ✅ env.reward = {env2.reward:.4f} (float, in [0,1], not NaN)")

    # ── Check 4: History uses actual jittered degradation_start_step ────
    print("\n[CHECK 4] History contains jittered actual_degradation_start (not config constant)...")
    history = env2._env._internal.history
    assert len(history) > 0, "History should not be empty after episode"

    # Read degradation_start_step from step_info (written by environment.py)
    step_info_degrade_start = history[0].get("degradation_start_step")
    # Read the actual jittered value from internal state
    actual_jittered_start = env2._env._internal.actual_degradation_start
    # Config constant for hard_multi
    config_constant = HARD_MULTI.degradation_start_step  # = 0

    print(f"  Config constant (degradation_start_step): {config_constant}")
    print(f"  step_info[degradation_start_step]: {step_info_degrade_start}")
    print(f"  internal.actual_degradation_start: {actual_jittered_start}")

    assert step_info_degrade_start is not None, \
        "step_info missing degradation_start_step — grade_episode() will break"
    assert step_info_degrade_start == actual_jittered_start, \
        (f"step_info uses wrong degradation onset! "
         f"Got {step_info_degrade_start}, expected {actual_jittered_start}. "
         f"This would corrupt adaptation scores in grade_episode().")
    print(f"  ✅ Jittered onset correctly propagated through step_info.")

    # ── Check 5: grade_episode() on history returns consistent score ─────
    print("\n[CHECK 5] grade_episode(history) matches env.reward...")
    grader_result = grade_episode(history)
    assert isinstance(grader_result, dict), "grade_episode should return dict"
    grader_score = float(grader_result["overall_score"])
    assert abs(grader_score - env2.reward) < 1e-6, \
        f"env.reward ({env2.reward}) != grade_episode score ({grader_score}). Mismatch."
    print(f"  ✅ grade_episode overall_score = {grader_score:.4f}, env.reward = {env2.reward:.4f}. Match confirmed.")

    # ── Check 6: 10-episode reward trajectory ────────────────────────────
    print("\n[CHECK 6] 10-episode reward trajectory (hard_multi, varying seeds)...")
    print("  Episode | Seed | Steps | Score | Reward-in-range")
    rewards = []
    for ep, seed in enumerate(range(10)):
        env3 = BudgetRouterGRPOEnv()
        env3.reset(scenario="hard_multi", seed=seed)
        done = False
        steps = 0
        while not done and steps < 30:
            # Alternate actions: A, B, A, B... (simple test policy)
            action_fn = env3.route_to_a if steps % 2 == 0 else env3.route_to_b
            try:
                action_fn()
                steps += 1
            except ValueError:
                done = True
        reward = env3.reward
        rewards.append(reward)
        in_range = "✅" if 0.0 <= reward <= 1.0 else "❌"
        print(f"  Ep {ep+1:02d}     | {seed:4d} | {steps:5d} | {reward:.4f} | {in_range}")

    import statistics
    if len(rewards) > 1:
        std = statistics.stdev(rewards)
        mean = statistics.mean(rewards)
        print(f"\n  Mean reward: {mean:.4f} | Std: {std:.4f}")
        if std < 0.03:
            print(
                f"  ⚠️  WARNING: Low reward variance (std={std:.4f}). GRPO may get weak gradient signal.\n"
                f"     Mitigation: Use num_generations=8, hard_multi scenario, and a small LLM\n"
                f"     at initialization that makes diverse routing decisions."
            )
        else:
            print(f"  ✅ Reward variance is sufficient for GRPO learning (std={std:.4f} > 0.03).")

    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED — BudgetRouterGRPOEnv is ready for GRPO training.")
    print("=" * 60)
    print("\nRecommended training config (Mac MPS / Colab):")
    print("  scenario: hard_multi")
    print("  num_generations: 8")
    print("  model: Qwen2.5-1.5B (Mac 16GB) / Qwen2.5-7B (Colab T4)")
    print("  Mac: TRL + PyTorch MPS (set PYTORCH_ENABLE_MPS_FALLBACK=1)")
    print("  Colab: Unsloth + vLLM on NVIDIA T4/A100")


if __name__ == "__main__":
    main()
