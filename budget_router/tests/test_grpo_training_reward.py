import pytest

from budget_router.reward import grade_episode
from train.grpo_env import BudgetRouterGRPOEnv
from train.learn_experiment import build_dataset, build_system_prompt, reward_func, summarize_training_rollout


def _step_once(env: BudgetRouterGRPOEnv) -> None:
    # Any routing action is fine; we just need non-empty history.
    # Use B as a reasonably stable default.
    try:
        env.route_to_b()
    except ValueError as e:
        # If an episode somehow terminates early, that's fine for the test harness,
        # but it would make the "partial episode" test invalid.
        raise AssertionError(f"Episode ended unexpectedly after one step: {e}") from e


def _run_to_completion(env: BudgetRouterGRPOEnv) -> None:
    # Drive the episode until the GRPO wrapper signals completion.
    while True:
        try:
            env.route_to_b()
        except ValueError:
            return


def test_reward_func_empty_history_returns_zero():
    env = BudgetRouterGRPOEnv()
    env.reset(scenario="hard_multi", seed=0)

    rewards = reward_func([env])
    assert rewards == [0.0]


def test_reward_func_partial_episode_is_progress_scaled_not_full_grader():
    env = BudgetRouterGRPOEnv()
    env.reset(scenario="hard_multi", seed=0)

    _step_once(env)

    internal = env._env._internal
    assert internal.history, "test precondition: history must be non-empty"
    assert not internal.episode_done, "test precondition: episode must be incomplete"

    grader = float(grade_episode(internal.history)["overall_score"])
    progress = internal.current_step / max(1, internal.max_steps)
    expected = grader * progress

    # This is the critical regression guard: training reward must not be equal
    # to the raw grader when the episode is incomplete.
    rewards = reward_func([env])
    assert rewards == [pytest.approx(expected, abs=1e-6)]
    assert rewards[0] != pytest.approx(grader, abs=1e-6)


def test_reward_func_complete_episode_equals_full_grader():
    env = BudgetRouterGRPOEnv()
    env.reset(scenario="hard_multi", seed=0)

    _run_to_completion(env)

    internal = env._env._internal
    assert internal.history, "test precondition: history must be non-empty"
    assert internal.episode_done, "test precondition: episode must be complete"

    grader = float(grade_episode(internal.history)["overall_score"])
    rewards = reward_func([env])
    assert rewards == [pytest.approx(grader, abs=1e-6)]


def test_training_rollout_summary_exposes_partial_episode_health():
    env = BudgetRouterGRPOEnv()
    env.reset(scenario="hard_multi", seed=0)

    _step_once(env)
    _step_once(env)

    summary = summarize_training_rollout([env])

    assert summary["env_steps_mean"] == pytest.approx(2.0)
    assert summary["env_steps_min"] == 2
    assert summary["env_steps_max"] == 2
    assert summary["episode_completion_rate"] == 0.0
    assert summary["progress_mean"] == pytest.approx(0.1)
    assert summary["raw_grader_mean"] > summary["training_reward_mean"]


def test_training_rollout_summary_exposes_action_sequence_diversity():
    same_a = BudgetRouterGRPOEnv()
    same_b = BudgetRouterGRPOEnv()
    different = BudgetRouterGRPOEnv()
    for env in (same_a, same_b, different):
        env.reset(scenario="hard_multi", seed=0)

    same_a.route_to_b()
    same_a.route_to_b()
    same_b.route_to_b()
    same_b.route_to_b()
    different.route_to_a()
    different.route_to_a()

    summary = summarize_training_rollout([same_a, same_b, different])

    assert summary["action_sequences"] == [
        "route_to_b route_to_b",
        "route_to_b route_to_b",
        "route_to_a route_to_a",
    ]
    assert summary["unique_action_sequences"] == 2
    assert summary["action_sequence_counts"] == {
        "route_to_b route_to_b": 2,
        "route_to_a route_to_a": 1,
    }


def test_grpo_tool_feedback_is_compact_for_multi_turn_budget():
    env = BudgetRouterGRPOEnv()
    env.reset(scenario="hard_multi", seed=0)

    feedback = env.route_to_b()

    assert len(feedback) < 180
    assert "steps_left=" in feedback
    assert "health=" in feedback


def test_explore_prompt_preserves_tool_format_without_deterministic_policy():
    prompt = build_system_prompt("explore")

    assert "<tool_call>" in prompt
    assert '"name": "route_to_a"' in prompt
    assert "route_to_a" in prompt
    assert "route_to_b" in prompt
    assert "route_to_c" in prompt
    assert "shed_load" in prompt
    assert "0.52" not in prompt
    assert "cheapest healthy provider" not in prompt.lower()
    assert "Observation:" not in prompt
    assert "route_to_a route_to_b route_to_c" not in prompt


def test_build_dataset_uses_requested_prompt_style():
    dataset = build_dataset(n=1, prompt_style="explore")
    system_prompt = dataset[0]["prompt"][0]["content"]

    assert system_prompt == build_system_prompt("explore")

