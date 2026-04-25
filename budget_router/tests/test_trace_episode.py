import importlib.util
from pathlib import Path


def _load_trace_episode():
    path = Path(__file__).resolve().parents[2] / "eval" / "trace_episode.py"
    spec = importlib.util.spec_from_file_location("trace_episode", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trace_episode_returns_step_rows_and_scores_for_heuristic():
    trace_episode = _load_trace_episode()

    result = trace_episode.trace_episode(task_name="hard_multi", seed=3, policy_name="heuristic")

    assert result["task"] == "hard_multi"
    assert result["seed"] == 3
    assert result["policy"] == "heuristic"
    assert result["steps"]
    assert len(result["steps"]) == result["episode_length"]
    assert result["total_reward"] == round(sum(step["reward"] for step in result["steps"]), 4)
    assert 0.0 <= result["grader"]["overall_score"] <= 1.0
    assert {"success_rate", "total_cost_spent", "average_latency_ms"}.issubset(result["metrics"])
    assert {
        "provider_a_status",
        "provider_b_status",
        "provider_c_status",
        "observed_budget_remaining",
    }.issubset(result["steps"][0])


def test_trace_episode_rejects_unknown_policy():
    trace_episode = _load_trace_episode()

    try:
        trace_episode.trace_episode(task_name="hard_multi", seed=3, policy_name="unknown")
    except ValueError as exc:
        assert "Unknown policy" in str(exc)
    else:
        raise AssertionError("unknown policy should raise ValueError")
