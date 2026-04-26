from __future__ import annotations

MAX_STEPS = 20

SCENARIOS = ["easy", "medium", "hard", "hard_multi"]

POLICY_CHOICES = [
    ("Heuristic", "heuristic"),
    ("LLM", "llm"),
]

try:
    import stable_baselines3  # type: ignore  # noqa: F401

    POLICY_CHOICES.append(("PPO (hard_multi)", "ppo"))
except Exception:
    pass

PPO_MODEL_PATH = "trained_models/ppo_hard_multi_100k.zip"
