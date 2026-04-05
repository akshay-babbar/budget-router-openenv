"""
Train a PPO agent on the Hard_Multi scenario.

This is the key experiment: Hard_Multi has a secondary provider cascade at step 10
(Provider B degrades after A). A reactive heuristic cannot conserve budget in advance
and scores ~0.6094. An RL agent with access to step_count + budget_remaining can
learn anticipatory routing and should materially exceed the heuristic.

Usage:
    uv run python train/train_ppo_hard_multi.py

Output:
    trained_models/ppo_hard_multi_100k.zip   — saved SB3 model
    trained_models/ppo_hard_multi_100k_tb/   — TensorBoard logs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from train.gym_wrapper import BudgetRouterGymEnv
from budget_router.tasks import HARD_MULTI

# ── Config ──────────────────────────────────────────────────────────────────
N_ENVS      = 4
TOTAL_STEPS = 100_000           # Hard_Multi needs more signal than Easy
SAVE_PATH   = "trained_models/ppo_hard_multi_100k"
LOG_PATH    = "trained_models/ppo_hard_multi_100k_tb"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[train:hard_multi] device={DEVICE}  n_envs={N_ENVS}  total_steps={TOTAL_STEPS:,}")
    print("[train:hard_multi] Scenario: Provider A degrades step 0, Provider B degrades step 10")
    print("[train:hard_multi] Heuristic baseline grader: 0.6094  (reactive, cannot conserve budget)")

    train_env = make_vec_env(
        lambda: BudgetRouterGymEnv(scenario=HARD_MULTI),
        n_envs=N_ENVS,
    )

    eval_env = BudgetRouterGymEnv(scenario=HARD_MULTI, seed=99)

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=10,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,          # slightly higher entropy to encourage exploration on harder task
        learning_rate=3e-4,
        verbose=1,
        device=DEVICE,
        tensorboard_log=LOG_PATH,
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=eval_cb,
        progress_bar=True,
    )

    model.save(SAVE_PATH)
    print(f"[train:hard_multi] Model saved → {SAVE_PATH}.zip")


if __name__ == "__main__":
    main()
