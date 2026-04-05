"""
Train a PPO agent on the Easy scenario using stable-baselines3.

Usage:
    uv run python train/train_ppo.py

Output:
    trained_models/ppo_easy_50k.zip   — saved SB3 model
    trained_models/ppo_easy_50k/      — TensorBoard logs

Config:
    N_ENVS      = 4          parallel environments
    TOTAL_STEPS = 50_000     timesteps
    DEVICE      = mps        Apple Silicon GPU (falls back to cpu if unavailable)
"""
from __future__ import annotations

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from train.gym_wrapper import BudgetRouterGymEnv
from budget_router.tasks import EASY

# ── Config ──────────────────────────────────────────────────────────────────
N_ENVS      = 4
TOTAL_STEPS = 50_000
SAVE_PATH   = "trained_models/ppo_easy_50k"
LOG_PATH    = "trained_models/ppo_easy_50k_tb"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[train] device={DEVICE}  n_envs={N_ENVS}  total_steps={TOTAL_STEPS:,}")

    # Vectorized training envs (4 parallel, stateless reset each episode)
    train_env = make_vec_env(
        lambda: BudgetRouterGymEnv(scenario=EASY),
        n_envs=N_ENVS,
    )

    # Separate eval env (single, deterministic)
    eval_env = BudgetRouterGymEnv(scenario=EASY, seed=99)

    # SB3 recommends stopping training once a reward threshold is hit to
    # prevent over-fitting. For Easy, heuristic gets ~7.88 mean reward.
    # We target > 6.0 as a sanity threshold (PPO may need more steps for parity).
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=8.5, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        eval_freq=max(5_000 // N_ENVS, 1),
        n_eval_episodes=10,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=512,          # rollout buffer size per env
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,        # small entropy bonus encourages exploration
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
    print(f"[train] Model saved → {SAVE_PATH}.zip")


if __name__ == "__main__":
    main()
