"""
Train PPO on CustomHopper with optional (i) domain–randomization
(UDR / CDR) and (ii) entropy scheduling.

Usage
------
$ python -m src.training.train_ppo_variants \
        --domain cdr \                  #  source | udr | cdr
        --entropy-schedule True \       #  toggle ES
        --total-timesteps 5_000_000 \
        --seed 0 14 42

Directory layout is controlled by SAVE_PATH / LOG_PATH so it fits the
repository skeleton discussed in the README.
"""
from __future__ import annotations

# ───────────────────────────── imports ──────────────────────────────
import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# local imports (require `pip install -e .` or PYTHONPATH=.)
from envs.custom_hopper import CustomHopper
from utils.schedules import EntropyScheduler, CurriculumScheduler

# ─────────────────────── CLI / experiment flags ─────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--device",           type=str, default="cuda")
    p.add_argument("--domain",           type=str, choices=["source", "udr", "cdr"],
                   default="source",
                   help="Mass–randomization regime used during training.")
    p.add_argument("--entropy-schedule", type=lambda x: x.lower() == "true",
                   default=True, metavar="BOOL")
    p.add_argument("--n-envs",           type=int, default=4,
                   help="Parallel workers (SubprocVecEnv).")
    p.add_argument("--seed",             type=int, nargs="+", default=[0],
                   help="One or more seeds.")
    # file/output options (rarely changed)
    p.add_argument("--hp-file",          default="models/ppo/best_hyperparameters.json")
    p.add_argument("--save-path",        default="models/ppo")
    p.add_argument("--log-path",         default="logs/csv/ppo")
    return p.parse_args()


ARGS = parse_args()

# select GPU if requested and available
DEVICE = "cuda:0" if ARGS.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
#                              Helper callbacks
# -----------------------------------------------------------------------------
class EpisodeCSVLogger(BaseCallback):
    """Log (episode, reward) whenever an episode terminates."""
    def __init__(self, csv_file: Path):
        super().__init__()
        self.csv_file = csv_file
        self.ep_count = 0
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "reward"])

    def _on_step(self) -> bool:  # noqa: D401
        for info in self.locals.get("infos", []):
            if (ep := info.get("episode")) is not None:
                self.ep_count += 1
                with open(self.csv_file, "a", newline="") as f:
                    csv.writer(f).writerow([self.ep_count, f"{ep['r']:.6f}"])
        return True


class LearningCurveLogger(BaseCallback):
    """Evaluate on the *target* env every eval_interval steps and log to CSV."""
    def __init__(self, eval_env: gym.Env, csv_file: Path,
                 eval_interval: int = 5_000, n_eval_episodes: int = 5):
        super().__init__()
        self.eval_env, self.csv_file = eval_env, csv_file
        self.eval_interval, self.n_eval_episodes = eval_interval, n_eval_episodes
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "mean_reward"])

    def _on_step(self) -> bool:  # noqa: D401
        if self.num_timesteps % self.eval_interval == 0:
            mean_rew, _ = evaluate_policy(self.model, self.eval_env,
                                          n_eval_episodes=self.n_eval_episodes,
                                          deterministic=True)
            with open(self.csv_file, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, f"{mean_rew:.6f}"])
        return True


class SaveAllBest(EvalCallback):
    """Keeps *every* new best model instead of overwriting a single file."""
    def __init__(self, eval_env: gym.Env, save_dir: Path,
                 eval_freq: int, name_prefix: str, **kwargs):
        super().__init__(eval_env, best_model_save_path=str(save_dir),
                         eval_freq=eval_freq, **kwargs)
        self.prefix = name_prefix

    def _on_step(self) -> bool:  # noqa: D401
        prev_best = getattr(self, "best_mean_reward", float("-inf"))
        proceed   = super()._on_step()
        if getattr(self, "best_mean_reward", prev_best) > prev_best:
            src = Path(self.best_model_save_path) / "best_model.zip"
            dst = Path(self.best_model_save_path) / f"{self.prefix}_{self.num_timesteps}_steps.zip"
            shutil.copyfile(src, dst)
        return proceed


# -----------------------------------------------------------------------------
#                      Environment factory (for vec env)
# -----------------------------------------------------------------------------
def make_env(domain: str, rank: int, seed: int):
    def _init():
        env = CustomHopper(domain=domain, total_timesteps=ARGS.total_timesteps)
        env = Monitor(env)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env
    return _init


# -----------------------------------------------------------------------------
#                              Training routine
# -----------------------------------------------------------------------------
def train_one_seed(run_seed: int) -> None:
    print(f"\n=== Training Domain={ARGS.domain.upper()} "
          f"ES={ARGS.entropy_schedule} Seed={run_seed} ===")

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    # path setup
    tag = f"{ARGS.domain}_es{int(ARGS.entropy_schedule)}_seed{run_seed}"
    save_dir = Path(ARGS.save_path) / tag
    log_dir  = Path(ARGS.log_path)  / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # vectorized env
    venv = SubprocVecEnv([make_env(ARGS.domain, i, run_seed)
                          for i in range(ARGS.n_envs)])

    # target eval env (always "target" domain)
    eval_env = DummyVecEnv([lambda: Monitor(CustomHopper(domain="target",
                                                         total_timesteps=ARGS.total_timesteps))])

    # load winning hyperparameters
    with open(ARGS.hp_file) as f:
        hp = json.load(f)

    model = PPO("MlpPolicy", venv, device=DEVICE,
                learning_rate=hp["learning_rate"],
                n_steps=hp["n_steps"],
                batch_size=hp["batch_size"],
                n_epochs=hp["n_epochs"],
                gamma=hp["gamma"],
                gae_lambda=hp["gae_lambda"],
                seed=run_seed,
                verbose=1,
                tensorboard_log=str(log_dir))

    # Callbacks -------------------------------------------------------
    callbacks = [
        # periodic checkpoints
        CheckpointCallback(save_freq=500_000 // ARGS.n_envs,
                           save_path=str(save_dir),
                           name_prefix="ckpt"),
        # keep *all* new best models
        SaveAllBest(eval_env, save_dir,
                    eval_freq=50_000 // ARGS.n_envs,
                    name_prefix="best",
                    n_eval_episodes=10,
                    deterministic=True,
                    verbose=0),
        # episode-level log
        EpisodeCSVLogger(csv_file=log_dir / "episode_returns.csv"),
        # smoothed learning curve
        LearningCurveLogger(eval_env,
                            csv_file=log_dir / "learning_curve.csv",
                            eval_interval=5_000,
                            n_eval_episodes=5),
    ]

    # optional schedules
    if ARGS.entropy_schedule:
        callbacks.append(
            EntropyScheduler(start_coef=0.01,
                             end_coef=1e-4,
                             total_timesteps=ARGS.total_timesteps)
        )
    if ARGS.domain == "cdr":
        callbacks.append(
            CurriculumScheduler(bounds_start=0.05,   # ±5 percent
                                bounds_end=0.40,     # ±40 percent
                                total_timesteps=ARGS.total_timesteps)
        )

    # ----------------------------------------------------------------
    model.learn(total_timesteps=ARGS.total_timesteps, callback=callbacks)
    model.save(save_dir / f"ppo_final_{tag}.zip")
    print(f"✔ Finished → {save_dir}")


def main() -> None:
    for s in ARGS.seed:
        train_one_seed(s)


if __name__ == "__main__":
    main()
