"""
Script for training a control policy on the Hopper environment
using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

Implements a configurable PPO training pipeline with logging, checkpointing,
multi-seed evaluation, entropy scheduling, and domain randomization.
"""

# -------------------- Imports -------------------- #
import os, sys, ctypes, random
import numpy as np
import gym
from datetime import datetime
import csv
import argparse
import torch
import json
import shutil
import multiprocessing
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Allow importing from parent directory (e.g., to access custom environments)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from env.custom_hopper import *  # Import custom MuJoCo Hopper environment
from project_paths import LOGS_DIR, MODELS_DIR

# -------------------- Argument Parser -------------------- #
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', '--timesteps', dest='n_episodes', default=1_000_000, type=int)  # Total training timesteps
    parser.add_argument('--print-every', default=100, type=int)  # Print frequency
    parser.add_argument('--device', default='cuda', type=str)  # Device to use
    parser.add_argument('--algorithm', default='PPO', type=str, choices=['PPO'])  # Algorithm (only PPO here)
    parser.add_argument('--domain', '--Domain', dest='domain', default='source', choices=['source', 'cdr', 'udr'])  # Domain type
    parser.add_argument('--entropy-scheduling', '--Entropy_Scheduling', dest='entropy_scheduling', default=False, type=str2bool)  # Entropy flag
    parser.add_argument('--seed', default=[0, 14, 42], type=int, nargs="+")  # Seeds to train on
    parser.add_argument('--n_envs', default=8, type=int)  # Number of parallel environments
    return parser.parse_args()

args = parse_args()

# -------------------- Config Setup -------------------- #
Total_timesteps = args.n_episodes

# Select device
device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
print(f"training on {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "training on cpu")

# Paths and logging directories
HP_PATH = MODELS_DIR / "PPO" / "best_hyperparameters.json"
ENV_ID = f'CustomHopper-{args.domain}-v0'  # Custom environment ID
EVAL_ENV = 'CustomHopper-source-v0'  # Evaluation always on source domain
PPO_MODELS_DIR = MODELS_DIR / "PPO"
SAVE_SUBDIR = (
    "vanilla" if args.domain == "source"
    else "cdr_es" if args.domain == "cdr" and args.entropy_scheduling
    else "cdr" if args.domain == "cdr"
    else "udr_es" if args.domain == "udr" and args.entropy_scheduling
    else "udr"
)
SAVE_PATH = PPO_MODELS_DIR / SAVE_SUBDIR
LOG_PATH = LOGS_DIR / "PPO_episode_rewards"
CHECKPOINT_PATH = PPO_MODELS_DIR / "checkpoints"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
USE_entropy_scheduler = args.entropy_scheduling
seeds = args.seed

# -------------------- Callback: Per-Episode Reward Logging -------------------- #
class CSVLoggerCallback(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode = 0
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'reward'])

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            ep_info = info.get('episode')
            if ep_info is not None:
                self.episode += 1
                reward = ep_info['r']
                with open(self.csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([self.episode, f"{reward:.6f}"])
        return True

# -------------------- Load Hyperparameters -------------------- #
def load_best_hyperparameters(path):
    with open(path, 'r') as f:
        return json.load(f)

# -------------------- Env Factory for SubprocVecEnv -------------------- #
def make_env(env_id, seed, rank):
    def _init():
        env = CustomHopper(domain=args.domain, total_timesteps=Total_timesteps)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return Monitor(env)
    return _init

# -------------------- Callback: Learning Curve Evaluation -------------------- #
class LearningCurveCallback(BaseCallback):
    def __init__(self, eval_env, csv_path, eval_interval=10000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['timesteps', 'mean_reward'])

    def _on_step(self):
        if self.num_timesteps % self.eval_interval == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([self.num_timesteps, f"{mean_reward:.6f}"])
            if self.verbose > 0:
                print(f"[EvalLogger] Step {self.num_timesteps} → MeanReward: {mean_reward:.2f}")
        return True

# -------------------- Callback: Save All Best Models -------------------- #
class SaveAllBestCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, prefix="best_model", n_eval_episodes=5, deterministic=True, render=False, verbose=0):
        super().__init__(eval_env, best_model_save_path, log_path, eval_freq, n_eval_episodes, deterministic, render, verbose)
        self.prefix = prefix

    def _on_step(self) -> bool:
        prev_best = getattr(self, "best_mean_reward", float("-inf"))
        cont = super()._on_step()
        if hasattr(self, "best_mean_reward") and self.best_mean_reward > prev_best:
            src = os.path.join(self.best_model_save_path, "best_model.zip")
            dst = os.path.join(self.best_model_save_path, f"{self.prefix}_{self.num_timesteps}_steps.zip")
            shutil.copyfile(src, dst)
        return cont

# -------------------- Train PPO Agent -------------------- #
def train_agent(algo, env_id, eval_env_id, USE_entropy_scheduler, total_timesteps, save_path, log_path, seed, csv_filename):
    best_hp = load_best_hyperparameters(HP_PATH)
    env = SubprocVecEnv([make_env(env_id, seed, i) for i in range(args.n_envs)])
    csv_logger = CSVLoggerCallback(csv_path=csv_filename)

    # Initialize PPO agent with loaded hyperparameters
    model = PPO('MlpPolicy', env, device=device,
                learning_rate=best_hp["learning_rate"],
                n_steps=best_hp["n_steps"],
                gamma=best_hp["gamma"],
                batch_size=best_hp["batch_size"],
                n_epochs=best_hp["n_epochs"],
                gae_lambda=best_hp["gae_lambda"],
                seed=seed,
                verbose=1,
                tensorboard_log=log_path)

    # Evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(eval_env_id))])
    eval_env.seed(seed)

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10**9 // args.n_envs,
        save_path=CHECKPOINT_PATH,
        name_prefix='rl_model'
    )

    learning_curve_cb = LearningCurveCallback(
        eval_env=eval_env,
        csv_path=LOGS_DIR / "Learning_Curve" / f"learning_curve_{algo}_{args.domain}_ES_{args.entropy_scheduling}_seed_{seed}_{total_timesteps}.csv",
        eval_interval=5000,
        n_eval_episodes=5,
        verbose=1
    )

    eval_callback = SaveAllBestCallback(
        eval_env=eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=10**9 // args.n_envs,
        prefix=f"EVAL_BEST_{algo}_{args.domain}_ES_{args.entropy_scheduling}_seed_{seed}",
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    callbacks = [checkpoint_callback, eval_callback, csv_logger, learning_curve_cb]

    # Optionally append entropy scheduler
    if USE_entropy_scheduler:
        callbacks.append(
            EntropyScheduler(
                start_coef=0.01, end_coef=1e-4,
                total_timesteps=total_timesteps
            )
        )

    # Train the PPO model
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save the trained model
    modelpath = os.path.join(save_path, f"{algo}_{args.domain}_ES_{args.entropy_scheduling}_seed_{seed}_({env_id}_{eval_env_id})_{total_timesteps}")
    model.save(modelpath)
    print(f"Model saved to {modelpath}")

# -------------------- Main Entry Point -------------------- #
def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(LOGS_DIR / "Learning_Curve", exist_ok=True)

    for run_seed in seeds:
        print(f"=== Running experiment with seed={run_seed} ===")
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        envname = args.domain
        evalenvname = "source"
        csv_filename = os.path.join(LOG_PATH, f"{args.algorithm}_{args.domain}_ES_{args.entropy_scheduling}_seed_{run_seed}_{Total_timesteps}({envname},{evalenvname}).csv")
        train_agent(args.algorithm, ENV_ID, EVAL_ENV, args.entropy_scheduling, Total_timesteps, SAVE_PATH, LOG_PATH, run_seed, csv_filename)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
