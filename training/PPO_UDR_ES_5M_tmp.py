"""
Train a PPO policy on the MuJoCo Hopper with optional
• Uniform Domain Randomization (UDR)
• Curriculum Domain Randomization (CDR)
• Entropy Scheduling (ES)

Launch example
--------------
python PPO_UDR_ES_5M_tmp.py \
    --Domain cdr --Entropy_Scheduling True --seed 0 14 42
"""
# ────────────────────────────── stdlib ──────────────────────────────
import os, sys, shutil, csv, random, argparse, multiprocessing
from datetime import datetime
# ────────────────────────────── 3rd-party ──────────────────────────
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    )
from stable_baselines3.common.evaluation import evaluate_policy
# ───────────────────────── project imports ─────────────────────────
# repository root/ → add parent dir so “src” is importable when the
# script is run from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.custom_hopper import CustomHopper
from src.utils.schedules import EntropyScheduler          # ← NEW
# -------------------------------------------------------------------

# ╭──────────────────────────── CLI ─────────────────────────────╮
def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser()
    p.add_argument('--n-episodes', type=int, default=5_000_000,
                   help='Total environment steps (= timesteps)')
    p.add_argument('--print-every', type=int, default=100,
                   help='Stdout progress every N episodes')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                   help='Torch device (auto-selects cuda:0 if available)')
    p.add_argument('--algorithm', choices=['PPO'], default='PPO')
    p.add_argument('--Domain', choices=['source', 'cdr', 'udr'],
                   default='cdr',
                   help='Which mass-randomization regime the *training* env uses')
    p.add_argument('--Entropy_Scheduling', action='store_true',
                   help='Enable entropy-decay schedule')
    p.add_argument('--seed', type=int, nargs='+', default=[0, 14, 42],
                   help='List of seeds to iterate over')
    p.add_argument('--n_envs', type=int, default=8,
                   help='Parallel workers (SubprocVecEnv)')
    return p.parse_args()

args = parse_args()
# ╰──────────────────────────────────────────────────────────────╯

# ──────────────────────── global constants ──────────────────────
TOTAL_TIMESTEPS = args.n_episodes
DEVICE = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

HP_PATH   = "../models/ppo/best_hyperparameters.json"   # Optuna output
SAVE_PATH = "../models/ppo"
LOG_PATH  = "../logs/ppo"
ENV_ID    = f"CustomHopper-{args.Domain}-v0"
EVAL_ENV  = "CustomHopper-target-v0"

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

# ────────────────────────── utilities ───────────────────────────
def load_best_hyperparameters(path: str) -> dict:
    """Load a JSON file written by Optuna / grid search."""
    import json
    with open(path, 'r') as fp:
        return json.load(fp)

def make_env(env_id: str, base_seed: int, rank: int):
    """Return a thunk for SubprocVecEnv so each worker has its own seed."""
    def _init():
        env = CustomHopper(domain=args.Domain, total_timesteps=TOTAL_TIMESTEPS)
        seed = base_seed + rank
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return Monitor(env)
    return _init

# ─────────────────────── logging callbacks ──────────────────────
class CSVLoggerCallback(BaseCallback):
    """Write (episode, reward) rows after every finished episode."""
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode  = 0
        # header
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'reward'])
        if verbose:
            print(f"[CSVLogger] Writing to {self.csv_path}")

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if (ep := info.get('episode')) is not None:
                self.episode += 1
                with open(self.csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([self.episode, f"{ep['r']:.6f}"])
        return True

class LearningCurveCallback(BaseCallback):
    """Evaluate the frozen policy every *eval_interval* steps and log CSV."""
    def __init__(self, eval_env, csv_path, eval_interval=10_000,
                 n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['timesteps', 'mean_reward'])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_interval == 0:
            mean_r, _ = evaluate_policy(self.model, self.eval_env,
                                        n_eval_episodes=self.n_eval_episodes,
                                        deterministic=True)
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([self.num_timesteps, f"{mean_r:.6f}"])
            if self.verbose:
                print(f"[Eval] {self.num_timesteps:,} → {mean_r:.2f}")
        return True

class SaveAllBestCallback(EvalCallback):
    """Every time EvalCallback beats the best score, dup the file with a suffix."""
    def __init__(self, eval_env, best_model_save_path, log_path,
                 prefix="best_model", eval_freq=50_000, **kwargs):
        super().__init__(eval_env=eval_env,
                         best_model_save_path=best_model_save_path,
                         log_path=log_path,
                         eval_freq=eval_freq,
                         deterministic=True,
                         render=False,
                         **kwargs)
        self.prefix = prefix

    def _on_step(self) -> bool:
        prev_best = getattr(self, "best_mean_reward", float("-inf"))
        continue_training = super()._on_step()
        # If a new best was found → duplicate the zip with a timestamp suffix
        if getattr(self, "best_mean_reward", float("-inf")) > prev_best:
            src = os.path.join(self.best_model_save_path, "best_model.zip")
            dst = os.path.join(
                self.best_model_save_path,
                f"{self.prefix}_{self.num_timesteps}_steps.zip")
            shutil.copyfile(src, dst)
        return continue_training

# ────────────────────────── training loop ───────────────────────
def train_agent(seed: int) -> None:
    """One full experiment for a single random seed."""
    print(f"\n═════════════  SEED {seed}  ═════════════")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build vectorized env
    vec_env = SubprocVecEnv([make_env(ENV_ID, seed, i)
                             for i in range(args.n_envs)])

    # Build evaluation env
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(EVAL_ENV))])
    eval_env.seed(seed)

    # CSV loggers ----------------------------------------------------------------
    csv_train = os.path.join(
        LOG_PATH, f"{args.algorithm}_{args.Domain}_ES{args.Entropy_Scheduling}"
                  f"_seed{seed}_{TOTAL_TIMESTEPS}_episodes.csv")
    csv_curve = os.path.join(
        LOG_PATH, f"learning_curve_{args.algorithm}_{args.Domain}"
                  f"_ES{args.Entropy_Scheduling}_seed{seed}.csv")

    callbacks = [
        CSVLoggerCallback(csv_train),
        LearningCurveCallback(eval_env, csv_curve, eval_interval=5_000, verbose=1),
        SaveAllBestCallback(eval_env,
                            best_model_save_path=SAVE_PATH,
                            log_path=LOG_PATH,
                            eval_freq=50_000//args.n_envs,
                            prefix=f"best_{args.Domain}_ES{args.Entropy_Scheduling}_seed{seed}",
                            n_eval_episodes=10,
                            verbose=0),
        CheckpointCallback(save_freq=50_000//args.n_envs,
                           save_path=SAVE_PATH,
                           name_prefix='rl_model')
    ]

    # ─────────────── Block B: add custom curriculum / extra callbacks here ──────
    # For example:
    # from src.utils.curriculum import MassBoundScheduler
    # callbacks.append(MassBoundScheduler(total_timesteps=TOTAL_TIMESTEPS))
    # ─────────────────────────────────────────────────────────────────────────────

    # Optional entropy schedule
    if args.Entropy_Scheduling:
        callbacks.append(
            EntropyScheduler(start_coef=0.01,
                             end_coef=1.0e-4,
                             total_timesteps=TOTAL_TIMESTEPS)
        )

    # Load hyper-parameters tuned offline
    hp = load_best_hyperparameters(HP_PATH)

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        device=DEVICE,
        seed=seed,
        verbose=1,
        tensorboard_log=LOG_PATH,
        **hp,                      # unpack learning_rate, n_steps, …
    )

    # ───────────── learn & save ─────────────
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    out_name = (f"{args.algorithm}_{args.Domain}_ES{args.Entropy_Scheduling}"
                f"_seed{seed}_{TOTAL_TIMESTEPS}_steps")
    model.save(os.path.join(SAVE_PATH, out_name))
    print(f"[✓] Finished seed {seed}, model saved → {out_name}.zip")

# ──────────────────────────── main guard ─────────────────────────
def main() -> None:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    print(f"Running on {'GPU '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    for seed in args.seed:
        train_agent(seed)

if __name__ == "__main__":
    main()
