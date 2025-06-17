import os , sys
import csv
import gym
import json
import wandb
import numpy as np
import torch
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback, CallbackList
from RL.env.custom_hopper import CustomHopper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *
import argparse
from datetime import datetime

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n-episodes', default=2000, type=int, help='Number of training episodes')
#     parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
#     parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
#     parser.add_argument('--algorithm' , default='PPO' ,type=str, choices=['PPO'], help='Algorithm to use for training')
#     parser.add_argument('--CDR', default=False, type=bool, choices=[True,False], help='toggle to use Curriculum Domain Randomization')
#     parser.add_argument('--Entropy_Scheduling', default=False, type=bool, choices=[True,False], help='toggle to use Entropy Scheduling')
#     parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
#     parser.add_argument('--n_envs', default=8, type=int, help='Number of parallel environments for training')
#
#     return parser.parse_args()
#
#
# args = parse_args()


# ---------- Config ---------- #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
HP_PATH = "models/PPO/best_hyperparameters.json"
ENV_TRAIN = 'CustomHopper-udr-v0'
ENV_TEST = 'CustomHopper-target-v0'
SAVE_PATH = f"./modelsPPO/{timestamp}"
SEEDS = [0,14,42]
TOTAL_TIMESTEPS = 1_000_000
EpisodeBasedReward_CSV = f"Logs/src_tgt_PPO_UDR_EpisodeBasedReward_CSV_{timestamp}.csv"
Learning_curve_CSV = f"Logs/src_tgt_PPO_UDR_learning_curve_{timestamp}_{TOTAL_TIMESTEPS}.csv"
CSV_OUT = f"Logs/csv/src_tgt_PPO_UDR_{timestamp}_{TOTAL_TIMESTEPS}.csv"
Domain = ["source","cdr","udr"]

# ---------- Episode-based reward ---------- #
class EpisodeBasedRewardCallback(BaseCallback):
    """
    Logs (episode, reward) to a CSV file at each episode end.
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode = 0

        # write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
        print(f"CSV logger initialized at {self.csv_path}")

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            ep_info = info.get('episode')
            if ep_info is not None:
                self.episode += 1
                reward = ep_info['r']
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode, f"{reward:.6f}"])
        return True

# ---------- average return every N timesteps ( learning curves ) ---------- #
class LearningCurveCallback(BaseCallback):
    """
    Logs (timesteps, mean_reward) to a CSV file every `eval_interval` steps.
    """

    def __init__(self, eval_env, csv_path, eval_interval=25000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes

        # write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timesteps', 'mean_reward'])

        if self.verbose > 0:
            print(f"EvalLogger initialized at {self.csv_path}")

    def _on_step(self):
        if self.num_timesteps % self.eval_interval == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, f"{mean_reward:.6f}"])

            if self.verbose > 0:
                print(f"[EvalLogger] Step {self.num_timesteps} → MeanReward: {mean_reward:.2f}")
        return True

def load_best_hyperparameters(path):
    with open(path, 'r') as f:
        return json.load(f)

# ---------- Utils ---------- #
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_env(env_id, seed, rank):
    def _init():
        env = CustomHopper(
            domain=Domain,
            total_timesteps=Total_timesteps
        )
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return Monitor(env)
    return _init

def main():
    assert os.path.exists(HP_PATH), "Best hyperparameters file not found. Run ppo_hyperparam_sweep_source_eval first."
    os.makedirs("Logs/csv", exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)

    best_hp = load_best_hyperparameters(HP_PATH)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": TOTAL_TIMESTEPS,
        "env_id_train": ENV_TRAIN,
        "env_id_test": ENV_TEST,
        **best_hp
    }

    run = wandb.init(project='ppo_final_train_transfer', config=config, sync_tensorboard=True)

    # Train 3 models with different seeds
    results = []
    models = []

    for i, seed in enumerate(SEEDS):
        set_seed(seed)
        print(f"\n--- Training PPO model with seed {i} - {seed} ---")

        # Training and test environments
        train_env = SubprocVecEnv([make_env(ENV_TRAIN, seed, i) for i in range(8)])
        test_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_TEST))])
        test_env.seed(seed)

        log_path = f"runs/{timestamp}/{run.id}/seed_{seed}" if i == 2 else None  # log only last one

        model = PPO("MlpPolicy", train_env,
                    learning_rate=best_hp["learning_rate"],
                    n_steps=best_hp["n_steps"],
                    gamma=best_hp["gamma"],
                    batch_size=best_hp["batch_size"],
                    n_epochs=best_hp["n_epochs"],
                    gae_lambda=best_hp["gae_lambda"],
                    seed=seed,
                    verbose=2,
                    tensorboard_log=log_path)


        episode_based_reward_logger = EpisodeBasedRewardCallback(csv_path=EpisodeBasedReward_CSV)
    
        learning_curve_logger = LearningCurveCallback(
            eval_env=test_env,  # evaluate on source env
            csv_path=Learning_curve_CSV,  # separate CSV for this
            eval_interval=10000,
            n_eval_episodes=5,
            verbose=1
        )

        callback = WandbCallback(model_save_path=f'{SAVE_PATH}/{run.id}/seed_{seed}',
                             verbose=2) if i == 2 else None

        checkpoint_callback = CheckpointCallback(
            save_freq=1000, save_path=f'{SAVE_PATH}/{run.id}/seed_{seed}', name_prefix='PPO_UDR'
        )

        eval_callback = EvalCallback(
            test_env, best_model_save_path=f'{SAVE_PATH}/{run.id}/seed_{seed}',
            log_path=log_path, eval_freq=5000,
            deterministic=True, render=False
        )

        callbacks = CallbackList([
            checkpoint_callback,
            eval_callback,
            episode_based_reward_logger,
            learning_curve_logger
        ])

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

        # ✅ Evaluate on *target* (different) environment
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True)
        print(f"Seed {seed} → Test env reward: {mean_reward:.2f}, Std: {std_reward:.2f}")

        results.append((mean_reward, std_reward))
        models.append(model)

        with open(CSV_OUT, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([seed, mean_reward, std_reward])

    run.finish()

    # Compute average performance on test env
    mean_mean_reward = sum(r[0] for r in results) / len(results)
    mean_std_reward = sum(r[1] for r in results) / len(results)

    print("\n=== Final PPO Evaluation (1M x 3 seeds on target env) ===")
    for i, seed in enumerate(SEEDS):
        print(f"Seed {seed}: Reward = {results[i][0]:.2f}, Std = {results[i][1]:.2f}")
    print(f"\nAverage reward: {mean_mean_reward:.2f}")
    print(f"Average std: {mean_std_reward:.2f}")

    # Save best model (trained on source, best tested on target)
    best_model_idx = max(range(len(results)), key=lambda i: results[i][0])
    best_model = models[best_model_idx]
    best_model.save(f"{SAVE_PATH}/best_model_ppo_source_target_{TOTAL_TIMESTEPS}")
    print(f"\n✅ Best model (seed {SEEDS[best_model_idx]}) saved to {SAVE_PATH}/best_model_ppo_source_target")

if __name__ == "__main__":
    main()
