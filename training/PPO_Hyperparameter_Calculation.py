import os
import csv
import gym
import json
import torch
import numpy as np
from datetime import datetime
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import env.custom_hopper  # Ensure this file exists and is correct

# ---------- Configuration ---------- #
ENV_TRAIN = 'CustomHopper-source-v0'
ENV_TEST = 'CustomHopper-source-v0'
SEEDS = [0,14,42]
TOTAL_TIMESTEPS = 350_000
SAVE_HP_PATH = "../Models/PPO/best_hyperparameters.json"
LOG_CSV = "Logs/PPO/ppo_hyperparam_sweep_source_eval.csv"
WANDB_PROJECT = "ppo_sweep_ss"

# ---------- Sweep Configuration ---------- #
sweep_config = {
    "method": "random",
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 3e-5, "max": 3e-3},
        "gamma": {"min": 0.9, "max": 0.9999},
        "batch_size": {"min": 32, "max": 128},
        "n_epochs": {"min": 5, "max": 20},
        "gae_lambda": {"min": 0.85, "max": 0.999}
    }
}

# ---------- Logging Setup ---------- #
os.makedirs("Logs", exist_ok=True)
os.makedirs("modelsPPO", exist_ok=True)
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "seed", "learning_rate", "n_steps", "gamma",
            "batch_size", "n_epochs", "gae_lambda", "mean_reward", "std_reward"
        ])

# ---------- Utils ---------- #
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------- Create parallel environments ---------- #
def make_vec_env(env_id, n_envs=8, seed=None):
    def make_env(rank):
        def _init():
            env = gym.make(env_id)
            env = Monitor(env)
            if seed is not None:
                env.seed(seed + rank)
            return env
        return _init
    return SubprocVecEnv([make_env(i) for i in range(n_envs)])


# ---------- Sweep Function ---------- #
def train_and_evaluate(train_env, test_env):
    with wandb.init(config=sweep_config, project=WANDB_PROJECT):
        config = wandb.config

        lr = config.learning_rate
        gamma = config.gamma
        bs = round(config.batch_size)
        nsteps = bs * 32
        nepochs = round(config.n_epochs)
        gl = config.gae_lambda

        print("--- Hyperparameters ---")
        print(f"lr={lr}, gamma={gamma}, batch_size={bs}, n_steps={nsteps}, n_epochs={nepochs}, gae_lambda={gl}")

        mean_rewards, std_rewards = [], []
        best_model, best_mean = None, -np.inf

        for seed in SEEDS:
            set_seed(seed)
            train_env.seed(seed)
            test_env.seed(seed)

            model = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=nsteps, gamma=gamma,
                        batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=seed, verbose=0)

            model.learn(total_timesteps=TOTAL_TIMESTEPS)

            mean_r, std_r = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True)

            mean_rewards.append(mean_r)
            std_rewards.append(std_r)

             # Log results
            # wandb.log({"mean_reward": mean_r})

            with open(LOG_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    wandb.run.id, seed, lr, nsteps, gamma, bs, nepochs, gl, mean_r, std_r
                ])

            if mean_r > best_mean:
                best_mean = mean_r
                best_model = model

        mean_mean_reward = np.mean(mean_rewards)
        wandb.log({"mean_mean_reward": mean_mean_reward})

        # Save model and config
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # model.save(f"./modelsPPO/best_model_ppo_ss_350_{timestamp}")
        print(f"Model saved with reward {mean_r:.2f}")

        best_hyperparameters = {
            "learning_rate": lr,
            "gamma": gamma,
            "batch_size": bs,
            "n_epochs": nepochs,
            "gae_lambda": gl,
            "n_steps": nsteps
        }
        with open(SAVE_HP_PATH, 'w') as f:
            json.dump(best_hyperparameters, f, indent=4)


# ---------- Main ---------- #
def main():
    dummy_train_env = make_vec_env(ENV_TRAIN, n_envs=8)
    test_env = gym.make(ENV_TEST)

    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=partial(train_and_evaluate, dummy_train_env, test_env), count=30)

if __name__ == "__main__":
    main()
