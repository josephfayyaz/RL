# ------------------ Imports ------------------ #
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

# Add parent directory to Python path so that 'src.env.custom_hopper' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.env.custom_hopper  # Custom MuJoCo Hopper environment (must be implemented correctly)

# ------------------ Configuration ------------------ #
ENV_TRAIN = 'CustomHopper-source-v0'
ENV_TEST = 'CustomHopper-source-v0'
SEEDS = [0, 14, 42]  # Multiple seeds for robustness across different training runs
TOTAL_TIMESTEPS = 350_000  # Total training steps per run
SAVE_HP_PATH = "../../models/PPO/best_hyperparameters.json"  # Where to store the best found hyperparameters
LOG_CSV = "../../Logs/PPO_sweep/ppo_hyperparam_sweep_source_eval.csv"  # Logging path for sweep results
WANDB_PROJECT = "ppo_sweep_ss"  # WandB project name

# ------------------ Sweep Configuration for wandb ------------------ #
# Random search over hyperparameter space. Goal is to maximize the mean reward.
sweep_config = {
    "method": "random",  # You can use "grid", "random", or "bayes"
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 3e-5, "max": 3e-3},     # Controls how quickly the agent updates. Too large => instability
        "gamma": {"min": 0.9, "max": 0.9999},            # Discount factor for future rewards: R_t = r_t + γ*r_{t+1} + ...
        "batch_size": {"min": 32, "max": 128},           # Number of samples per policy update
        "n_epochs": {"min": 5, "max": 20},               # Number of epochs to optimize the surrogate loss
        "gae_lambda": {"min": 0.85, "max": 0.999}        # Smoothing factor for Generalized Advantage Estimation
    }
}

# ------------------ Logging Directories ------------------ #
# Create directories if they don't exist
os.makedirs("Logs", exist_ok=True)
os.makedirs("modelsPPO", exist_ok=True)

# If CSV doesn't exist, initialize it with header
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "seed", "learning_rate", "n_steps", "gamma",
            "batch_size", "n_epochs", "gae_lambda", "mean_reward", "std_reward"
        ])

# ------------------ Utilities ------------------ #
def set_seed(seed):
    # Ensure reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ------------------ Vectorized Environment Creation ------------------ #
def make_vec_env(env_id, n_envs=8, seed=None):
    # Parallel environment creation using subprocesses
    def make_env(rank):
        def _init():
            env = gym.make(env_id)
            env = Monitor(env)  # Logs episode statistics
            if seed is not None:
                env.seed(seed + rank)  # Different seed per subprocess
            return env
        return _init
    return SubprocVecEnv([make_env(i) for i in range(n_envs)])  # Create n_envs in parallel

# ------------------ Main Sweep Logic ------------------ #
def train_and_evaluate(train_env, test_env):
    with wandb.init(config=sweep_config, project=WANDB_PROJECT):
        config = wandb.config

        # Extract hyperparameters from current WandB config
        lr = config.learning_rate
        gamma = config.gamma
        bs = round(config.batch_size)
        nsteps = bs * 32  # n_steps = batch_size * 32 (typical PPO heuristic)
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

            # Initialize PPO model with MLP policy
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=lr,
                n_steps=nsteps,
                gamma=gamma,
                batch_size=bs,
                n_epochs=nepochs,
                gae_lambda=gl,
                seed=seed,
                verbose=0
            )

            # Train the model for the given timesteps
            model.learn(total_timesteps=TOTAL_TIMESTEPS)

            # Evaluate the trained model on the test environment
            mean_r, std_r = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True)

            mean_rewards.append(mean_r)
            std_rewards.append(std_r)

            # Append results to CSV log
            with open(LOG_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    wandb.run.id, seed, lr, nsteps, gamma, bs, nepochs, gl, mean_r, std_r
                ])

            # Track best model
            if mean_r > best_mean:
                best_mean = mean_r
                best_model = model

        # Log the average performance across seeds
        mean_mean_reward = np.mean(mean_rewards)
        wandb.log({"mean_mean_reward": mean_mean_reward})

        print(f"Model saved with reward {mean_r:.2f}")

        # Save best hyperparameters to JSON
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

# ------------------ Entry Point ------------------ #
def main():
    # Use SubprocVecEnv for training for efficiency
    dummy_train_env = make_vec_env(ENV_TRAIN, n_envs=8)
    test_env = gym.make(ENV_TEST)  # Single test env, no need for vectorization

    # Register sweep and run the train_and_evaluate function 30 times
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=partial(train_and_evaluate, dummy_train_env, test_env), count=30)

# Run the sweep experiment
if __name__ == "__main__":
    main()
