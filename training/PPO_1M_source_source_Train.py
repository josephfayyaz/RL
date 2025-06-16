"""
Train a model in the source environment with the same hyperparameters found in the source -> source configuration.
Evaluate the model on the source environment every 50 episodes.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym

from env.custom_hopper import *

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    print("🟢 Initializing Source Environment...")
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)

    best_hp = {
        'learning_rate': 0.0012541462101916157,
        'gamma': 0.9885202328222382,
        'batch_size': 123,
        'n_epochs': 11,
        'gae_lambda': 0.9472233613269306,
        'n_steps': 3936
    }

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 400_000,
        "env_id": "CustomHopper-source-v0"
    }

    print("📡 Initializing Weights & Biases...")
    run = wandb.init(project='mldl_2024_template', config=config, sync_tensorboard=True)

    print("\n🚀 Training PPO model with seed=0...")
    model_1 = PPO("MlpPolicy", source_env, **best_hp, seed=0, verbose=2)
    model_1.learn(total_timesteps=400_000)
    print("✅ Evaluating model_1...")
    mean_reward_1, std_reward_1 = evaluate_policy(model_1, source_env, n_eval_episodes=50, deterministic=True)

    print("\n🚀 Training PPO model with seed=14...")
    model_2 = PPO("MlpPolicy", source_env, **best_hp, seed=14, verbose=2)
    model_2.learn(total_timesteps=400_000)
    print("✅ Evaluating model_2...")
    mean_reward_2, std_reward_2 = evaluate_policy(model_2, source_env, n_eval_episodes=50, deterministic=True)

    print("\n🚀 Training PPO model with seed=42...")
    model_3 = PPO("MlpPolicy", source_env, **best_hp, seed=42, verbose=2, tensorboard_log=f'runs/{run.id}')
    model_3.learn(total_timesteps=400_000, callback=WandbCallback(model_save_path=f'models/{run.id}', verbose=2))
    print("✅ Evaluating model_3...")
    mean_reward_3, std_reward_3 = evaluate_policy(model_3, source_env, n_eval_episodes=50, deterministic=True)

    run.finish()

    print("\n📊 Evaluation Results:")
    print(f"Mean reward 1: {mean_reward_1:.2f}")
    print(f"Mean reward 2: {mean_reward_2:.2f}")
    print(f"Mean reward 3: {mean_reward_3:.2f}")

    mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3) / 3
    mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3) / 3

    print(f"\n📈 Average of 3 Seeds → Mean Reward: {mean_mean_reward:.2f}, Std: {mean_std_reward:.2f}")

    print("💾 Saving best-performing model to ./Models/PPO/best_model_ppo_ss_400...")
    models = [(model_1, mean_reward_1), (model_2, mean_reward_2), (model_3, mean_reward_3)]
    best_model, _ = max(models, key=lambda item: item[1])
    best_model.save("./Models/PPO/best_model_ppo_ss_400")

    print("✅ Training complete!")


if __name__ == '__main__':
    main()
