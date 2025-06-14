"""
    Train a model in the source environment with the same hyperparameters found in the source -> source configuration.
    Evaluate the model on the source environment every 50 episode
"""

import gym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)

    best_hp = {
        'lr': 0.0012541462101916157,
        'gamma': 0.9885202328222382,
        'batch_size': 123,
        'n_epochs': 11,
        'gl': 0.9472233613269306,
        'nsteps': 3936}

    config = {"policy_type": "MlpPolicy", "total_timesteps": 1_000_000, "env_id": source_env}  # for the wandb run

    run = wandb.init(project='mldl_2024_template', config=config, sync_tensorboard=True)  # inizialize Wandb

    # train and evaluate three model with different seeds [0, 14, 42]
    model_1 = PPO("MlpPolicy", source_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'],
                  gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'],
                  gae_lambda=best_hp['gl'], seed=0, verbose=2)
    model_1.learn(total_timesteps=1_000_000)
    mean_reward_1, std_reward_1 = evaluate_policy(model_1, source_env, n_eval_episodes=50, deterministic=True)

    model_2 = PPO("MlpPolicy", source_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'],
                  gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'],
                  gae_lambda=best_hp['gl'], seed=14, verbose=2)
    model_2.learn(total_timesteps=1_000_000)
    mean_reward_2, std_reward_2 = evaluate_policy(model_2, source_env, n_eval_episodes=50, deterministic=True)

    model_3 = PPO("MlpPolicy", source_env, learning_rate=best_hp['lr'], n_steps=best_hp['nsteps'],
                  gamma=best_hp['gamma'], batch_size=best_hp['batch_size'], n_epochs=best_hp['n_epochs'],
                  gae_lambda=best_hp['gl'], seed=42, verbose=2, tensorboard_log=f'runs/{run.id}')
    model_3.learn(total_timesteps=1_000_000, callback=WandbCallback(model_save_path=f'models/{run.id}',
                                                                    verbose=2))  # visualize last model trainig curve with wandb callback
    mean_reward_3, std_reward_3 = evaluate_policy(model_3, source_env, n_eval_episodes=50, deterministic=True)

    run.finish()

    print(f"Mean reward 1: {mean_reward_1}, Mean reward 2: {mean_reward_2}, Mean reward 3: {mean_reward_3}")
    # Mean reward 1: 1153.18382046, Mean reward 2: 1401.9460041800003, Mean reward 3: 1612.98179253998

    mean_mean_reward = (mean_reward_1 + mean_reward_2 + mean_reward_3) / 3
    mean_std_reward = (std_reward_1 + std_reward_2 + std_reward_3) / 3

    print("Mean reward: ", mean_mean_reward)
    print("Mean std: ", mean_std_reward)

    """
        1M
        Mean Reward: 1389.3705390599935
        Std Reward: 156.28460931540724
    """

    # save best model wrt the mean reward
    models = [(model_1, mean_reward_1), (model_2, mean_reward_2), (model_3, mean_reward_3)]
    best_model, best_mean_reward = max(models, key=lambda item: item[1])
    best_model.save("./models/best_model_ppo_ss")


if __name__ == '__main__':
    main()