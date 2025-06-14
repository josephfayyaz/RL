"""
Finding the best set of hyperparameters in the source_env->source_env configuration, using a wandb random sweep for 30 runs.
The model with the best hyperparameters is trained for 350k timesteps (three seeds) and saved as "best_model_ppo_ss_350".
"""

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import wandb
from functools import partial
import csv
import os

def main():
    source_env = gym.make('CustomHopper-source-v0')
    source_env = Monitor(source_env)

    best_parameters = {
        'best_mean_reward': 0,
        'best_std_reward': 0,
        'lr': 0,
        'n_steps': 0,
        'gamma': 0,
        'batch_size': 0,
        'n_epochs': 0,
        'gl': 0
    }

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "mean_mean_reward",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {"min": 0.00003, "max": 0.003},
            "gamma": {"min": 0.9, "max": 0.9999},
            "batch_size": {"min": 32, "max": 128},
            "n_epochs": {"min": 5, "max": 20},
            "gae_lambda": {"min": 0.85, "max": 0.999}
        }
    }

    sweep_id_ss = wandb.sweep(sweep_config, project="ppo_sweep_ss__5")

    # CSV setup
    csv_filename = "../Logs/PPO/ppo_sweep_log.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "run_id", "seed", "learning_rate", "n_steps", "gamma", "batch_size", "n_epochs", "gae_lambda",
                "mean_reward", "std_reward"
            ])

    def train_and_evaluate(train_env, test_env):
        with wandb.init(config=sweep_config):
            config = wandb.config

            lr = config.learning_rate
            gamma = config.gamma
            bs = round(config.batch_size)
            nsteps = bs * 32
            nepochs = round(config.n_epochs)
            gl = config.gae_lambda

            print("learning_rate:", lr)
            print("n_steps:", nsteps)
            print("gamma:", gamma)
            print("batch_size:", bs)
            print("n_epochs:", nepochs)
            print("gae_lambda:", gl)

            results = []

            for seed in [0, 14, 42]:
                model = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=nsteps, gamma=gamma,
                            batch_size=bs, n_epochs=nepochs, gae_lambda=gl, seed=seed, verbose=0)
                model.learn(total_timesteps=350000)
                mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, deterministic=True)
                results.append((model, mean_reward, std_reward))

            mean_rewards = [r[1] for r in results]
            std_rewards = [r[2] for r in results]

            mean_mean_reward = sum(mean_rewards) / len(mean_rewards)
            mean_std_reward = sum(std_rewards) / len(std_rewards)

            wandb.log({"mean_mean_reward": mean_mean_reward})

            # Save best model
            if mean_mean_reward > best_parameters['best_mean_reward']:
                best_parameters['best_mean_reward'] = mean_mean_reward
                best_parameters['best_std_reward'] = mean_std_reward
                best_model = max(results, key=lambda r: r[1])[0]
                best_model.save("model/best_model_ppo_ss_350")

                best_parameters.update({
                    'lr': lr, 'n_steps': nsteps, 'gamma': gamma, 'batch_size': bs,
                    'n_epochs': nepochs, 'gl': gl
                })

            # Log to CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for seed, (_, r_mean, r_std) in zip([0, 14, 42], results):
                    writer.writerow([
                        wandb.run.id, seed, lr, nsteps, gamma, bs, nepochs, gl, r_mean, r_std
                    ])

    # Wrap for wandb sweep
    p_train_and_evaluate = partial(train_and_evaluate, source_env, source_env)
    wandb.agent(sweep_id_ss, p_train_and_evaluate, count=30)

    print("Best mean reward:", best_parameters['best_mean_reward'])
    print("Best std reward:", best_parameters['best_std_reward'])
    print("Best hyperparameters:", best_parameters)

if __name__ == '__main__':
    main()




# Best mean reward: 527.1323770533332
# Best std reward: 23.189308784236726
# Best hyperparameters: {'best_mean_reward': 527.1323770533332, 'best_std_reward': 23.189308784236726, 'lr': 8.234655384179927e-05, 'n_steps': 2400, 'gamma': 0.9337442569785808, 'batch_size': 75, 'n_epochs': 16, 'gl': 0.9970783069680812}
