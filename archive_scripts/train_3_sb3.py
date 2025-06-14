"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""

import gym
from env.custom_hopper import *
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
#     parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
#     parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
#     parser.add_argument('--algorithm', default='PPO', type=str, choices=['PPO'], help='Algorithm to use for training')
#     return parser.parse_args()
#

# args = parse_args()

N_episodes = 300000
print_time = 500
device = "cuda"
main_algorithm= 'PPO'


def train_agent(algo, env_id, total_timesteps, save_path, log_path):
    env = gym.make(env_id)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,
                                             name_prefix='rl_model')

    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                                 log_path=log_path, eval_freq=5000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save(os.path.join(save_path, f"{algo}_final_model"))


if __name__ == "__main__":
    ALGO = main_algorithm
    ENV_ID = 'CustomHopper-source-v0'  # Change to your specific environment
    TIMESTEPS = N_episodes
    SAVE_PATH = 'modelsPPO/'
    LOG_PATH = './logs/'

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    train_agent(ALGO, ENV_ID, TIMESTEPS, SAVE_PATH, LOG_PATH)