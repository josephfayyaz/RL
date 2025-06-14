import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from env.custom_hopper import *

from stable_baselines3 import PPO

import argparse
# RL project
# Import your custom environment if necessary
# from env.custom_hopper import *
# from env.custom_hopper_saghal import *


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--model', default='/home/joseph/python-proj/1/Models/PPO/best_model_ppo_ss_400.zip',
                        type=str, help='Model path')

    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=500, type=int, help='Number of test episodes')
    parser.add_argument('--algorithm', default='PPO', type=str, choices=['PPO'], help='Algorithm to use for training')

    return parser.parse_args()


args = parse_args()


def test_saved_model(algo, env_id, model_path, num_episodes=1000):
    env = gym.make(env_id)

    # Load the trained model
    model = PPO.load(model_path)

    # Test the model
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Reward = {episode_reward}")


if __name__ == "__main__":
    # Example usage
    ALGO = args.algorithm
    ENV_ID = 'CustomHopper-source-v0'  # Change to your specific environment
    MODEL_PATH = args.model  # './models/PPO_final_model.zip'  # Change to the path of your saved model
    NUM_EPISODES = 2000  # Number of episodes to test the agent

    test_saved_model(ALGO, ENV_ID, MODEL_PATH, NUM_EPISODES)