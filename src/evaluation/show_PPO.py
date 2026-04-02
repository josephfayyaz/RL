import sys
import os
import ctypes
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# mujoco_path = "C:/.mujoco/mujoco210/bin"  # manually append library for running on windoes
# os.environ["PATH"] += ";" + mujoco_path
# ctypes.CDLL(os.path.join(mujoco_path, "mujoco210.dll"))
import gym
from src.env import *

from stable_baselines3 import PPO

from project_paths import MODELS_DIR
# RL project
# Import your custom environment if necessary
from env.custom_hopper import *
# from env.custom_hopper_saghal import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(MODELS_DIR / 'PPO' / 'vanilla' / 'PPO_source_ES_False_seed_0_CustomHopper_source_v0_1000000.zip'),
                        type=str, help='Model path')

    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='evaluation the simulator')
    parser.add_argument('--episodes', default=500, type=int, help='Number of test episodes')
    parser.add_argument('--domain', default='source', choices=['source', 'cdr', 'udr', 'target'], help='Environment domain')

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
    ENV_ID = f'CustomHopper-{args.domain}-v0'
    MODEL_PATH = args.model
    NUM_EPISODES = args.episodes

    test_saved_model("PPO", ENV_ID, MODEL_PATH, NUM_EPISODES)
