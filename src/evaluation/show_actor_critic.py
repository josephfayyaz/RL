"""Test an RL agent on the OpenAI Gym Hopper environment"""
import sys
import os
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

import torch
import gym
from env import *

from env.custom_hopper import *
from agents.agent_ac import Agent_ac as Agent, Policy_ac as Policy
from project_paths import MODELS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(MODELS_DIR / 'actor_critic' / 'model_actor_critic_source_8_5M.mdl'), type=str, help='Model path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of evaluation episodes')
    parser.add_argument('--domain', default='source', choices=['source', 'cdr', 'udr', 'target'], help='Environment domain')
    return parser.parse_args()



def main():
    args = parse_args()
    env = gym.make(f'CustomHopper-{args.domain}-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model, map_location=args.device), strict=True)

    agent = Agent(policy, device=args.device)

    for episode in range(args.episodes):
        done = False
        test_reward = 1
        state = env.reset()


        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")


if __name__ == '__main__':
    main()
