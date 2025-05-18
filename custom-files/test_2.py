"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent_2 import Agent, Policy


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default=None, type=str, help='Model path')
#     parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
#     parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
#     parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
#
#     return parser.parse_args()
#
#
# args = parse_args()

model= "/home/joseph/python-proj/1/custom-files/model-normal/model-6.mdl"
device= "cuda"
render= "True"
episodes= 30



def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(model), strict=True)

    agent = Agent(policy, device=device)

    for episode in range(episodes):
        done = False
        test_reward = 1
        state = env.reset()


        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")


if __name__ == '__main__':
    main()