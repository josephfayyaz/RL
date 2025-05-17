"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

# from env.custom_hopper import *
from env.custom_hopper_saghal import *
from agent_2 import Agent, Policy


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
#     parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
#     parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
#     parser.add_argument('--algorithm', default='reinforce', type=str, choices=['reinforce', 'reinforce_baseline', 'actor_critic'], help='Algorithm to use for training [reinforce, reinforce_baseline]')
#     return parser.parse_args()
#
#
# args = parse_args()
#

N_episodes = 10000
print_time = 500
device = "cuda"
main_algorithm= 'reinforce_baseline'

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

    for episode in range(N_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over

            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward

        agent.update_policy(main_algorithm)  # This line added

        if (episode + 1) % print_time == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)

    torch.save(agent.policy.state_dict(), "model-5-custom-saghal.mdl")


if __name__ == '__main__':
    main()

