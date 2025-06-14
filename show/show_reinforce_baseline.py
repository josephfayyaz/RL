"""Test an RL agent on the OpenAI Gym Hopper environment"""


import torch
from env.custom_hopper import *
from agents.agent_baseline import Agent_states as Agent, Policy_states as Policy , BaselineNetwork


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

model= "/home/joseph/python-proj/1/custom-files/Models/model_reinforce_baseline/model_reinforce_baseline_2_100K.mdl"
device= "cuda"
render= "True"
episodes= 100



def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    baseline_network = BaselineNetwork(observation_space_dim)
    # Load only the policy weights (you trained and saved only the policy)
    policy.load_state_dict(torch.load(model), strict=True)

    # Construct the agent with both
    agent = Agent(policy, baseline_network , device=device)
    # agent = Agent(policy, baseline_network,device="cuda")
    for episode in range(episodes):
        done = False
        test_reward = 0
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