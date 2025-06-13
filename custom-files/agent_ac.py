# Agent for Actor-Critic
# Based on agent.py from https://github.com/gabrieletiboni/mldl_2024_template

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

torch.autograd.set_detect_anomaly(True)  # Used for debugging phase


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Methods that learn approximation to both policy and value function are called
# Actor-Critic methods
# the first two layers create a representation of the state and the two output layers map the
# representation to desired outputs. A larger network would be able to hold a better representation but
# would take longer to train
class Policy_ac(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network

            Actor - A reference to the learned policy
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        # Apply a linear transformation from in_features to out_features
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.75
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
            Critic network

            Critic - Refers to the learned value-function, usually a state-value function
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)  # only one: state-value function

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    # The policy is parametrized through a neural network
    def forward(self, x):
        """
            Actor - Reference to the learned policy [13.1]
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)

        normal_dist = Normal(action_mean, sigma)

        """
            Critic - Refers to the value function (usually a state-value function) [13.1]
            Not present in the REINFORCE algorithm - Comment out everything below for REINFORCE, and
            make sure it's returning only normal_dist
        """

        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        # output actor (action probabilities) and critic (state value)
        return normal_dist, state_value


class Agent_ac(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        discounted_returns = discount_rewards(rewards, self.gamma)

        # compute estimate state values for the observed state
        _, state_values = self.policy(states)  # computing the critic network
        state_values = state_values.squeeze(-1)  # removing unnecessary dimensions

        # calculating expected return for next state
        _, state_values_next = self.policy(next_states)
        state_values_next = state_values_next.squeeze(-1)
        # BOOTSTRAPPED RETURNS
        # note that if done = 1 this is a terminal step, therefore the discounted returns remain the same
        # note that for AC i only have to add the reward of the single step, instead of the whole return
        state_values_next = rewards + self.gamma * state_values_next * (1.0 - done)

        baseline = state_values  # baseline changes according to the state value in the MDP, less variance that constant
        advantages = state_values_next - baseline  # r_(t+1) + gamma*state_values_next - state_values
        actor_loss = (-action_log_probs * advantages).sum()

        critic_loss = F.mse_loss(state_values, state_values_next.detach())
        actor_critic_loss = actor_loss + critic_loss

        # first we zero grad to avoid gradient accumulation.
        # by performing backward() we compute the gradients of the
        # policy parameters with respect to this loss, and optimizer.step() updates the
        # parameters in the direction that minimizes this loss
        self.optimizer.zero_grad()
        actor_critic_loss.backward()
        self.optimizer.step()

        self.action_log_probs = []
        self.rewards = []
        self.states = []
        self.next_states = []
        self.done = []

        return

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:  # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

