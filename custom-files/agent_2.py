import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus

        # init_sigma = 0.5
        init_sigma = 1
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)  # First hidden layer for Critic
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)  # Second hidden layer for Critic
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)            # Output layer for Critic


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))  # First hidden layer with Tanh activation
        x_critic = self.tanh(self.fc2_critic(x_critic))  # Second hidden layer with Tanh activation
        state_value = self.fc3_critic(x_critic)  # Output layer producing the state value


        return normal_dist , state_value


class Agent(object):
    def __init__(self, policy, device='cuda'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.baseline = 20.0
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.update_policy_methods = {
        'reinforce': self.update_policy_reinforce,
        'reinforce_baseline': self.update_policy_reinforce_baseline,
        'actor_critic': self.update_policy_actor_critic
        }

    def update_policy_reinforce(self, action_log_probs, rewards):
        discounted_returns = discount_rewards(rewards, self.gamma)
        # Normalize returns
        returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
        # Compute policy gradient loss
        policy_loss = -torch.sum(action_log_probs * returns)
        return policy_loss

    def update_policy_reinforce_baseline(self, action_log_probs, rewards):
        discounted_returns = discount_rewards(rewards, self.gamma)
        # Subtract the baseline
        adjusted_returns = discounted_returns - self.baseline
        # Normalize the adjusted returns
        adjusted_returns = (adjusted_returns - adjusted_returns.mean()) / (adjusted_returns.std() + 1e-8)
        # Compute policy gradient loss
        policy_loss = -torch.sum(action_log_probs * adjusted_returns)
        return policy_loss

    def update_policy_actor_critic(self, action_log_probs, rewards):
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        discounted_returns = discount_rewards(rewards, self.gamma)
        values, next_values = self.get_values(states, next_states)  # Get current and next state values from Critic
        advantages = discounted_returns - values.squeeze(-1)
        critic_loss = F.mse_loss(values.squeeze(-1), discounted_returns)

        policy_loss = -(action_log_probs * advantages.detach()).sum()  # Compute Actor loss

        total_loss = policy_loss + critic_loss  # Combine losses

        return total_loss

    def get_values(self, states, next_states):
        values = []
        next_values = []
        for state, next_state in zip(states, next_states):
            _, value = self.policy(state)  # Get value for the current state
            _, next_value = self.policy(next_state)  # Get value for the next state
            values.append(value)
            next_values.append(next_value)
        return torch.stack(values), torch.stack(next_values)  # Return stacked values

    def update_policy(self, algorithm):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        policy_loss = self.update_policy_methods[algorithm](action_log_probs, rewards)


        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        # return



    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
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
        # The next line is added :
        # self.values.append(value)
