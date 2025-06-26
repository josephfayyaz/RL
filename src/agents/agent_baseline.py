# === REINFORCE Agent ===
# This agent implements the REINFORCE algorithm, optionally using a baseline:
# baseline = 0      → pure REINFORCE
# baseline = const  → constant baseline
# baseline = mean   → average reward as baseline

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import os
import random
import gym


def discount_rewards(r, gamma):
    """
    Computes discounted return:
        G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# === Policy Network (Actor Only) ===
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learnable standard deviation for Gaussian policy π(a|s)
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        # Initialize weights for stability
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass: returns a Gaussian distribution over actions.
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist


# === REINFORCE Agent Class ===
class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99  # discount factor

        # Experience buffers
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        # Choose baseline here:
        # self.baseline = 0       # No baseline
        self.baseline = 20       # Constant baseline = 20
        # self.baseline = 100     # Constant baseline = 100
        # For mean-reward baseline, see update_policy()

        self.tot_rewards = []  # Track all returns for mean-reward baseline

    def update_policy(self):
        """
        Update the policy using REINFORCE:
            ∇J(θ) = E[ ∇ log π_θ(a|s) * (G_t - b) ]
        """

        # Stack episode buffers to tensors
        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # --- Compute discounted returns G_t ---
        discounted_returns = discount_rewards(rewards, self.gamma)

        # Accumulate total reward for computing mean-reward baseline
        self.tot_rewards.extend(rewards.cpu().numpy().flatten().tolist())

        # Uncomment to use mean baseline:
        # self.baseline = np.mean(self.tot_rewards)

        # --- Compute policy loss ---
        # L = -∑ log π(a|s) * (G_t - baseline)
        policy_loss = (-action_log_probs * (discounted_returns - self.baseline)).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode buffer
        self.action_log_probs = []
        self.rewards = []

        return

    def get_action(self, state, evaluation=False):
        """
        Sample or return mean action from π(a|s)
        """
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy(x)

        if evaluation:
            return normal_dist.mean, None  # Deterministic action
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()  # log(π(a))
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """
        Save experience tuple for training after episode.
        """
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
