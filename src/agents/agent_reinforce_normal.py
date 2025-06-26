# REINFORCE agent supporting different baselines:
# baseline = 0        → no baseline (vanilla REINFORCE)
# baseline = 20/100   → constant baseline
# baseline = mean     → moving average reward baseline (optional in code)

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import os
import random
import gym


def discount_rewards(r, gamma):
    """
    Compute discounted rewards G_t for each timestep:
    G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# =============== POLICY NETWORK (Actor-only) ===============
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64  # hidden layer size
        self.tanh = torch.nn.Tanh()

        # --- Actor MLP: maps state → mean of Gaussian over actions
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # --- Learnable std deviation (shared across actions)
        self.sigma_activation = F.softplus  # ensures std > 0
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of MLP layers with normal distribution,
        and biases to 0, for more stable training.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns a Normal (Gaussian) distribution over actions π(a|s)
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist


# ================= REINFORCE AGENT =================
class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99

        # Trajectory buffers (1 episode)
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        # === Baseline Options ===
        self.baseline = 0  # ← this is REINFORCE without baseline
        # self.baseline = 20  # ← REINFORCE with constant baseline = 20
        # self.baseline = 100 # ← REINFORCE with constant baseline = 100
        # (for mean-reward baseline, see update_policy comment below)

        self.tot_rewards = []  # To track all past rewards for avg baseline

    def update_policy(self):
        """
        REINFORCE update:
        ∇J(θ) ≈ ∇ log π_θ(a_t | s_t) * (G_t - b)
        """
        # Stack all episode trajectories
        action_log_probs = torch.stack(self.action_log_probs).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # === Compute discounted returns G_t ===
        discounted_returns = discount_rewards(rewards, self.gamma)

        # Optional: for mean-reward baseline
        self.tot_rewards.extend(rewards.cpu().numpy().flatten().tolist())
        # self.baseline = np.mean(self.tot_rewards)  # ← enable for moving average baseline

        # === Policy Loss ===
        # Loss = -∑ log π(a|s) * (G_t - baseline)
        policy_loss = (-action_log_probs * (discounted_returns - self.baseline)).sum()

        # === Backprop ===
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
            return normal_dist.mean, None  # deterministic: mean of π
        else:
            action = normal_dist.sample()
            # For multi-dimensional actions, use total log prob: log π(a_1,...,a_d) = sum log π(a_i)
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """
        Store (s_t, a_t, r_t, s_{t+1}) for policy update after episode.
        """
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
