# Agent for Actor-Critic

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

torch.autograd.set_detect_anomaly(True)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy_ac(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigma_activation = F.softplus
        init_sigma = 0.75
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        # Critic
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Actor forward
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        # Critic forward
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        return normal_dist, state_value


class Agent_ac(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.entropy_beta = 0.01  # encourage exploration
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1).float()
        done = torch.tensor(self.done, dtype=torch.float32, device=self.train_device)

        # Optional reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Bootstrapped target: R + gamma * V(s')
        _, state_values = self.policy(states)
        state_values = state_values.squeeze(-1).float()

        _, next_state_values = self.policy(next_states)
        next_state_values = next_state_values.squeeze(-1).float()

        targets = rewards + self.gamma * next_state_values * (1.0 - done)
        advantages = targets.detach() - state_values

        # Actor Loss
        entropy = -torch.sum(torch.stack([dist.entropy().mean() for dist, _ in [self.policy(s.unsqueeze(0)) for s in states]])) / len(states)
        actor_loss = (-action_log_probs * advantages.detach()).sum() - self.entropy_beta * entropy

        # Critic Loss
        critic_loss = F.mse_loss(state_values, targets.detach())

        # Total loss
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Clear buffers
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        return actor_loss.item(), critic_loss.item(), entropy.item()

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, state_value = self.policy(x)

        if evaluation:
            return normal_dist.mean, None
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward]))
        self.done.append(done)
