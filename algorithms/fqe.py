"""
Fitted Q-Evaluation (FQE) for off-policy policy evaluation.

Given a target policy π and a dataset D = {(s,a,r,s')}, FQE learns Q^π(s,a)
by iteratively fitting:
    Q_{k+1}(s,a) = r + γ * (1-d) * Q_k(s', π(s'))

The estimated policy value is E_{s~D}[Q^π(s, π(s))].
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FQENetwork(nn.Module):
    """Twin Q-network for FQE (reduces overestimation)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_num=2):
        super().__init__()

        def build_q():
            layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
            for _ in range(hidden_num - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q()
        self.q2 = build_q()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class FQE:
    """Fitted Q-Evaluation for off-policy policy value estimation.

    Args:
        obs_dim: observation dimensionality
        action_dim: action dimensionality
        hidden_dim: hidden layer size
        hidden_num: number of hidden layers
        discount: discount factor γ
        lr: learning rate
        tau: soft target update rate
        device: torch device
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_num=2,
                 discount=0.99, lr=3e-4, tau=0.005, device="cpu"):
        self.discount = discount
        self.tau = tau
        self.device = device

        self.q_net = FQENetwork(obs_dim, action_dim, hidden_dim, hidden_num).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).requires_grad_(False).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def train_step(self, batch, policy):
        """One FQE Bellman backup step.

        Args:
            batch: dict with keys observations, actions, rewards, next_observations, terminals
            policy: policy with .sample(obs) -> (action, log_prob, mean)

        Returns:
            q_loss (float)
        """
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(batch["terminals"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Get action from the TARGET policy (the one we're evaluating)
            next_action, _, _ = policy.sample(next_obs)
            target_q = self.target_q_net.q_min(next_obs, next_action)
            target_q = rewards + self.discount * (1.0 - terminals) * target_q

        q1, q2 = self.q_net(obs, actions)
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target
        for tp, sp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

        return loss.item()

    def estimate_value(self, buffer, policy, num_samples=10000):
        """Estimate V^π = E_s[Q^π(s, π(s))] over sampled states.

        Args:
            buffer: replay buffer with .sample(n) method
            policy: the policy to evaluate
            num_samples: how many states to average over

        Returns:
            estimated_value (float)
        """
        batch = buffer.sample(min(num_samples, buffer.size))
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = policy.sample(obs)
            q_val = self.q_net.q_min(obs, action)
        return q_val.mean().item()

    def fit(self, buffer, policy, num_steps=5000, batch_size=256):
        """Full FQE fitting loop.

        Args:
            buffer: replay buffer
            policy: the policy to evaluate
            num_steps: number of gradient steps
            batch_size: minibatch size

        Returns:
            estimated_value (float)
        """
        for step in range(num_steps):
            batch = buffer.sample(batch_size)
            self.train_step(batch, policy)

        return self.estimate_value(buffer, policy)

    def reset(self):
        """Reset Q-network weights for evaluating a new policy."""
        self.q_net._init_weights()
        self.target_q_net = copy.deepcopy(self.q_net).requires_grad_(False).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.optimizer.defaults['lr'])
