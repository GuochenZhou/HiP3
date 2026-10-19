import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import update_exponential_moving_average


class SAC(nn.Module):
    def __init__(self, critic, policy, policy_optimizer_ctor, critic_optimizer_ctor, alpha=0.2, discount=0.99, target_update_rate=0.005, device="cpu"):
        super().__init__()
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(critic).requires_grad_(False)
        self.policy = policy.to(device)
        self.q_optimizer = critic_optimizer_ctor(self.critic.parameters())
        self.policy_optimizer = policy_optimizer_ctor(self.policy.parameters())
        self._discount = discount
        self.target_update_rate = target_update_rate
        self.device = device
        self._log_alpha_max = 0.0
        self.max_norm = 1.0
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.exp()
        else:
            self._alpha = alpha

    def update_parameter(self, data):
        observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(data["next_observations"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(data["rewards"], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(data["terminals"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_observations)

            qf1_next_target, qf2_next_target = self.target_critic(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_state_log_pi
            target_q = rewards + self._discount * (1. - terminals.float()) * min_qf_next_target

        # Update Q function for reward
        qf1, qf2 = self.critic(observations, actions)
        qf1_loss = F.mse_loss(qf1, target_q)
        qf2_loss = F.mse_loss(qf2, target_q)

        self.q_optimizer.zero_grad(set_to_none=True)
        (qf1_loss + qf2_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_norm, norm_type=2)
        self.q_optimizer.step()

        a, log_probs, _ = self.policy.sample(observations)

        qf1, qf2 = self.critic(observations, a)
        min_qf_pi = torch.min(qf1, qf2)

        policy_loss = ((self._alpha * log_probs) - min_qf_pi).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
        self.policy_optimizer.step()

        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha.exp() * (log_probs.detach() + self._target_entropy).detach()).mean()
            self._alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            # nn.utils.clip_grad_norm_(self._log_alpha, 0.05)
            self._alpha_optim.step()
            with torch.no_grad():
                self._log_alpha.clamp_(min=-5.0, max=self._log_alpha_max)
            self._alpha = self._log_alpha.exp()

        # Update target Q network
        update_exponential_moving_average(self.target_critic, self.critic, self.target_update_rate)

        return policy_loss.detach().cpu().numpy()
    
    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action_mode = self.policy.sample(state)
                result = action_mode.cpu().numpy()
            else:
                action_sample, _, _ = self.policy.sample(state)
                result = action_sample.cpu().numpy()
        return result[0] if single else result

    def get_alpha(self):
        return self._alpha
