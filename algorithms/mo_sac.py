import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.util import update_exponential_moving_average, compute_cosine_similarity, FrankWolfeSolver, \
    vector_to_gradients, get_mode, compute_kl_similarity


class SAC(nn.Module):
    def __init__(self, critic_r, critic_u, policy, policy_optimizer_ctor, critic_optimizer_ctor, alpha=0.2,
                 log_alpha_max=0.0, discount=0.99, target_update_rate=0.005, ref_vector=np.array([0.5, 0.5]),
                 max_coeff=np.array([100, 100]), psi=-0.999, device="cpu"):
        super().__init__()
        self.critic_r = critic_r.to(device)
        self.target_critic_r = copy.deepcopy(critic_r).requires_grad_(False)
        self.critic_u = critic_u.to(device)
        self.target_critic_u = copy.deepcopy(critic_u).requires_grad_(False)
        self.policy = policy.to(device)
        self.q_optimizer_r = critic_optimizer_ctor(self.critic_r.parameters())
        self.q_optimizer_u = critic_optimizer_ctor(self.critic_u.parameters())
        self.policy_optimizer = policy_optimizer_ctor(self.policy.parameters())
        self._log_alpha_max = log_alpha_max
        self._discount = discount
        self.target_update_rate = target_update_rate
        self.ref_vector = ref_vector
        self.max_coeff = torch.from_numpy(max_coeff).float().to(device)
        self.max_norm = 1.0
        self.psi = psi
        self.device = device
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor(alpha).to(device)

    def update_critic(self, data):
        observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(data["next_observations"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(data["rewards"], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(data["terminals"], dtype=torch.float32, device=self.device)
        uncertainties = torch.as_tensor(data["uncertainties"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_observations)

            qf1_next_target_r, qf2_next_target_r = self.target_critic_r(next_observations, next_state_action)
            min_qf_next_target_r = torch.min(qf1_next_target_r, qf2_next_target_r) - self._alpha.detach() * next_state_log_pi
            target_q_r = rewards + self._discount * (1. - terminals.float()) * min_qf_next_target_r

            qf1_next_target_u, qf2_next_target_u = self.target_critic_u(next_observations, next_state_action)
            min_qf_next_target_u = torch.min(qf1_next_target_u, qf2_next_target_u) - self._alpha.detach() * next_state_log_pi
            target_q_u = uncertainties + self._discount * (1. - terminals.float()) * min_qf_next_target_u

        # Update Q function for reward
        qf1_r, qf2_r = self.critic_r(observations, actions)
        qf1_loss_r = F.mse_loss(qf1_r, target_q_r)
        qf2_loss_r = F.mse_loss(qf2_r, target_q_r)
        self.q_optimizer_r.zero_grad(set_to_none=True)
        (qf1_loss_r + qf2_loss_r).backward()
        nn.utils.clip_grad_norm_(self.critic_r.parameters(), max_norm=self.max_norm, norm_type=2)
        self.q_optimizer_r.step()

        # Update Q function for uncertainty
        qf1_u, qf2_u = self.critic_u(observations, actions)
        qf1_loss_u = F.mse_loss(qf1_u, target_q_u)
        qf2_loss_u = F.mse_loss(qf2_u, target_q_u)
        self.q_optimizer_u.zero_grad(set_to_none=True)
        (qf1_loss_u + qf2_loss_u).backward()
        nn.utils.clip_grad_norm_(self.critic_u.parameters(), max_norm=self.max_norm, norm_type=2)
        self.q_optimizer_u.step()

        # Update target Q network
        update_exponential_moving_average(self.target_critic_r, self.critic_r, self.target_update_rate)
        update_exponential_moving_average(self.target_critic_u, self.critic_u, self.target_update_rate)

    def update_policy(self, data, ref_vector=None):
        observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=self.device)

        a, log_probs, _ = self.policy.sample(observations)

        # policy loss for reward
        qf1_r, qf2_r = self.critic_r(observations, a)
        min_qf_pi_r = torch.min(qf1_r, qf2_r)
        policy_loss_r = ((self._alpha.detach() * log_probs) - min_qf_pi_r).mean()

        # policy loss for uncertainty
        qf1_u, qf2_u = self.critic_u(observations, a)
        min_qf_pi_u = torch.min(qf1_u, qf2_u)
        policy_loss_u = ((self._alpha.detach() * log_probs) - min_qf_pi_u).mean()

        # calculate for ref-vector
        if ref_vector is None:
            ref_vector = torch.from_numpy(self.ref_vector).float().to(self.device)
        else:
            ref_vector = torch.from_numpy(ref_vector).float().to(self.device)
        policy_loss_c = -compute_cosine_similarity(policy_loss_r / self.max_coeff[0], policy_loss_u / self.max_coeff[1],
                                                   ref_vector.detach() / self.max_coeff)

        if policy_loss_c > self.psi and policy_loss_r > 0 and policy_loss_u > 0:
            # correction stage
            if (policy_loss_r / policy_loss_u) < (ref_vector[0] / ref_vector[1]):
                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_loss_r.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
                self.policy_optimizer.step()
            else:
                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_loss_u.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
                self.policy_optimizer.step()
        else:
            # ascending stage
            # get the gradient for reward
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss_r.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
            grads_reward = []
            for name, param in self.policy.named_parameters():
                grad = param.grad
                grads_reward.append(grad.reshape(-1))
            grads_reward = torch.cat(grads_reward)

            # get the gradient for uncertainty
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss_u.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
            grads_uncertainty = []
            for name, param in self.policy.named_parameters():
                grad = param.grad
                grads_uncertainty.append(grad.reshape(-1))
            grads_uncertainty = torch.cat(grads_uncertainty)

            # get the gradient for ref-vector
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss_c.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
            grads_ref = []
            for name, param in self.policy.named_parameters():
                grad = param.grad
                grads_ref.append(grad.reshape(-1))
            grads_ref = torch.cat(grads_ref)

            g_stack_gpu = torch.stack([grads_reward, grads_uncertainty, grads_ref])
            g_stack_np = g_stack_gpu.detach().cpu().numpy()
            alphas = FrankWolfeSolver(g_stack_np)
            stamp = 'r+u+c'
            g_hat = alphas[0] * grads_reward + alphas[1] * grads_uncertainty + alphas[2] * grads_ref
            vector_to_gradients(g_hat, self.policy.parameters())
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm, norm_type=2)
            self.policy_optimizer.step()

        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha.exp() * (log_probs.detach() + self._target_entropy).detach()).mean()
            self._alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self._log_alpha, 0.05)
            self._alpha_optim.step()
            with torch.no_grad():
                self._log_alpha.clamp_(max=self._log_alpha_max)
            self._alpha = self._log_alpha.detach().exp()

        return policy_loss_r.detach().cpu().numpy(), policy_loss_u.detach().cpu().numpy(), policy_loss_c.detach().cpu().numpy()

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
