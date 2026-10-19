import os
import time
import copy
import numpy as np
import numpy.random
from numpy.random import default_rng
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gym
import d4rl
import logz
import tensorflow as tf

from models.tf_dynamics_models.fake_env import FakeEnv
from models.tf_dynamics_models.constructor import format_samples_for_training
from models.policy import GaussianPolicy, QNetwork
from algorithms.mo_sac import SAC
from utils.util import fast_calculate_pareto_fronts
from utils.offlinebuffer import ReplayBuffer
from static_fns.infos import get_scores
from filter import get_filter

LOG_SIG_MIN = -20


class Traineroffline(object):
    def __init__(
            self,
            dynamics_model,
            eval_env,
            static_fns,
            offline_buffer,
            reward_penalty_coef,
            rollout_length,
            rollout_batch_size,
            batch_size,
            log_dir,
            args
    ):
        self.dynamics_model = dynamics_model
        self.env = eval_env
        self.static_fns = static_fns
        self.fake_env = FakeEnv(
            self.dynamics_model,
            self.static_fns,
            penalty_coeff=reward_penalty_coef,
            penalty_learned_var=True
        )
        self.offline_buffer = offline_buffer
        self.model_buffer = None
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        # self._rollout_freq = int(rollout_batch_size * rollout_length)
        self._rollout_freq = 1000
        self._batch_size = batch_size
        self._eval_episodes = args.eval_episodes
        self._repeat_num = args.repeat_num
        self._num_pareto = args.num_pareto
        self._reset_interval = args.reset_interval
        self._update_per_step = args.update_per_step
        self._start_training = args.start_training
        self._log_dir = log_dir
        self.args = args
        self.rng = default_rng(args.seed)
        self.init_obs = self.offline_buffer.get_init_obs()
        self.logz = logz
        self.logz.configure_output_dir(log_dir)
        self.cur_policy = None
        self.r_filter = lambda x: np.clip(
            np.where(np.logical_or(np.isnan(x), np.isinf(x)), 0, x),
            0,
            20.0
        )
        self.u_filter = lambda x: np.clip(
            np.where(np.logical_or(np.isnan(x), np.isinf(x)), 25, x),
            0,
            25.0
        )

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self, policy, steps):
        init_transitions = self._sample_initial_transitions()
        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            if steps < self._start_training:
                actions = []
                for k in range(len(observations)):
                    actions.append(self.env.action_space.sample())
                actions = np.array(actions)
            else:
                # sample action with policy
                actions = policy.select_action(observations)
            next_observations, rewards, terminals, infos = self.fake_env.step(observations, actions)

            # clipping
            next_observations = np.clip(next_observations, -100, 100)
            unpenalized_rewards, uncertainties = \
                self.r_filter(infos['unpenalized_rewards']), self.u_filter(infos['penalty'])
            uncertainties = np.exp(-uncertainties / 1.5)
            self.model_buffer.add_batch(observations, next_observations, actions,
                                        unpenalized_rewards, terminals, uncertainties)

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

    def init_model_buffer(self):
        fake_batch = self.offline_buffer.sample_all()
        data = {
            "observations": fake_batch["observations"],
            "actions": fake_batch["actions"],
            "next_observations": fake_batch["next_observations"],
            "terminals": fake_batch["terminals"],
            "rewards": fake_batch["rewards"],
        }
        next_observations, rewards, terminals, infos = self.fake_env.step(data["observations"], data["actions"])

        # clipping
        next_observations = np.clip(next_observations, -100, 100)
        unpenalized_rewards, uncertainties = \
            self.r_filter(infos['unpenalized_rewards']), self.u_filter(infos['penalty'])
        uncertainties = np.exp(-uncertainties / 1.5)
        self.model_buffer.add_batch(data["observations"], next_observations, data["actions"],
                                    unpenalized_rewards, terminals, uncertainties)

    def save_dynamics_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dynamics_model.save(save_path, timestep=0)

    def train_dynamics(self):
        data = self.offline_buffer.sample_all()

        # generate dynamic model
        train_inputs, train_outputs = format_samples_for_training(data)
        max_epochs = 1 if self.dynamics_model.model_loaded else None
        loss = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self._batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2
        )
        self.save_dynamics_model(
            save_path=os.path.join(self._log_dir, "dynamics_model")
        )
        return loss

    def train_policy(self, ref_num, iter_g, iter_l):
        start_time = time.time()
        policy_pool = []
        ref_vectors = []
        fitnesses = np.zeros((self._num_pareto, 2))
        returns = np.zeros(self._num_pareto)
        normal_returns = np.zeros(self._num_pareto)
        _, expert_score = get_scores(self.args.task)

        max_coeff = np.array([LOG_SIG_MIN - expert_score / self._eval_episodes *
                              (1 - self.args.discount ** self._eval_episodes) / (1 - self.args.discount), LOG_SIG_MIN -
                              (1 - self.args.discount ** self._eval_episodes) / (1 - self.args.discount)])

        print("max_coeff", max_coeff)
        # get ref_vectors
        for i in range(ref_num):
            tau_c = (0.9 - 0.1) / (ref_num - 1)
            w_1 = 0.9 - i * tau_c
            w_2 = 0.1 + i * tau_c
            ref_vec = np.array([w_1, w_2])
            ref_vec *= max_coeff
            ref_vectors.append(ref_vec)
        print(ref_vectors)

        num_timesteps = 0
        num_p = 0
        max_reward, max_normal_reward = 0.0, 0.0

        model_buffer = ReplayBuffer(
            buffer_size=int(self.args.buffer_size),  # 1e6
            obs_shape=self.args.obs_shape,
            obs_dtype=np.float32,
            action_dim=self.args.action_dim,
            action_dtype=np.float32
        )
        self.model_buffer = model_buffer

        # Diverse policies
        for i in range(ref_num):
            w_1 = 0.9 - i * tau_c
            w_2 = 0.1 + i * tau_c

            ref_timesteps = 0

            # Diverse Pareto Policies
            for _ in range(iter_g):
                # reset agent
                if num_timesteps % self._reset_interval == 0:
                    # generate critic
                    critic_r = QNetwork(self.args.obs_dim, self.args.action_dim, hidden_dim=self.args.hidden_dim,
                                        hidden_num=self.args.hidden_num)
                    critic_u = QNetwork(self.args.obs_dim, self.args.action_dim, hidden_dim=self.args.hidden_dim,
                                        hidden_num=self.args.hidden_num)

                    # generate policy
                    policy = GaussianPolicy(self.args.obs_dim, self.args.action_dim,
                                            self.args.hidden_dim, self.args.hidden_num, self.args.action_space)

                    # generate alpha
                    if self.args.auto_alpha:
                        target_entropy = self.args.target_entropy if self.args.target_entropy \
                            else -np.prod(self.env.action_space.shape)

                        # log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
                        log_alpha = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True, device=self.args.device)
                        # alpha_optim = torch.optim.AdamW([log_alpha], lr=self.args.alpha_lr)
                        alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
                        self.args.alpha = (target_entropy, log_alpha, alpha_optim)

                    print("Current α:", self.args.alpha)

                    sac_policy = SAC(
                        critic_r=critic_r,
                        critic_u=critic_u,
                        policy=policy,
                        critic_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.critic_learning_rate),
                        policy_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.policy_learning_rate),
                        # critic_optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=self.args.critic_learning_rate, weight_decay=1e-4),
                        # policy_optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=self.args.policy_learning_rate, weight_decay=1e-4),
                        alpha=self.args.alpha,
                        log_alpha_max=self.args.log_alpha_max,
                        discount=self.args.discount,
                        target_update_rate=self.args.target_update_rate,
                        ref_vector=ref_vectors[i],
                        max_coeff=max_coeff,
                        psi=self.args.psi,
                        device=self.args.device
                    )
                    self.cur_policy = sac_policy

                # generate new data
                if num_timesteps % self._rollout_freq == 0:
                    self.rollout_transitions(self.cur_policy, ref_timesteps)

                if ref_timesteps < self._reset_interval:
                    data = self.model_buffer.sample(batch_size=self._batch_size)
                    self.cur_policy.update_critic(data)
                    r_loss, u_loss, c_loss = self.cur_policy.update_policy(data, ref_vectors[i])
                else:
                    for _ in range(self._update_per_step):
                        data = self.model_buffer.sample(batch_size=self._batch_size)
                        self.cur_policy.update_critic(data)
                        r_loss, u_loss, c_loss = self.cur_policy.update_policy(data, ref_vectors[i])
                if ((num_timesteps + 1) % self.args.present_freq_g == 0) or \
                        (num_timesteps % (iter_g + 2 * iter_l) == 0):
                    print("r_loss: ", r_loss, "u_loss: ", u_loss, "c_loss: ", c_loss)
                    eval_returns = np.array([self._evaluate_true(policy=self.cur_policy)
                                             for _ in range(self._repeat_num)])
                    normalized_epoch_returns = d4rl.get_normalized_score(self.args.task, eval_returns) * 100.0
                    max_reward = np.max([max_reward, np.mean(eval_returns)])
                    max_normal_reward = np.max([max_normal_reward, np.mean(normalized_epoch_returns)])

                    self.logz.log_tabular("Iterations", num_timesteps + 1)
                    self.logz.log_tabular("Time", time.time() - start_time)
                    self.logz.log_tabular("Ref_angle", np.arctan(w_2 / w_1) / np.pi * 180)
                    self.logz.log_tabular("Episode_return_mean", eval_returns.mean())
                    self.logz.log_tabular("Episode_return_std", eval_returns.std())
                    self.logz.log_tabular("Normalized_reward_mean", normalized_epoch_returns.mean())
                    self.logz.log_tabular("Normalized_reward_std", normalized_epoch_returns.std())
                    self.logz.log_tabular("Max_return_rollout", max_reward)
                    self.logz.log_tabular("Max_normalized_return_rollout", max_normal_reward)
                    if num_timesteps < iter_g + iter_l:
                        self.logz.log_tabular("Max_pool_normalized_return_rollout", max_normal_reward)
                    else:
                        self.logz.log_tabular("Max_pool_normalized_return_rollout", np.max(normal_returns))
                    self.logz.log_tabular("Min_pool_normalized_return_rollout", np.min(normal_returns))
                    self.logz.log_tabular("Std_pool_normalized_return", normal_returns.std())
                    if num_timesteps < iter_g + iter_l:
                        self.logz.log_tabular("Max_pool_return_rollout", max_reward)
                    else:
                        self.logz.log_tabular("Max_pool_return_rollout", np.max(returns))
                    self.logz.log_tabular("Min_pool_return_rollout", np.min(returns))
                    self.logz.log_tabular("Std_pool_return", returns.std())
                    self.logz.log_tabular("Alpha", self.cur_policy._alpha.detach().cpu().numpy())
                    self.logz.log_tabular("Loss_r", -r_loss)
                    self.logz.log_tabular("Loss_u", -u_loss)
                    self.logz.dump_tabular()

                    point_r, point_u, steps = self._evaluate_fake(policy=self.cur_policy)
                    if num_p >= self._num_pareto:
                        policy_array = np.concatenate(
                            (policy_array, np.expand_dims(copy.deepcopy(sac_policy).cpu(), axis=0)), axis=0)
                        fitnesses = np.concatenate((fitnesses, np.array([[point_r, point_u]])))
                        returns = np.append(returns, eval_returns.mean())
                        normal_returns = np.append(normal_returns, normalized_epoch_returns.mean())
                        assert policy_array.shape[0] == self._num_pareto + 1 and \
                               len(returns) == self._num_pareto + 1

                        # refine pareto front via non-dominated sorting
                        fronts = fast_calculate_pareto_fronts(fitnesses)

                        # selected fixed number of pareto fronts
                        flat_fronts = np.array([item for sublist in fronts for item in sublist])
                        policy_array = policy_array[flat_fronts[:self._num_pareto]]
                        fitnesses = fitnesses[flat_fronts[:self._num_pareto], :]
                        returns = returns[flat_fronts[:self._num_pareto]]
                        normal_returns = normal_returns[flat_fronts[:self._num_pareto]]

                    else:
                        policy_pool.append(copy.deepcopy(sac_policy).cpu())
                        fitnesses[num_p] = np.array([[point_r, point_u]])
                        policy_array = np.array(policy_pool)
                        returns[num_p] = np.mean(eval_returns)
                        normal_returns[num_p] = np.mean(normalized_epoch_returns)
                    num_p += 1

                num_timesteps += 1
                ref_timesteps += 1

            point_r, point_u, steps = self._evaluate_fake(policy=self.cur_policy)
            print("Fake env_reward: ", point_r, "Fake env uncertainty: ", point_u)

            # generate new ref vector
            for v in range(2):
                tau_c = (0.9 - 0.1) / (ref_num - 1)
                w_1 = 0.9 - i * tau_c
                w_2 = 0.1 + i * tau_c
                ref_vec = np.array([w_1, w_2]) + np.array([-5e-2 * (-1) ** v, 5e-2 * (-1) ** v])
                ref_vec *= max_coeff

                # generate new policies
                sac_policy = copy.deepcopy(self.cur_policy)

                # Local Pareto Extension
                for _ in range(iter_l):
                    # train policies
                    if num_timesteps % self._rollout_freq == 0:
                        self.rollout_transitions(sac_policy, ref_timesteps)
                    for _ in range(self._update_per_step):
                        data = self.model_buffer.sample(batch_size=self._batch_size)
                        sac_policy.update_critic(data)
                        r_loss, u_loss, c_loss = sac_policy.update_policy(data, ref_vec)

                    # store policy in the pool
                    if (num_timesteps + 1) % self.args.present_freq_l == 0:
                        print("r_loss: ", r_loss, "u_loss: ", u_loss, "c_loss: ", c_loss)
                        eval_returns = np.array([self._evaluate_true(policy=sac_policy)
                                                 for _ in range(self._repeat_num)])
                        normalized_epoch_returns = d4rl.get_normalized_score(self.args.task, eval_returns) * 100.0
                        max_reward = np.max([max_reward, np.mean(eval_returns)])
                        max_normal_reward = np.max([max_normal_reward, np.mean(normalized_epoch_returns)])
                        self.logz.log_tabular("Iterations", num_timesteps + 1)
                        self.logz.log_tabular("Time", time.time() - start_time)
                        self.logz.log_tabular("Ref_angle", np.arctan(w_2 / w_1) / np.pi * 180)
                        self.logz.log_tabular("Episode_return_mean", eval_returns.mean())
                        self.logz.log_tabular("Episode_return_std", eval_returns.std())
                        self.logz.log_tabular("Normalized_reward_mean", normalized_epoch_returns.mean())
                        self.logz.log_tabular("Normalized_reward_std", normalized_epoch_returns.std())
                        self.logz.log_tabular("Max_return_rollout", max_reward)
                        self.logz.log_tabular("Max_normalized_return_rollout", max_normal_reward)
                        if num_timesteps < iter_g + iter_l:
                            self.logz.log_tabular("Max_pool_normalized_return_rollout", max_normal_reward)
                        else:
                            self.logz.log_tabular("Max_pool_normalized_return_rollout", np.max(normal_returns))
                        self.logz.log_tabular("Min_pool_normalized_return_rollout", np.min(normal_returns))
                        self.logz.log_tabular("Std_pool_normalized_return", normal_returns.std())
                        if num_timesteps < iter_g + iter_l:
                            self.logz.log_tabular("Max_pool_return_rollout", max_reward)
                        else:
                            self.logz.log_tabular("Max_pool_return_rollout", np.max(returns))
                        self.logz.log_tabular("Min_pool_return_rollout", np.min(returns))
                        self.logz.log_tabular("Std_pool_return", returns.std())
                        self.logz.log_tabular("Alpha", sac_policy._alpha.detach().cpu().numpy())
                        self.logz.log_tabular("Loss_r", -r_loss)
                        self.logz.log_tabular("Loss_u", -u_loss)
                        self.logz.dump_tabular()

                        point_r, point_u, steps = self._evaluate_fake(policy=sac_policy)

                        if num_p >= self._num_pareto:
                            policy_array = np.concatenate(
                                (policy_array, np.expand_dims(copy.deepcopy(sac_policy).cpu(), axis=0)), axis=0)
                            # fitnesses = np.concatenate((fitnesses, np.array([[-r_loss, -u_loss]])))
                            fitnesses = np.concatenate((fitnesses, np.array([[point_r, point_u]])))
                            returns = np.append(returns, eval_returns.mean())
                            normal_returns = np.append(normal_returns, normalized_epoch_returns.mean())
                            assert policy_array.shape[0] == self._num_pareto + 1 and \
                                   len(returns) == self._num_pareto + 1

                            # refine pareto front via non-dominated sorting
                            fronts = fast_calculate_pareto_fronts(fitnesses)

                            # selected fixed number of pareto fronts
                            flat_fronts = np.array([item for sublist in fronts for item in sublist])
                            policy_array = policy_array[flat_fronts[:self._num_pareto]]
                            fitnesses = fitnesses[flat_fronts[:self._num_pareto], :]
                            returns = returns[flat_fronts[:self._num_pareto]]
                            normal_returns = normal_returns[flat_fronts[:self._num_pareto]]

                        else:
                            policy_pool.append(copy.deepcopy(sac_policy).cpu())
                            fitnesses[num_p] = np.array([[point_r, point_u]])
                            # fitnesses[num_p] = np.array([[-r_loss, -u_loss]])
                            policy_array = np.array(policy_pool)
                            returns[num_p] = np.mean(eval_returns)
                            normal_returns[num_p] = np.mean(normalized_epoch_returns)
                        num_p += 1

                    num_timesteps += 1
                    ref_timesteps += 1

                point_r, point_u, steps = self._evaluate_fake(policy=sac_policy)
                print("Fake env_reward: ", point_r, "Fake env uncertainty: ", point_u)
            self.cur_policy = None

            alpha = sac_policy.get_alpha()
            print("alpha: ", alpha)
            print(fitnesses, "fitness")

        print(returns, "returns")

        # save fronts
        file_name = 'front.txt'
        file = open(os.path.join(self._log_dir, file_name), "w")
        for i in range(len(fitnesses)):
            file.write(str(fitnesses[i][0]) + " " + str(fitnesses[i][1]) + "\n")

        # save policy
        for i in range(policy_array.shape[0]):
            torch.save(policy_array[i].state_dict(), os.path.join(self._log_dir, "offline_sac_{}.pth".format(i)))

        print("total time: {:.3f}s".format(time.time() - start_time))

    #  evalaute the policy in real environment
    def _evaluate_true(self, policy):
        episode_r = 0.
        ob = self.env.reset()
        for _ in range(self._eval_episodes):
            action = policy.select_action(ob, evaluate=True)
            ob, reward, terminal, _ = self.env.step(action)
            episode_r += reward
            if terminal:
                break
        return episode_r

    # evaluate the policy in environment model
    def _evaluate_fake(self, policy):
        episode_r = 0.
        episode_u = 0.
        steps = 0
        for _ in range(self._repeat_num):
            idx = self.rng.integers(0, self.init_obs.shape[0])
            ob = self.init_obs[idx]
            for _ in range(self._eval_episodes):
                action = policy.select_action(ob, evaluate=True)
                ob, reward, terminal, infos = self.fake_env.step(ob, action)

                # clipping
                ob = np.clip(ob, -100, 100)
                reward = self.r_filter(infos['unpenalized_rewards'][0])
                uncertainty = self.u_filter(infos['penalty'][0][0])

                episode_r += reward
                episode_u += np.exp(-uncertainty / 1.5)
                steps += 1
                if terminal:
                    break
        episode_r /= self._repeat_num
        episode_u /= self._repeat_num
        steps /= self._repeat_num

        return episode_r, episode_u, steps
