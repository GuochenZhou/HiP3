import os
import time
import copy
import numpy as np
import torch
import d4rl
import logz

from models.tf_dynamics_models.fake_env import FakeEnv
from models.policy import GaussianPolicy, QNetwork
from algorithms.sac_online import SAC_online
from algorithms.fqe import FQE
from utils.onlinebuffer import ReplayBuffer
from sample_env import EnvSampler
from models.linucb import LinUCB
from utils.util import RunningMeanStd


class Traineronline(object):
    def __init__(
            self,
            dynamics_model,
            eval_env,
            static_fns,
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
        self.model_buffer = None
        self.env_buffer = None
        self.env_sampler = None
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._rollout_freq = int(rollout_batch_size * rollout_length)
        self._batch_size = batch_size
        self._eval_episodes = args.eval_episodes
        self._repeat_num = args.repeat_num
        self._num_pareto = args.num_pareto
        self._num_pool_online = args.num_pool_online
        self._log_dir = log_dir
        self.args = args
        self.logz = logz
        self.logz.configure_output_dir(log_dir)
        self.bandit = None
        self.online_policy_pool = []
        self.fixed_policy = None
        self.offline_dataset_buffer = None  # for FQE evaluation
        self.reward_normalizer = RunningMeanStd() if getattr(args, 'normalize_reward', False) else None
        self.r_filter = lambda x: np.clip(
            np.where(np.logical_or(np.isnan(x), np.isinf(x)), 0, x),
            0,
            20.0
        )

    def _sample_initial_transitions(self):
        return self.env_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self):
        init_transitions = self._sample_initial_transitions()
        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            if self.fixed_policy is None:
                actions = []
                for i in range(len(observations)):
                    select_policy_num, _ = self.bandit.play(observations[i])
                    actions.append(self.online_policy_pool[select_policy_num].select_action(observations[i]))
                actions = np.array(actions)
            else:
                actions = self.fixed_policy.select_action(observations)
            next_observations, rewards, terminals, infos = self.fake_env.step(observations, actions)

            # clipping
            next_observations = np.clip(next_observations, -100, 100)
            rewards = self.r_filter(rewards)
            self.model_buffer.add_batch(observations, actions, next_observations,
                                        rewards, terminals)

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

    def save_dynamics_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dynamics_model.save(save_path, timestep=0)

    def load_dynamics_model(self, loadpath):
        if not os.path.exists(loadpath):
            raise Exception("Model direct doesn't exist")
        self.dynamics_model.load_params()

    def set_rollout_length(self, epoch_step):
        rollout_length = (min(max(self.args.rollout_min_length + (epoch_step - self.args.rollout_min_epoch)
                                  / (self.args.rollout_max_epoch - self.args.rollout_min_epoch) * (
                                              self.args.rollout_max_length - self.args.rollout_min_length),
                                  self.args.rollout_min_length), self.args.rollout_max_length))
        return int(rollout_length)

    def resize_model_pool(self):
        rollouts_per_epoch = self.args.rollout_batch_size * self.args.steps_per_epochs / self.args.train_model_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self.args.model_retain_epochs * model_steps_per_epoch

        state, action, next_state, reward, done = self.model_buffer.sample_all()
        new_model_buffer = ReplayBuffer(
            buffer_size=int(new_pool_size),
            obs_shape=self.args.obs_shape,
            obs_dtype=np.float32,
            action_dim=self.args.action_dim,
            action_dtype=np.float32
        )
        new_model_buffer.add_batch(state, action, next_state, reward, done)
        self.model_buffer = new_model_buffer

    def train_predict_model(self, max_epochs=None):
        # Get all samples from environment
        state, action, next_state, reward, done = self.env_buffer.sample_all()
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

        self.fake_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs=max_epochs)

    def train_policy(self):
        start_time = time.time()
        offline_policy_pool = []
        offline_returns = np.zeros(self._num_pareto)
        offline_normal_returns = np.zeros(self._num_pareto)
        returns = np.zeros(self._num_pool_online)
        normal_returns = np.zeros(self._num_pool_online)
        max_reward, max_normal_reward = 0.0, 0.0
        num_epochs = 0
        iterations = 0
        select_policy_num = 0

        # initialize the model buffer
        rollouts_per_epoch = self.args.rollout_batch_size * self.args.steps_per_epochs / self.args.train_model_freq
        model_steps_per_epoch = int(1 * rollouts_per_epoch)
        new_pool_size = self.args.model_retain_epochs * model_steps_per_epoch
        model_buffer = ReplayBuffer(
            buffer_size=int(new_pool_size),
            obs_shape=self.args.obs_shape,
            obs_dtype=np.float32,
            action_dim=self.args.action_dim,
            action_dtype=np.float32
        )
        self.model_buffer = model_buffer

        # initialize the env buffer
        env_buffer = ReplayBuffer(
            buffer_size=int(1e6),
            obs_shape=self.args.obs_shape,
            obs_dtype=np.float32,
            action_dim=self.args.action_dim,
            action_dtype=np.float32
        )
        self.env_buffer = env_buffer

        # initialize environment sampler
        self.env_sampler = EnvSampler(self.env, max_path_length=self._eval_episodes)

        # init alpha
        if self.args.auto_alpha:
            target_entropy = self.args.target_entropy if self.args.target_entropy \
                                    else -float(int(np.prod(self.env.action_space.shape) / 2))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = self.args.alpha

        if self.args.select_policy is None:
            # reload all offline policies
            for i in range(self._num_pareto):
                # generate critic
                critic = QNetwork(self.args.obs_dim, self.args.action_dim, hidden_dim=self.args.hidden_dim, hidden_num=self.args.hidden_num)

                # generate policy
                policy = GaussianPolicy(self.args.obs_dim, self.args.action_dim,
                                        self.args.hidden_dim, self.args.hidden_num, self.args.action_space)
                # policy path
                if self.args.offline_dir is not None:
                    policy_path = os.path.join(self.args.offline_dir, "offline_sac_{}.pth".format(i))
                else:
                    policy_path = None

                sac_policy = SAC_online(
                    critic=critic,
                    policy=policy,
                    critic_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.critic_learning_rate),
                    policy_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.policy_learning_rate),
                    alpha=copy.deepcopy(alpha),
                    discount=self.args.discount,
                    target_update_rate=self.args.target_update_rate,
                    device=self.args.device,
                    ckpt_path=policy_path
                )
                offline_policy_pool.append(sac_policy)
            print("Successfully load offline policies")
            # initial evaluation — use FQE or online rollouts
            eval_method = getattr(self.args, 'eval_method', 'online')
            if eval_method == 'fqe':
                print("Evaluating offline policies with FQE...")
                fqe_steps = getattr(self.args, 'fqe_steps', 5000)
                fqe_batch_size = getattr(self.args, 'fqe_batch_size', 256)
                fqe_values = self._evaluate_fqe(
                    offline_policy_pool,
                    fqe_steps=fqe_steps,
                    fqe_batch_size=fqe_batch_size
                )
                for i in range(self.args.num_pareto):
                    offline_returns[i] = fqe_values[i]
                    offline_normal_returns[i] = fqe_values[i]  # FQE values used for ranking only
            else:
                for i in range(self.args.num_pareto):
                    eval_returns = np.array([self._evaluate_true(policy=offline_policy_pool[i])
                                             for _ in range(self._repeat_num)])
                    normalized_epoch_returns = d4rl.get_normalized_score(self.args.offline_task, eval_returns) * 100.0
                    offline_returns[i] = np.mean(eval_returns)
                    offline_normal_returns[i] = np.mean(normalized_epoch_returns)

            # sort policies
            sort_list = np.argsort(-offline_returns)
            for i in range(self._num_pool_online):
                self.online_policy_pool.append(offline_policy_pool[sort_list[i]])
                returns[i] = offline_returns[sort_list[i]]
                normal_returns[i] = offline_normal_returns[sort_list[i]]

            print("returns", returns)
            # initialize the contextual bandit
            self.bandit = LinUCB(narms=self._num_pool_online, ndims=self.args.obs_dim, alpha=self.args.linucb_alpha, device=self.args.device)
        else:
            # generate critic
            critic = QNetwork(self.args.obs_dim, self.args.action_dim, hidden_dim=self.args.hidden_dim, hidden_num=self.args.hidden_num)

            # generate policy
            policy = GaussianPolicy(self.args.obs_dim, self.args.action_dim,
                                    self.args.hidden_dim, self.args.hidden_num, self.args.action_space)
            # policy path
            if self.args.offline_dir is not None:
                policy_path = os.path.join(self.args.offline_dir, "offline_sac_{}.pth".format(self.args.select_policy))
            else:
                policy_path = None

            sac_policy = SAC_online(
                critic=critic,
                policy=policy,
                critic_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.critic_learning_rate),
                policy_optimizer_ctor=lambda params: torch.optim.Adam(params, lr=self.args.policy_learning_rate),
                alpha=copy.deepcopy(alpha),
                discount=self.args.discount,
                target_update_rate=self.args.target_update_rate,
                device=self.args.device,
                ckpt_path=policy_path
            )
            self.fixed_policy = sac_policy

        # initialize the dynamic model
        # self.train_predict_model(max_epochs=1)

        # train the policy
        for _ in range(self.args.epochs):
            for cur_step in range(self.args.steps_per_epochs):
                # warmup stage
                if iterations < self.args.warmup_steps:
                    if self.args.select_policy is None:
                        select_policy_num, prob = self.bandit.play(self.env_sampler.sample_state())
                        cur_state, action, next_state, reward, done, info = self.env_sampler.sample(
                            self.online_policy_pool[select_policy_num])
                        self.bandit.update(select_policy_num, reward, cur_state)
                    else:
                        cur_state, action, next_state, reward, done, info = self.env_sampler.sample(
                            self.fixed_policy)
                    self.env_buffer.add(cur_state, action, next_state, reward, done)
                    loss = 0.0

                else:
                    # generate new data from model
                    if cur_step % self.args.train_model_freq == 0 and self.args.real_ratio < 1.0:
                        # train predict model
                        self.train_predict_model()
                        self.rollout_transitions()

                    # sample from environment
                    if self.args.select_policy is None:
                        select_policy_num, prob = self.bandit.play(self.env_sampler.sample_state())
                        cur_state, action, next_state, reward, done, info = self.env_sampler.sample(
                            self.online_policy_pool[select_policy_num])
                        self.bandit.update(select_policy_num, reward, cur_state)
                    else:
                        cur_state, action, next_state, reward, done, info = self.env_sampler.sample(
                            self.fixed_policy)
                    self.env_buffer.add(cur_state, action, next_state, reward, done)

                    # train policy
                    for _ in range(self.args.update_per_step):
                        real_sample_size = int(self._batch_size * self.args.real_ratio)
                        fake_sample_size = self._batch_size - real_sample_size
                        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
                        real_batch = self.env_buffer.sample(batch_size=real_sample_size)
                        data = {
                            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]],
                                                           axis=0),
                            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
                            "next_observations": np.concatenate(
                                [real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
                            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
                            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
                        }
                        if self.reward_normalizer is not None:
                            self.reward_normalizer.update(data["rewards"].flatten())
                            data["rewards"] = self.reward_normalizer.normalize(data["rewards"])
                        if self.args.select_policy is None:
                            loss = self.online_policy_pool[select_policy_num].update_parameter(data)
                        else:
                            loss = self.fixed_policy.update_parameter(data)
                iterations += 1

            # evaluate the result
            if self.args.select_policy is None:
                eval_returns = np.array([self._evaluate_true() for _ in range(self._repeat_num)])
            else:
                eval_returns = np.array([self._evaluate_true(policy=self.fixed_policy)
                                         for _ in range(self._repeat_num)])
            normalized_epoch_returns = d4rl.get_normalized_score(self.args.offline_task, eval_returns) * 100.0
            max_reward = np.max([max_reward, np.mean(eval_returns)])
            max_normal_reward = np.max([max_normal_reward, np.mean(normalized_epoch_returns)])
            self.logz.log_tabular("Iterations", iterations)
            self.logz.log_tabular("Time", time.time() - start_time)
            self.logz.log_tabular("Episode_return_mean", eval_returns.mean())
            self.logz.log_tabular("Episode_return_std", eval_returns.std())
            self.logz.log_tabular("Normalized_reward_mean", normalized_epoch_returns.mean())
            self.logz.log_tabular("Normalized_reward_std", normalized_epoch_returns.std())
            self.logz.log_tabular("Max_return_rollout", max_reward)
            self.logz.log_tabular("Max_normalized_return_rollout", max_normal_reward)
            self.logz.log_tabular("Loss", loss)
            self.logz.dump_tabular()
            num_epochs += 1
            if self.args.select_policy is None:
                if iterations > self.args.warmup_steps:
                    self.bandit.update_epoch()

        # save policy
        if self.args.select_policy is None:
            for i in range(self.args.num_pool_online):
                torch.save(self.online_policy_pool[i].state_dict(), os.path.join(self._log_dir, "online_sac_{}.pth".format(i)))
        else:
            torch.save(self.fixed_policy.state_dict(), os.path.join(self._log_dir, "online_sac.pth"))

    def _load_offline_dataset(self):
        """Load D4RL offline dataset into a buffer for FQE training."""
        import gym
        offline_env = gym.make(self.args.offline_task)
        dataset = d4rl.qlearning_dataset(offline_env)
        buffer = ReplayBuffer(
            buffer_size=len(dataset["observations"]),
            obs_shape=self.args.obs_shape,
            obs_dtype=np.float32,
            action_dim=self.args.action_dim,
            action_dtype=np.float32
        )
        buffer.load_dataset(dataset)
        self.offline_dataset_buffer = buffer
        print(f"Loaded offline dataset with {buffer.size} transitions for FQE")

    def _evaluate_fqe(self, policy_pool, fqe_steps=5000, fqe_batch_size=256):
        """Evaluate a pool of policies using FQE on the offline dataset.

        Args:
            policy_pool: list of SAC_online policies to evaluate
            fqe_steps: number of FQE training steps per policy
            fqe_batch_size: batch size for FQE training

        Returns:
            estimated_values: np.array of estimated values for each policy
        """
        if self.offline_dataset_buffer is None:
            self._load_offline_dataset()

        fqe = FQE(
            obs_dim=self.args.obs_dim,
            action_dim=self.args.action_dim,
            hidden_dim=self.args.hidden_dim,
            hidden_num=self.args.hidden_num,
            discount=self.args.discount,
            lr=getattr(self.args, 'fqe_lr', 3e-4),
            device=self.args.device
        )

        estimated_values = np.zeros(len(policy_pool))
        for i, sac_policy in enumerate(policy_pool):
            fqe.reset()
            value = fqe.fit(
                self.offline_dataset_buffer,
                sac_policy.policy,
                num_steps=fqe_steps,
                batch_size=fqe_batch_size
            )
            estimated_values[i] = value
            print(f"  FQE policy {i}: estimated value = {value:.2f}")

        return estimated_values

    def _evaluate_true(self, policy=None):
        episode_r = 0.
        ob = self.env.reset()
        for _ in range(self._eval_episodes):
            if policy is None:
                select_policy_num, _ = self.bandit.play(ob)
                action = self.online_policy_pool[select_policy_num].select_action(ob, evaluate=True)
            else:
                action = policy.select_action(ob, evaluate=True)
            ob, reward, terminal, _ = self.env.step(action)
            episode_r += reward
            if terminal:
                break
        return episode_r
