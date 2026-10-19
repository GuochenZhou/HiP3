import numpy as np


class ReplayBuffer:
    def __init__(
            self,
            buffer_size,
            obs_shape,
            obs_dtype,
            action_dim,
            action_dtype,
    ):
        self.max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self.ptr = 0
        self.size = 0

        self.init_obs = None
        self.observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, action, next_obs, reward, terminal):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset):
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        init, init_obs = True, []
        for i in range(len(observations)):
            if init is True:
                init_obs.append(observations[i])
                init = False
            if terminals[i]:
                init = True
        self.init_obs = np.array(init_obs)
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self.ptr = len(observations)
        self.ptr = self.ptr % self.max_size
        self.size = len(observations)


    def add_batch(self, obs, actions, next_obs, rewards, terminals):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = np.array(obs[:first_add_size]).copy()
            self.next_observations[begin:end] = np.array(next_obs[:first_add_size]).copy()
            self.actions[begin:end] = np.array(actions[:first_add_size]).copy()
            self.rewards[begin:end] = np.array(rewards[:first_add_size]).copy()
            self.terminals[begin:end] = np.array(terminals[:first_add_size]).copy()

            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = np.array(obs[first_add_size:]).copy()
            self.next_observations[begin:end] = np.array(next_obs[first_add_size:]).copy()
            self.actions[begin:end] = np.array(actions[first_add_size:]).copy()
            self.rewards[begin:end] = np.array(rewards[first_add_size:]).copy()
            self.terminals[begin:end] = np.array(terminals[first_add_size:]).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = np.array(obs).copy()
            self.next_observations[begin:end] = np.array(next_obs).copy()
            self.actions[begin:end] = np.array(actions).copy()
            self.rewards[begin:end] = np.array(rewards).copy()
            self.terminals[begin:end] = np.array(terminals).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):

        batch_indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy()
        }

    def sample_batch(self, batch_size):

        batch_indices = np.random.randint(0, self.size, size=batch_size)

        return self.observations[batch_indices].copy(), self.actions[batch_indices].copy(), \
               self.next_observations[batch_indices].copy(), self.rewards[batch_indices].copy(),\
               self.terminals[batch_indices].copy()

    def sample_all(self):
        return self.observations[:self.size].copy(), self.actions[:self.size].copy(), \
               self.next_observations[:self.size].copy(), self.rewards[:self.size].copy(),\
               self.terminals[:self.size].copy()

    def get_init_obs(self):
        return self.init_obs.copy()
