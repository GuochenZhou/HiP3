import numpy as np


class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        # Ensure that obs, act, and next_obs have the expected shape (batch_size, dim)
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        # In the default implementation, we don't have any state-based termination
        done = np.array([False] * obs.shape[0])
        done = done[:, None]  # Reshape to (batch_size, 1)
        return done
