import csv
from datetime import datetime
import json
import os
import pickle
from pathlib import Path
import random
import string
import sys

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Iterable, Optional


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RunningMeanStd:
    """Welford's online algorithm for running mean/std."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if hasattr(x, '__len__') else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def set_default_device():
    """Set the default device.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def to_torch_device(x_np):
    return torch.FloatTensor(x_np)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'


def get_mode(dist):
    """Get the (transformed) mode of the distribution.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1134
    """
    if isinstance(dist, td.categorical.Categorical):
        mode = torch.argmax(dist.logits, -1)
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.Independent):
        mode = get_mode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_mode(dist.base_dist)
        mode = base_mode
        for transform in dist.transforms:
            mode = transform(mode)
    return mode


def epsilon_greedy_sample(dist, eps=0.1):
    """Generate greedy sample that maximizes the probability.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1106
    """

    def greedy_fn(dist):
        greedy_action = get_mode(dist)
        if eps == 0.0:
            return greedy_action
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return dist.sample()
    else:
        return greedy_fn(dist)


def __min_norm_2(v1_v1, v1_v2, v2_v2):
    """Helper function for the Frank Wolve Solver, compute the min alpha of
    the norm squared between two gradients
    .. math::
      \alpha = \frac{(\overline{\theta} - \theta)^{T} \overline{\theta}}
              {\|\theta - \overline{\theta}\|_{2}^{2}}
      where v1_v1 = \theta^{T}\theta
        v1_v2 = \theta^{T}\overline{\theta}
        v2_v2 = \overline{\theta}^{T}\overline{\theta}
    source: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
    (page 5, alg 1)
    """
    if v1_v1 <= v1_v2:
        return 1.0
    if v2_v2 <= v1_v2:
        return 0.0
    # calculate and return alpha
    return (v2_v2 - v1_v2) / (v1_v1 + v2_v2 - 2 * v1_v2)


def compute_cosine_similarity(rollouts_r, rollouts_u, ref_vector):
    col_vector = torch.column_stack((rollouts_r, rollouts_u))
    cos_sim = torch.cosine_similarity(col_vector[0], ref_vector, dim=0)
    return cos_sim


def compute_kl_similarity(rollouts_r, rollouts_u, ref_vector):
    col_vector = torch.column_stack((rollouts_r, rollouts_u))
    kl_sim = F.kl_div(col_vector.softmax(dim=-1).log(), ref_vector.softmax(dim=-1), reduction='sum')
    return kl_sim


def FrankWolfeSolver(gradients, max_iter=200, min_change=1e-3):
    """ FrankWolfeSolver class. Inherits from the COPSolver
    class.
    FrankWolfeSolver is used to calculate the numerical solutions
    for the QCOP for 2 or more gradients
    Attributes:
    max_iter: max number of iterations for the algorithm
    min_change: minimum change stopping criterion. The algorithms
    stop when the difference between iterations is lower than
    min_change
    """

    if max_iter < 0:
        raise ValueError('Argument, max_iter must be positive')
    if min_change < 0:
        raise ValueError('Arguement: min_change must be positive')

    if gradients is None:
        raise TypeError('Argument: gradients must be set.')
    for gradient in gradients:
        if len(gradient) != len(gradients[0]):
            raise ValueError('Argument: gradients must have the same length')

    # number of objectives
    n = len(gradients)

    # if there is only 1 gradient, alpha is equal to 1
    if n == 1:
        return np.ones(1)

    # initialize alphas
    alphas_list = np.ones(n) / n

    # precompute gradient products
    M = gradients @ gradients.T

    for i in range(max_iter):
        # find objective whose gradient gives the smallest product
        # (min row in M)
        min_objective = np.argmin(alphas_list @ M)

        # find min gamma
        # v1 = alphas_list @ gradients   <- combined gradient
        # v2 = gradients[min_objective]  <- min gradient
        v1_v1 = alphas_list @ M @ alphas_list
        v1_v2 = alphas_list @ M[min_objective]
        v2_v2 = M[min_objective, min_objective]
        min_gamma = __min_norm_2(v1_v1, v1_v2, v2_v2)

        # update alpha
        new_alphas_list = min_gamma * alphas_list
        new_alphas_list[min_objective] += 1 - min_gamma

        # if update is smaller than min_change stop
        if np.sum(np.abs(new_alphas_list - alphas_list)) \
            < min_change:
            return new_alphas_list
        else:
            alphas_list = new_alphas_list

    return alphas_list


def calculate_domination_matrix(fitnesses):
    pop_size = fitnesses.shape[0]
    num_objectives = fitnesses.shape[1]

    # print(pop_size, num_objectives)
    fitness_grid_x = np.zeros([pop_size, pop_size, num_objectives])
    fitness_grid_y = np.zeros([pop_size, pop_size, num_objectives])

    for i in range(pop_size):
        fitness_grid_x[i, :, :] = fitnesses[i]
        fitness_grid_y[:, i, :] = fitnesses[i]

    larger_or_equal = fitness_grid_x >= fitness_grid_y
    larger = fitness_grid_x > fitness_grid_y

    return np.logical_and(np.all(larger_or_equal, axis=2), np.any(larger, axis=2))


def fast_calculate_pareto_fronts(fitnesses):
    """
    https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
    """
    # Calculate dominated set for each individual
    domination_sets = []
    domination_counts = []

    domination_matrix = calculate_domination_matrix(fitnesses)
    pop_size = fitnesses.shape[0]

    for i in range(pop_size):
        current_dimination_set = set()
        domination_counts.append(0)
        for j in range(pop_size):
            if domination_matrix[i, j]:
                current_dimination_set.add(j)
            elif domination_matrix[j, i]:
                domination_counts[-1] += 1

        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            # print("Done")
            break
        # print("Front: ",current_front)
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[
                individual] = -1  # this individual is already accounted for, make it -1 so ==0 will not find it anymore
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1

    return fronts


def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def vector_to_gradients(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def extract_sub_dict(prefix, dict):

    def _remove_prefix(s, prefix):
        if s.startswith(prefix):
            return s[len(prefix):]
        else:
            return s

    sub_dict = {
            _remove_prefix(k, prefix + '.'): v
            for k, v in dict.items() if k.startswith(prefix)
        }

    return sub_dict
