import argparse
import datetime
import os
import random
import importlib
import glob

# Add CUDA 11 pip-installed libraries to LD_LIBRARY_PATH before importing TF
_nvidia_lib_dirs = glob.glob(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '.venv', 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))
if _nvidia_lib_dirs:
    _extra = os.pathsep.join(_nvidia_lib_dirs)
    os.environ['LD_LIBRARY_PATH'] = _extra + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import gym
import d4rl
import d4rl.gym_mujoco

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector
import tensorflow.compat.v1 as tf

from models.tf_dynamics_models.constructor import construct_model
from trainer_offline import Traineroffline
from utils.offlinebuffer import ReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="HiPoP")
    parser.add_argument("--task", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--psi', type=float, default=-0.999)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--policy-learning-rate', type=float, default=3e-4)
    parser.add_argument('--critic-learning-rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)  # 0.005
    parser.add_argument('--auto-alpha', default=True)  # True
    parser.add_argument('--target-entropy', type=int, default=None)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--log-alpha-max', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--optim', type=str, default="Adam")

    parser.add_argument('--num_pareto', type=int, default=100) 
    parser.add_argument('--num_refs', type=int, default=5) 
    parser.add_argument('--iter_g', '-n', type=int, default=180000) 
    parser.add_argument('--iter_l', type=int, default=10000)

    parser.add_argument('--start-training', type=int, default=10000)
    parser.add_argument('--reset-interval', type=int, default=200000)
    parser.add_argument('--update-per-step', type=int, default=2)

    # dynamics model's arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.2)
    parser.add_argument("--rollout-length", type=int, default=5)  # 5 for hopper, 3 for halfcheetah, 1 for walker2d
    parser.add_argument("--rollout-batch-size", type=int, default=50000)  # 50
    parser.add_argument("--buffer-size", type=int, default=1e6)  # 1e6
    parser.add_argument("--dynamics-model-dir", type=str,
                        default=None)
  
    parser.add_argument("--repeat_num", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--present_freq_g", type=int, default=1000) 
    parser.add_argument("--present_freq_l", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.obs_dim = np.prod(args.obs_shape)
    args.action_space = env.action_space
    args.action_dim = np.prod(env.action_space.shape)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # TF GPU config: use GPU if available (CUDA 11 pip libs are on LD_LIBRARY_PATH)
    config = tf.ConfigProto()
    if torch.cuda.is_available():
        gpu_id = args.device.replace("cuda:", "").replace("cuda", "0")
        config.gpu_options.visible_device_list = gpu_id
        config.gpu_options.allow_growth = True
        # Limit TF GPU memory to 40% so PyTorch also has room
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
    else:
        config = tf.ConfigProto(device_count={"GPU": 0})
        config.gpu_options.visible_device_list = ""

    # create dynamics model
    dynamics_model = construct_model(
        obs_dim=args.obs_dim,
        act_dim=args.action_dim,
        hidden_dim=200,
        num_networks=args.n_ensembles,
        num_elites=args.n_elites,
        session=tf.Session(config=config),
        model_type="mlp",
        separate_mean_var=True,
        name="BNN_0",  # set to none if want to renew one
        load_dir=args.dynamics_model_dir,
    )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)

    # create static fns
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns

    # log
    if args.logdir is not None:
        dir_path = args.logdir
        if not (os.path.exists(dir_path)):
            os.makedirs(dir_path)
        log_dir = dir_path +'/{}/{}_{}/offline_{}/update_step_{}/seed_{}'.format(args.task, args.algo_name, args.rollout_length, args.optim, args.update_per_step, args.seed)
        if not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
    else:
        log_dir = None

    args.log_dir = log_dir

    # create trainer
    trainer = Traineroffline(
        dynamics_model,
        eval_env=env,
        static_fns=static_fns,
        offline_buffer=offline_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        args=args
    )

    # pretrain dynamics model on the whole dataset
    trainer.train_dynamics()

    # begin train
    trainer.train_policy(args.num_refs, args.iter_g, args.iter_l)


if __name__ == "__main__":
    train()
