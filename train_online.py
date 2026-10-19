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
import tensorflow.compat.v1 as tf


from models.tf_dynamics_models.constructor import construct_model
from trainer_online import Traineronline
from utils.onlinebuffer import ReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="HiP3")
    parser.add_argument("--task", type=str, default="Hopper-v3")
    parser.add_argument("--offline_task", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--alter_task", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--policy-learning-rate', type=float, default=3e-4)
    parser.add_argument('--critic-learning-rate', type=float, default=3e-4)
    parser.add_argument('--target-update-rate', type=float, default=0.005)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=None)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)  # 3e-4 for Mujoco, 3e-5 for robotics
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hidden-num', type=int, default=2)
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--select-policy', type=int, default=None)

    parser.add_argument('--num-pareto', type=int, default=100)
    parser.add_argument('--num-pool-online', type=int, default=5) 
    parser.add_argument('--real-ratio', type=float, default=0.2)  # 0.2 for mujoco
    parser.add_argument('--bandit-type', type=str, default="linucb")
    parser.add_argument('--linucb-alpha', type=float, default=1.0)

    # FQE evaluation arguments
    parser.add_argument('--eval-method', type=str, default='online', choices=['online', 'fqe'],
                        help='Policy evaluation method for initial selection: online rollouts or FQE')
    parser.add_argument('--fqe-steps', type=int, default=5000,
                        help='Number of gradient steps for FQE fitting per policy')
    parser.add_argument('--fqe-batch-size', type=int, default=256,
                        help='Batch size for FQE training')
    parser.add_argument('--fqe-lr', type=float, default=3e-4,
                        help='Learning rate for FQE Q-network')
    parser.add_argument('--normalize-reward', action='store_true', default=False,
                        help='Normalize rewards with running mean/std during online training')
    
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--steps-per-epochs', type=int, default=1000)
    parser.add_argument('--update-per-step', type=int, default=10)

    # dynamics model's arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.0)
    parser.add_argument("--model_hidden_dim", type=int, default=200)  # 200 for mujoco, 400 for robotics
    parser.add_argument("--model_learning_rate", type=float, default=1e-3)  # 1e-3 for mujoco 5e-4 for robotics
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--offline-dir", type=str,
                        default=None)

    parser.add_argument("--repeat_num", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--train_model_freq", type=int, default=500) 
    parser.add_argument('--model_retain_epochs', type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()
    
def train(args=get_args()):
    # create env
    if args.alter_task is False:
        env = gym.make(args.task)
    else:
        if "halfcheetah" in args.offline_task:
            env = gym.make(args.task, forward_reward_weight=0.85, ctrl_cost_weight=0.2, reset_noise_scale=0.2)
            env.model.opt.gravity[2] = -11.5
            env.sim.model.geom_friction[:] *= 1.2
        elif "walker2d" in args.offline_task:
            env = gym.make(args.task, forward_reward_weight=0.85, ctrl_cost_weight=0.2, reset_noise_scale=0.05)
            env.model.opt.gravity[2] = -11.5
            env.sim.model.geom_friction[:] *= 1.2
        elif " hopper" in args.offline_task:
            env = gym.make(args.task, forward_reward_weight=0.85, ctrl_cost_weight=0.2, reset_noise_scale=0.1)
            env.model.opt.gravity[2] = -11.5
            env.sim.model.geom_friction[:] *= 1.2
        else:
            env = gym.make(args.task)
            env.sim.model.geom_friction[:] *= 1.2
            env.sim.model.opt.gravity[:] = [0, 0, -11.5]
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

    # create static fns
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task.lower()}"
    static_fns = importlib.import_module(import_path).StaticFns

    # log
    if args.logdir is not None:
        dir_path = args.logdir
        if not (os.path.exists(dir_path)):
            os.makedirs(dir_path)
        if args.select_policy is None:
            if args.alter_task is False:
                log_dir = dir_path +'/{}/{}_{}/online_{}_{}/{}_{}/seed_{}'\
                    .format(args.offline_task, args.algo_name, args.rollout_length, args.optim, args.hidden_num,
                            args.bandit_type, args.num_pool_online, args.seed)
            else:
                log_dir = dir_path +'/{}/{}_{}/online_alter_{}_{}/{}_{}//seed_{}'\
                    .format(args.offline_task, args.algo_name, args.rollout_length, args.optim, args.hidden_num,
                            args.bandit_type, args.num_pool_online, args.seed)
        else:
            if args.alter_task is False:
                log_dir = dir_path +'/{}/{}_{}/online/policy_{}/seed_{}'.format(args.offline_task, args.algo_name,
                                                                         args.rollout_length, args.select_policy,
                                                                                args.seed)
            else:
                log_dir = dir_path + '/{}/{}_{}/online_alter/policy_{}/seed_{}'.format(args.offline_task, args.algo_name,
                                                                         args.rollout_length, args.select_policy,
                                                                                       args.seed)
        if not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
        if args.offline_dir is not None:
            args.offline_dir = args.offline_dir + '/seed_{}'.format(args.seed)
            dynamics_model_dir = args.offline_dir + '/dynamics_model'
        else:
            dynamics_model_dir = None
    else:
        log_dir = None

    args.dynamics_model_dir = dynamics_model_dir
    args.log_dir = log_dir

    # TF GPU config: use GPU if available (CUDA 11 pip libs are on LD_LIBRARY_PATH)
    config = tf.ConfigProto()
    if torch.cuda.is_available():
        gpu_id = args.device.replace("cuda:", "").replace("cuda", "0")
        config.gpu_options.visible_device_list = gpu_id
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
    else:
        config = tf.ConfigProto(device_count={"GPU": 0})
        config.gpu_options.visible_device_list = ""

    # create dynamics model
    dynamics_model = construct_model(
        obs_dim=args.obs_dim,
        act_dim=args.action_dim,
        hidden_dim=args.model_hidden_dim,
        num_networks=args.n_ensembles,
        num_elites=args.n_elites,
        session=tf.Session(config=config),
        model_type="mlp",
        separate_mean_var=True,
        name="BNN_0_0",  # set to none if want to renew one
        load_dir=args.dynamics_model_dir,
        learning_rate=args.model_learning_rate,
    )

    # create trainer
    trainer = Traineronline(
        dynamics_model,
        eval_env=env,
        static_fns=static_fns,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        args=args
    )

    # Get the offline model
    if args.dynamics_model_dir is not None:
        trainer.load_dynamics_model(args.dynamics_model_dir)

    # begin train
    trainer.train_policy()


if __name__ == "__main__":
    train()
