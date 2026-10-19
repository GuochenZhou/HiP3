import copy
import torch

from algorithms.sac import SAC
from utils.util import extract_sub_dict


class SAC_online(SAC):
    def __init__(self, critic, policy, critic_optimizer_ctor, policy_optimizer_ctor, alpha=0.2, discount=0.99, target_update_rate=0.005,
                 device="cpu", ckpt_path=None, copy_to_target=True):

        super().__init__(critic, policy, critic_optimizer_ctor, policy_optimizer_ctor, alpha=alpha, discount=discount,
                         target_update_rate=target_update_rate, device=device)

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:
            map_location = None
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic_r", checkpoint)

            self.policy.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)
