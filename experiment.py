# import os
# import argparse
# from copy import deepcopy
# from tqdm import trange
# import pandas as pd

# from mushroom_rl.algorithms.actor_critic import PPO, TRPO, DDPG, TD3, SAC
# from mushroom_rl.core import Core, Logger
# from mushroom_rl.policy import GaussianTorchPolicy, OrnsteinUhlenbeckPolicy, ClippedGaussianPolicy
# from mushroom_rl.utils.dataset import compute_J, parse_dataset
# from atacom.environments.circular_motion import CircleEnvAtacom, CircleEnvTerminated, CircleEnvErrorCorrection
# from network import *

# from SimEnv import SimEnv

# def build_agent_PPO(mdp_info, actor_lr, critic_lr, n_features, batch_size, eps_ppo, 
#                     lam, ent_coeff, **kwargs):
#     policy_params = dict(
#         std_0 = 0.5,
#         n_features = n_features,
#         use_cuda = torch.cuda.is_available()
#     )

#     policy = GaussianTorchPolicy(PPONetwork,
#                                 mdp_info.observation_space.shape,
#                                 mdp_info.action_space.shape,
#                                 **policy_params)

#     critic_params = dict(network = PPONetwork,
#                         optimizer = {'class': optim.Adam,
#                                     'params': {'lr': critic_lr}},
#                         loss = F.mse_loss,
#                         n_features = n_features,
#                         batch_size = batch_size,
#                         input_shape = mdp_info.observation_space.shape,
#                         output_shape = (1,))
    
#     ppo_params = dict(actor_optimizer={'class': optim.Adam,
#                                        'params': {'lr': actor_lr}},
#                       n_epochs_policy=4,
#                       batch_size=batch_size,
#                       eps_ppo=eps_ppo,
#                       lam=lam,
#                       ent_coeff=ent_coeff,
#                       critic_params=critic_params)    
    
#     build_params = dict(compute_entropy_with_states=False,
#                         compute_policy_entropy=True)

#     return PPO(mdp_info, policy, **ppo_params), build_params
    
# if __name__ == '__main__':
#     env = SimEnv()
#     agent, build_params = build_agent_PPO(mdp_info = env.info)

import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC, PPO, TD3
from SimEnv45 import SimEnv45

# env = SimEnv45(time_step = 0.5)
env = SimEnv45(time_step = 0.5)
device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.15)
print('model: ', model)
model = model.learn(total_timesteps=300000, eval_freq=1000)

# Evaulation
obs = env.reset(eval=True)

for _ in range(50):
    action, _states = model.predict(obs, deterministic=True)
    print('======')
    print('action: ', action)
    obs, rewards, dones, info = env.step(action)
    print('obs: ', obs)
    print('reward: ', rewards)
    env.render()