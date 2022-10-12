import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC, PPO, TD3
from horizon import HoriEnv
from vertical import VertEnv
from piecewise import SimEnv

device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# horienv = HoriEnv(time_step = 0.5)
# horimodel = PPO('MlpPolicy', horienv, verbose=1, device=device, ent_coef=0.1, seed=1)
# horimodel = horimodel.learn(total_timesteps=250000, eval_freq=1000)

# vertenv = VertEnv(time_step = 0.5)
# vertmodel = PPO('MlpPolicy', vertenv, verbose=1, device=device, ent_coef=0.1, seed=1)
# vertmodel = vertmodel.learn(total_timesteps=250000, eval_freq=1000)

env = SimEnv(time_step = 0.5)
model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.1, seed=1)
model = model.learn(total_timesteps=500000, eval_freq=1000)

# env = SimEnvSim(time_step = 0.5)
# model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.15, seed=64)
# model = model.learn(total_timesteps=250000, eval_freq=1000)

# env = SimEnv45(time_step = 0.5)
# model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.15, seed=64)
# model = model.learn(total_timesteps=250000, eval_freq=1000)

# Evaulation
obs = env.reset(eval=True)
intersection = env.intersection
next_manifold = False
for _ in range(50):
        # if np.linalg.norm(obs - intersection) <= 0.5 and not next_manifold:
        #         next_manifold = True
        # if not next_manifold:
        #         action, _states = horimodel.predict(obs, deterministic=True)
        #         obs, rewards, dones, info = env.step(action)
        # else:
        #         action, _states = vertmodel.predict(obs, deterministic=True)
        #         obs, rewards, dones, info = env.step(action)

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print('======')
        print('action: ', action)
        print('obs: ', obs)
        print('reward: ', rewards)
        env.render()