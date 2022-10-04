import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC, PPO, TD3
from horizon import SimEnv

device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

env = SimEnv(time_step = 0.5)
model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.1, seed=1)
model = model.learn(total_timesteps=250000, eval_freq=1000)

# env = SimEnvSim(time_step = 0.5)
# model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.15, seed=64)
# model = model.learn(total_timesteps=250000, eval_freq=1000)

# env = SimEnv45(time_step = 0.5)
# model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.15, seed=64)
# model = model.learn(total_timesteps=250000, eval_freq=1000)

# Evaulation
obs = env.reset(eval=True, prob=0.0)

for _ in range(50):
    action, _states = model.predict(obs, deterministic=True)
    print('======')
    print('action: ', action)
    obs, rewards, dones, info = env.step(action)
    print('obs: ', obs)
    print('reward: ', rewards)
    env.render()