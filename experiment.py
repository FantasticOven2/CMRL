import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC, PPO, TD3
from SimEnv45 import SimEnv45
from SimEnv import SimEnv
from SimEnvSim import SimEnvSim
from SimEnvLine import SimEnvLine

env = SimEnv(time_step = 0.5)
# env = SimEnvSim(time_step = 0.5)
# env = SimEnv45(time_step = 0.5)
# env = SimEnvLine(time_step = 0.5)
device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.1, seed=150)
print('model: ', model)
model = model.learn(total_timesteps=500000, eval_freq=1000)

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