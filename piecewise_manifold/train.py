import os
import random
from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from stable_baselines3 import SAC, PPO, TD3
from horizon import HoriEnv
from vertical import VertEnv
from piecewise import SimEnv

class MetaPolicy(nn.Module):
  def __init__(self):
    super(MetaPolicy, self).__init__()
    self.fc_map = nn.Sequential(
         nn.Linear(2, 128),
         nn.ReLU(),
         nn.Linear(128, 64),
         nn.ReLU(),
         nn.Linear(64, 2)
     )
    
  def forward(self, x):
      return self.fc_map(x)

device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

horienv = HoriEnv(time_step = 0.5)
horimodel = PPO('MlpPolicy', horienv, verbose=1, device=device, ent_coef=0.1, seed=1)
horimodel = horimodel.learn(total_timesteps=250000, eval_freq=1000)

vertenv = VertEnv(time_step = 0.5)
vertmodel = PPO('MlpPolicy', vertenv, verbose=1, device=device, ent_coef=0.1, seed=1)
vertmodel = vertmodel.learn(total_timesteps=250000, eval_freq=1000)

env = SimEnv(time_step = 0.5)
# model = PPO('MlpPolicy', env, verbose=1, device=device, ent_coef=0.1, seed=1)
# model = model.learn(total_timesteps=500000, eval_freq=1000)

# Evaulation
# obs = env.reset(eval=True)
# intersection = env.intersection
# next_manifold = False
# for _ in range(50):
#         if np.linalg.norm(obs - intersection) <= 0.5 and not next_manifold:
#                 next_manifold = True
#         if not next_manifold:
#                 action, _states = horimodel.predict(obs, deterministic=True)
#                 obs, rewards, dones, info = env.step(action)
#         else:
#                 action, _states = vertmodel.predict(obs, deterministic=True)
#                 obs, rewards, dones, info = env.step(action)

#         # action, _states = model.predict(obs, deterministic=True)
#         # obs, rewards, dones, info = env.step(action)
#         print('======')
#         print('action: ', action)
#         print('obs: ', obs)
#         print('reward: ', rewards)
#         env.render()

obs = env.reset(eval=True)
obs = torch.tensor(obs, dtype=torch.float).to(device)
intersection = torch.tensor(env.intersection)

next_manifold = False
meta_policy = MetaPolicy()
meta_policy = meta_policy.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(meta_policy.parameters(), lr=0.001)

for i in tqdm(range(15000)):


  optimizer.zero_grad()

  pred = meta_policy(obs)

  if torch.linalg.norm(obs.cpu() - intersection) <= 0.5 and not next_manifold:
    next_manifold = True
  if not next_manifold:
      action, _states = horimodel.predict(obs.cpu().numpy(), deterministic=True)
      obs, rewards, dones, info = env.step(action)
  else:
      action, _states = vertmodel.predict(obs.cpu().numpy(), deterministic=True)
      obs, rewards, dones, info = env.step(action)
  
  if dones:
    next_manifold = False
    obs = env.reset()

  action = torch.tensor(action).to(device)
  obs = torch.tensor(obs, dtype=torch.float).to(device)
  loss = criterion(pred, action)
  
  loss.backward()
  optimizer.step()

obs = env.reset(eval=True)
obs = torch.tensor(obs, dtype=torch.float).to(device)
for _ in range(20):
  action = meta_policy(obs)
  obs, rewards, dones, info = env.step(action)
  obs = torch.tensor(obs, dtype=torch.float).to(device)
  
  print('======')
  print('action: ', action)
  print('obs: ', obs)
  print('reward: ', rewards)
  print(dones)
  env.render()
