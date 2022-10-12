import torch
import numpy as np 

from mushroom_rl.utils.viewer import Viewer

import gym
from gym import Env
from gym.spaces import Discrete, Box

class SimEnv(Env):
  
    def __init__(self, time_step = 0.1, eps = 0.5):
        self.time_step = time_step

        self._viewer = Viewer(env_width = 22, env_height = 22, background = (0, 0, 0))
        
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(np.array([-10, -10, 0]), np.array([10, 10, 1]))
        # mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        # super().__init__(mdp_info)

        self.state = np.array([0.0, 0.0, 0])
        # self.state = np.array([10.0, 0.0, 1.0, 0.0])
        self.intersection = np.array([10.0, 0.0, 0])
        self.goal = np.array([10.0, 10.0, 1])
        self.eps = eps

        self.next_manifold = False

    def _horizontal_reward(self):
        if self.state[1] >= 0.5 or self.state[1] <= -0.5:
            reward = -100
        else:
            reward = np.exp(-np.linalg.norm(self.intersection - self.state))
        return reward

    def _vertical_reward(self):
        if self.state[0] >= 10.5 or self.state[0] <= 9.5:
            reward = -100
        else:
            reward = np.exp(-np.linalg.norm(self.goal - self.state))
        return reward

    def step(self, action):
        self.state[0] += action[0]
        self.state[1] += action[1]

        done = False
        if np.linalg.norm(self.goal - self.state) <= 0.5:
            done = True
        if self.state[0] >= 12 or self.state[0] <= -12 or self.state[1] >= 12 or self.state[1] <= 12:
            done = True
        
        if np.linalg.norm(self.intersection - self.state) <= 0.5 and not self.next_manifold:
            self.next_manifold = True
            self.state[2] = 1
        
        if not self.next_manifold:
            reward = self._horizontal_reward()
        else:
            reward = self._vertical_reward()

        info = {}
        return self.state, reward, done, info

    def render(self):
        start = np.array([10.0, 10.0])
        intersection = np.array([20.0, 10.0])
        goal = np.array([20.0, 20.0])
        agent = np.array([10.0, 10.0])

        hline_start = np.array([0.0, 10])
        hline_end = np.array([22, 10])
        vline_start = np.array([20, 0])
        vline_end = np.array([20, 22])
        self._viewer.line(hline_start, hline_end, color = (100, 100, 100), width = 5)
        self._viewer.line(vline_start, vline_end, color = (100, 100, 100), width = 5)
        self._viewer.circle(center = start, radius = 0.3, color = (255, 0, 0))
        self._viewer.circle(center = intersection, radius = 0.3, color = (0, 0, 255))
        self._viewer.circle(center = goal, radius = 0.3, color = (0, 255, 0))
        self._viewer.circle(center = agent + self.state[:2], radius = 0.3, color = (50, 100, 150))
        self._viewer.display(self.time_step)

    def reset(self, eval=False):
        self.next_manifold = False
        self.state = np.array([0.0, 0.0, 0])
        return self.state
