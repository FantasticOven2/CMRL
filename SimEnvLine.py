import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# from mushroom_rl.algorithms.actor_critic import PPO, SAC, TD3
# from mushroom_rl.core import Environment, MDPInfo
# from mushroom_rl.utils import spaces
from mushroom_rl.utils.viewer import Viewer

import gym
from gym import Env
from gym.spaces import Discrete, Box


class SimEnvLine(Env):

    def __init__(self, time_step = 0.1, eps = 0.5):
        self.time_step = time_step
        
        self._viewer = Viewer(env_width = 22, env_height = 22, background = (0, 0, 0))

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(np.array([-10, -10]), np.array([10, 10]))

        self.state = np.array([0.0, 0.0])
        self.intersection = np.array([10.0, 0.0])
        self.goal = np.array([20.0, 0.0])

        self.next_manifold = False

    def check_constraints(self):
        on_second_manifold = (self.state[1] != 0)
        on_first_manifold = (self.state[1] != 0)
        return on_first_manifold * on_second_manifold

    def _reward(self, state):
        done = False

        if self.check_constraints():
            reward = -100
            done = True

        elif np.array_equal(state, self.intersection):
            if self.next_manifold:
                print('Here')
                reward = -np.linalg.norm(self.goal - self.state)
            else:
                reward = 1
            
        elif np.array_equal(state, self.goal):
            reward = 100
            done = True
        
        else:
            if self.next_manifold:
                reward = -np.linalg.norm(self.goal - self.state)
            else:
                reward = -np.linalg.norm(self.goal - self.state)
        return reward, done

    def step(self, action):
        ### Projection ###
        if self.next_manifold:
            _action = np.array([action[0], 0.0])
        
        else:
            _action = np.array([action[0], 0.0])
        
        self.state[0] += _action[0]
        self.state[1] += _action[1]
        reward, done = self._reward(self.state)

        if self.state[0] == 10.0 and self.state[1] == 0.0:
            self.next_manifold = True
        
        info = {}
        return self.state, reward, done, info

    def render(self):
        start = np.array([0.0, 10.0])
        intersection = np.array([10.0, 10.0])
        goal = np.array([20.0, 10.0])
        agent = np.array([0.0, 10.0])

        hline_start = np.array([0.0, 10.0])
        hline_end = np.array([22, 10])
        # vline_start = np.array([10, 10])
        # vline_end = np.array([20, 20])
        self._viewer.line(hline_start, hline_end, color = (100, 100, 100), width = 5)
        # self._viewer.line(vline_start, vline_end, color = (100, 100, 100), width = 5)
        self._viewer.circle(center = start, radius = 0.3, color = (255, 0, 0))
        self._viewer.circle(center = intersection, radius = 0.3, color = (0, 0, 255))
        self._viewer.circle(center = goal, radius = 0.3, color = (0, 255, 0))
        self._viewer.circle(center = agent + self.state[:2], radius = 0.3, color = (50, 100, 150))
        self._viewer.display(self.time_step)

    def reset(self, eval=False, prob=0.4):
        # print('RESET')
        # if not eval:
        #     rand = np.random.uniform()
        #     if rand < prob:
        #         self.state = np.array([0.0, 10.0])
        #     else:
        #         self.state = np.array([10.0, 0.0])
        self.state = np.array([0.0, 0.0])
        self.next_manifold = False
        return self.state

if __name__ == '__main__':
    env = SimEnvLine()
    