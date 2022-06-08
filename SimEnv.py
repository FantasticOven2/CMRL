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

class SimEnv(Env):
  
    def __init__(self, time_step = 0.1, eps = 0.5):
        self.time_step = time_step

        self._viewer = Viewer(env_width = 22, env_height = 22, background = (0, 0, 0))
        
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(np.array([-10, -10, 0, 0]), np.array([10, 10, 1, 1]))
        # mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        # super().__init__(mdp_info)

        self.state = np.array([0.0, 0.0, 1.0, 0.0])
        # self.state = np.array([10.0, 0.0, 1.0, 0.0])
        self.intersection = np.array([10.0, 0.0, 1.0, 0.0])
        self.goal = np.array([10.0, 10.0, 0.0, 1.0])

        self.eps = eps
        # self.next_manifold = False


    def _sparseReward(self, state):
        done = False
        if np.array_equal(state, self.goal):
            reward = 100
            done = True
        elif np.array_equal(state, self.intersection) and self.state[3] == 0:
            reward = 10
        else:
            reward = 0
        return reward, done
        

    def check_constraints(self):
        return ((self.state[1] < -self.eps) + (self.state[1] > self.eps)) * (
            (self.state[0] < 10 - self.eps) + (self.state[0] > 10 + self.eps)) 

    def _reward(self, state):
        done = False

        ### Fall off the manifold ###
        if self.check_constraints():
            # Negative of l2 distance to current manifold
            # if self.state[2] == 1.0:
            #     reward = -np.abs(state_1 - 10)
            # else:
            #     reward = -np.abs(state_2)
            reward = -100
            done = True
        
        ### Reach the interseciton ###
        elif np.array_equal(state, self.intersection):
            if self.state[3] == 1.0:
                reward = -np.abs(self.state[1] - 10)
            else:
                reward = 10

        ### Reach the goal ###
        elif np.array_equal(state, self.goal):
            reward = 100
            done = True

        ### Direct the agent to intersection point / goal
        else: 
            if self.state[3] == 1.0:
                reward = -np.abs(self.state[1] - 10)
                print('second manfold: ', reward)
            else: 
                reward = - np.abs(self.state[0] - 10)
                print('first manifold: ', reward)
        return reward, done

    def step(self, action):
        ### Projection ###
        if self.state[3] == 1.0: 
            _action = np.array([0.0, action[1]])
            # _action = np.array([[0.0, 0.0], [0.0, 1.0]]) @ _action
        else:
            _action = np.array([action[0], 0.0])
            # _action = np.array([[1.0, 0.0], [0.0, 0.0]]) @ _action

        self.state[0] += _action[0]
        self.state[1] += _action[1]
        
        reward, done = self._reward(self.state)
        # reward, done = self._sparseReward(self.state)

        if self.state[0] == 10.0 and self.state[1] == 0.0:
            self.state[2] = 0.0
            self.state[3] = 1.0

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

    def reset(self, eval=False, prob=0.4):
        if not eval:
            rand = np.random.uniform()
            if rand < prob:
                print('1')
                self.state = np.array([0.0, 0.0, 1.0, 0.0])
            else:
                print('2')
                self.state = np.array([10.0, 0.0, 1.0, 0.0])
        # self.next_manifold = False
        else: 
            self.state = np.array([0.0, 0.0, 1.0, 0.0])
        return self.state

if __name__ == '__main__':
    env = SimEnv()
    env.state = np.array([1, 0.4, 1, 0])
    print(env.check_constraints())
    env.state = np.array([0.0, 0.0, 1.0, 0.0])
    print(env.check_constraints())
    env.state = np.array([1, -0.4, 1, 0])
    print(env.check_constraints())
    env.state = np.array([0, 0, 1, 0])
    print(env.check_constraints())
    env.state = np.array([10, 0, 1, 0])
    print(env.check_constraints())
    env.state = np.array([10.1, 2, 1, 0])
    print(env.check_constraints())
    env.state = np.array([9.9, 2, 1, 0])
    print(env.check_constraints())

    # for _ in range(10):
    #     env.state[0] += 1
    #     env.render() 