import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.viewer import Viewer


class CircularMotion(Environment):
    """
    Base environment of CircularMotion Environment
    A point is moving on the 2D unit circular_motion.
    Control actions are 2d acceleration along each direction
    """

    def __init__(self, time_step=0.01, horizon=500, gamma=0.99, random_init=False):
        self.time_step = time_step
        self.random_init = random_init
        inf_array = np.ones(4) * np.inf
        observation_space = spaces.Box(low=-inf_array, high=inf_array)
        action_space = spaces.Box(low=-np.ones(2) * 1, high=np.ones(2) * 1)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        self._viewer = Viewer(env_width=2.5, env_height=2.5, background=(255, 255, 255))
        self.step_action_function = None
        self.action_scale = np.array([10., 10.])
        super().__init__(mdp_info)

        self.c = np.zeros(3)
        self.constr_logs = list()

    def reset(self, state=None):
        if state is None:
            if self.random_init:
                y = np.random.uniform(-.5, 1)
                x = np.sqrt(1 - y ** 2) * np.sign(np.random.uniform(-1, 1))
                dx = np.random.uniform(-1, 1)
                dy = -x * dx / y
                v = np.array([dx, dy])
                v = v / np.linalg.norm(v) * np.random.uniform(0, 1)
                self._state = np.array([x, y, v[0], v[1]])
            else:
                self._state = np.array([-1., 0., 0., 0.0])
        else:
            if abs(state[0] ** 2 + state[1] ** 2 - 1) < 1e-6 and abs(state[0] * state[2] - state[1] * state[3]) < 1e-6:
                self._state = state
            else:
                raise ValueError("Can not reset to the state: ", state)
        return self._state

    
    def step(self, action):
        self.check_constraint()

        if self.step_action_function is not None:
            action = self.step_action_function(self._state, action)
        
        self._action = np.clip(action ,self.info.action_space.low, self.info.action_space.high)
        self._action = self._action * self.action_scale

        self._state[:2] += self._state[2:4] * self.time_step + self._action * self.time_step ** 2 / 2
        self._state[2:4] += self._action * self.time_step

        reward = np.exp(-np.linalg.norm(np.array([1., 0.]) - self._state[:2]))

        return self._state, reward, False, dict() 

    
    ### Display circle and lines
    def render(self):
        offset = np.array([1.25, 1.25])
        self._viewer.circle(center=offset, radius=1, color=(0., 0., 255), width=5)
        self._viewer.line(start=np.array([-1.25, -0.5]) + offset, end=np.array([1.25, -0.5]) + offset, 
                            color=(255, 20, 147), width=3)
        self._viewer.square(center=np.array([1.0, 0.0]) + offset, angle=0, edge=0.05, color=(50, 205, 50))

        pos = self._state[:2] + offset
        self._viewer.circle(center=pos, radius=0.03, color=(255, 0, 0))
        self._viewer.display(self.time_step)

    
    def _create_sim_state(self):
        reutrn self._state

    
    def _create_observation(self, state):
        return state
    

    def check_constraint(self):
        q = self._state[:2]
        dq = self._state[2:4]

        self.c = self.get_c(q, dq)
        self.c[0] = np.abs(self.c[0])
        self.constr_logs.append(self.c)


    def get_c(self, q, dq):
        return np.concatenate([self.c_1(q), self.c_2(q), self.c_3(q, dq)])
    

    @staticmethod
    def c_1(q):
        return np.array([q[0] ** 2 + q[1] ** 2 - 1])
    
    @staticmethod
    def c_2(q):
        return np.array([-q[1] - 0.5])

    @staticmethod
    def c_3(q, dq):
        return np.abs(dq) - 1

    
    def get_constraints_logs(self):
        constr_logs = np.array(self.constr_logs)
        c_avg = np.mean(np.max(constr_logs[:, :2], axis=1))
        c_max = np.max(constr_logs[:, :2])
        c_dq_max = np.max(constr_logs[:, 2:])
        self.constr_logs.clear()
        return c_avg, c_max, c_dq_max

    
    
    

