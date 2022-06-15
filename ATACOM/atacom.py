from mushroom_rl.utils.spaces import *
from atacom.utils.null_space_coordinate import rref, gram_schmidt, pinv_null

class AtacomEnvWrapper:
    
    """
    Environment wrapper of ATACOM
    """

    def __init__(self, base_env, dim_q, vel_max, acc_max, f=None, g=None, Kc=1, Kq=10, time_step=0.01):
        """
        Constructor
        Args:
            base_env(mushroomrl.Core.Environment): The base environment inherited from
            dim_q (int): [int] dimension of the directly controllable variable
            vel_max (array, float): the maximum velocity of the directly contollable variable
            acc_max (array, float): the maximum acceleration of the directly controllable variable
            f (ViabilityConstraint, ConstraintsSet): the equality constraint f(q) = 0
            g (ViabilityConstraint, ConstraintsSet): the inequality constraint g(q) = 0
            Kc (array, float): the scaling factor for error correction
            Ka (array, float): the scaling factor for the vaibility acceleration bound
            time_step (float): the step size for time discretization
        """ 
        self.env = base_env
        self.dims = {'q': dim_q, 'f': 0, 'g': 0}
        self.f = f
        self.g = g
        self.time_step = time_step
        self._logger = None

        if self.f is not None:
            assert self.dims['q'] == self.f.dim_q, "Input dimension is different in f"
            self.dims['f'] = self.f.dim_out
        if self.g is not None:
            assert self.dims['q'] == slef.g.dim_q, "Input dimension is different in g"
            self.dims['g'] = self.g.dim_out
            self.s = np.zeros(self.dims['g'])
        
        self.dims['null'] = self.dims['q'] - self.dims['f']
        self.dims['c'] = self.dims['f'] + self.dims['g']

        if np.isscalar(Kc):
            self.K_c = np.ones(self.dims['c']) * Kc
        else:
            self.K_c = Kc

        self.q = np.zeros(self.dims['q'])
        self.dq = np.zeros(self.dims['q'])

        self._mdp_info = self.env.info.copy()
        self._mdp_info.action_space = Box(low=-np.ones(self.dims['null']), high=np.ones(self.dims['null']))

        if np.isscalar(vel_max):
            self.vel_max = np.ones(self.dims['q']) * vel_max
        else:
            self.vel_max = vel_max
            assert np.shape(self.vel_max)[0] == self.dims['q']
        
        if np.isscalar(acc_max):
            self.acc_max = np.ones(self.dims['q']) * acc_max
        else:
            self.acc_max = acc_max
            assert np.shape(self.acc_max)[0] == self.dims['q']

        if np.isscalar(Kq):
            self.K_q = np.ones(self.dims['q']) * Kq
        else:
            self.K_q = Kq
            assert np.shape(self.K_q)[0] == self.dims['q']

        self.alpha_max = np.ones(self.dims['null']) * self.acc_max.max()

        self.state = self.env.reset()
        self._act_a = None
        self._act_b = None
        self._act_err = None

        self.constr_logs = list()
        self.env.step_action_function = self.step_action_function

    def _get_q(self, state):
        raise NotImplementedError
    
    def _get_dq(self, state):
        raise NotImplementedError
    
    def acc_to_ctrl_action(self, ddq):
        raise NotImplementedError
    
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, state=None):
        self.state = self.env.reset(state)
        self.q = self._get_q(self.state)
        self.dq = self._get_dq(self.state)
        self._compute_slack_variables()
        return self.state
    
    def render(self):
        self.env.render()

    def stop(self):
        self.env.stop()
    
    def step(self, action):
        alpha = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        alpha = alpha * self.alpha_max

        self.state, reward, absorb, info = self.env.step(alpha)
        self.q = self._get_q(self.state)
        self.dq = self._get_dq(self.state)
        if not hasattr(self.env, 'get_constraints_logs'):
            self._update_constraint_stats(self.q, self.dq)
        return self.state.copy(), reward, absorb, info
    
    def acc_truncation(self, dq, ddq):
        acc_u = np.maximum(np.minimum(self.acc_max, -self.K_q * (dq - self.vel_max)), -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max, -self.K_q * (dq + self.vel_max)), self.acc_max)
        ddq = np.clip(ddq, acc_l, acc_u)
        return ddq

    def step_action_function(self, sim_state, alpha):
        self.state = self.env._create_observation(sim_state)

        Jc, psi = self._construct_Jc_psi(self.q, self.s, self.dq)
        Jc_inv, Nc = pinv_null(Jc)
        Nc = rref(Nc[:, :self.dims['null']], row_vectors=False, tol=0.05)

        self._act_a = -Jc_inv @ psi
        self._act_b = Nc @ alpha
        self._act_err = self._compute_error_correction(self.q, self.dq, self.s, Jc_inv)
        ddq_ds = self._act_a + self._act_b + self._act_err

        self.s += ddq_ds[self.dims['q']:(self.dims['q'] + self.dims['g'])] * self.time_step

        ddq = self.acc_truncation(self.dq, ddq_ds[:self.dims['q']])
        ctrl_action = self.acc_to_ctrl_action(ddq)
        return ctrl_action

    ## TODO: utils function and helper functions for step_action_function()