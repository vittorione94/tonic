'''Some basic non-learning agents used for example for debugging.'''

import numpy as np

from tonic import agents


class NormalRandom(agents.Agent):
    '''Random agent producing actions from normal distributions.'''

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        batch_size = len(observations)
        shape = (batch_size, self.action_size)
        return self.np_random.normal(self.loc, self.scale, shape)


class UniformRandom(agents.Agent):
    '''Random agent producing actions from uniform distributions.'''

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        batch_size = len(observations)
        shape = (batch_size, self.action_size)
        return self.np_random.uniform(-1, 1, shape)


class OrnsteinUhlenbeck(agents.Agent):
    '''Random agent producing correlated actions from an OU process.'''

    def __init__(self, scale=0.2, clip=2, theta=.15, dt=1e-2):
        self.scale = scale
        self.clip = clip
        self.theta = theta
        self.dt = dt

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)
        self.train_actions = None
        self.test_actions = None

    def step(self, observations, steps):
        return self._train_policy(observations)

    def test_step(self, observations, steps):
        return self._test_policy(observations)

    def _train_policy(self, observations):
        if self.train_actions is None:
            shape = (len(observations), self.action_size)
            self.train_actions = np.zeros(shape)
        self.train_actions = self._next_actions(self.train_actions)
        return self.train_actions

    def _test_policy(self, observations):
        if self.test_actions is None:
            shape = (len(observations), self.action_size)
            self.test_actions = np.zeros(shape)
        self.test_actions = self._next_actions(self.test_actions)
        return self.test_actions

    def _next_actions(self, actions):
        noises = self.np_random.normal(size=actions.shape)
        noises = np.clip(noises, -self.clip, self.clip)
        next_actions = (1 - self.theta * self.dt) * actions
        next_actions += self.scale * np.sqrt(self.dt) * noises
        next_actions = np.clip(next_actions, -1, 1)
        return next_actions

    def update(self, observations, rewards, resets, terminations, steps):
        self.train_actions *= (1. - resets)[:, None]

    def test_update(self, observations, rewards, resets, terminations, steps):
        self.test_actions *= (1. - resets)[:, None]


class DEP(agents.Agent):
    '''Random agent producing correlated actions from an OU process.'''

    def __init__(self, alpha=1e-1, alpha_t=0.03, k=100, force_scale=0.003):
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.k = k
        self.force_scale = force_scale

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)
        self.C = self.np_random.uniform(-1, 1, (1, self.action_size, self.action_size))
        self.bias = self.np_random.uniform(-1, 1, (1, self.action_size))
        self.train_actions = None

        self.test_actions = None
        self.prev_obs = None  # used to compute sensor changes
        self.prev_obs2 = np.zeros(self.action_size)  # used to compute sensor changes

    def step(self, observations, steps):
        return self._train_policy(observations)

    def test_step(self, observations, steps):
        return self._test_policy(observations)

    def _test_policy(self, observations):
        if self.test_actions is None:
            shape = (self.action_size, self.action_size)
            self.test_actions = np.zeros(shape)
        
        num_batch = observations.shape[0]
        if num_batch !=  self.C.shape[0]:
            self.C = np.repeat(self.C, num_batch, 0)
            self.bias = np.repeat(self.bias, num_batch, 0)

        # DEP
        muscle_lengths = observations[:, :self.action_size]
        muscle_forces = observations[:, self.action_size:self.action_size*2] * self.force_scale
        observations = (muscle_lengths + muscle_forces) 
        self.prev_obs = observations

        norm = np.linalg.norm(self.C)
        self.test_actions = np.tanh((((self.C/ norm) * self.k) @ observations[:,:, None]).squeeze(-1)  + self.bias)
        return self.test_actions

    def update(self, observations, rewards, resets, terminations, steps):
        self.train_actions *= (1. - resets)[:, None]

    def test_update(self, observations, rewards, resets, terminations, steps):
        muscle_lengths = observations[:, :self.action_size]
        muscle_forces = observations[:, self.action_size:self.action_size*2] * self.force_scale
        observations = (muscle_lengths + muscle_forces) 
        
        delta2 = (self.prev_obs - self.prev_obs2)
        delta2 = np.expand_dims(delta2, 2)
        self.prev_obs2 = self.prev_obs
        delta = (observations - self.prev_obs)
        delta = np.expand_dims(delta, 2)
        delta2 = np.transpose(delta2, axes=(0, 2, 1))

        self.C = (1 - self.alpha) * self.C + self.alpha * (delta @ delta2)
        self.bias = (1 - self.alpha_t) * self.bias - self.alpha_t * self.test_actions
        self.test_actions *= (1. - resets)[:, None]        

class Constant(agents.Agent):
    '''Agent producing a unique constant action.'''

    def __init__(self, constant=0):
        self.constant = constant

    def initialize(self, observation_space, action_space, seed=None):
        self.action_size = action_space.shape[0]

    def step(self, observations, steps):
        return self._policy(observations)

    def test_step(self, observations, steps):
        return self._policy(observations)

    def _policy(self, observations):
        shape = (len(observations), self.action_size)
        return np.full(shape, self.constant)
