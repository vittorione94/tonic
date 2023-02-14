'''Non-differentiable noisy exploration methods.'''

import numpy as np


class NoActionNoise:
    def __init__(self, start_steps=20000):
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets, observations=None):
        pass


class NormalActionNoise:
    def __init__(self, scale=0.1, start_steps=20000):
        self.scale = scale
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)
            noises = self.scale * self.np_random.normal(size=actions.shape)
            actions = (actions + noises).astype(np.float32)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets, observations=None):
        pass


class OrnsteinUhlenbeckActionNoise:
    def __init__(
        self, scale=0.1, clip=2, theta=.15, dt=1e-2, start_steps=20000
    ):
        self.scale = scale
        self.clip = clip
        self.theta = theta
        self.dt = dt
        self.start_steps = start_steps

    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)
        self.noises = None

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            actions = self.policy(observations)

            if self.noises is None:
                self.noises = np.zeros_like(actions)
            noises = self.np_random.normal(size=actions.shape)
            noises = np.clip(noises, -self.clip, self.clip)
            self.noises -= self.theta * self.noises * self.dt
            self.noises += self.scale * np.sqrt(self.dt) * noises
            actions = (actions + self.noises).astype(np.float32)
            actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets, observations=None):
        if self.noises is not None:
            self.noises *= (1. - resets)[:, None]


class DEPActionNoise:
    def __init__(
        self, scale=0.1, clip=2, theta=.15, dt=1e-2, start_steps=20000, epsilon=0.01, alpha=1e-2, alpha_t=0.03, DEP_horizon=4,
        k=100,  force_scale=0.003
    ):
        self.scale = scale
        self.clip = clip
        self.theta = theta
        self.dt = dt
        self.start_steps = start_steps

        self.eps = epsilon
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.k = k
        self.force_scale = force_scale
        self.DEP_horizon = DEP_horizon


    def initialize(self, policy, action_space, seed=None):
        self.policy = policy
        self.action_size = action_space.shape[0]
        self.np_random = np.random.RandomState(seed)

        self.C = self.np_random.uniform(-1, 1, (1, self.action_size, self.action_size))
        self.bias = self.np_random.uniform(-1, 1, (1, self.action_size))

        self.actions = None
        self.prev_obs = np.zeros((1, self.action_size))  # used to compute ensro changes
        self.prev_obs2 = np.zeros((1, self.action_size))  # used to compute ensro changes

        self.DEP = False
        self.DEP_steps = self.DEP_horizon
        self.noises = None

    def __call__(self, observations, steps):
        if steps > self.start_steps:
            p = self.np_random.random()
            if not self.DEP and p < self.eps:
                self.DEP = True

            muscle_lengths = observations[:, :self.action_size]
            muscle_forces = observations[:, self.action_size:self.action_size*2] * self.force_scale
            DEP_observations = (muscle_lengths + muscle_forces) 
            self.prev_obs = DEP_observations

            if self.DEP:
                num_batch = observations.shape[0]
                if num_batch !=  self.C.shape[0]:
                    self.C = np.repeat(self.C, num_batch, 0)
                    self.bias = np.repeat(self.bias, num_batch, 0)

                norm = np.linalg.norm(self.C)
                self.DEP_actions = np.tanh((((self.C/ norm) * self.k) @ self.prev_obs[:,:, None]).squeeze(-1)  + self.bias)
                actions = self.DEP_actions

                self.DEP_steps -= 1 

                if self.DEP_steps <= 0:
                    self.DEP = False
                    self.DEP_steps = self.DEP_horizon
            else:
                # OU
                actions = self.policy(observations)

                if self.noises is None:
                    self.noises = np.zeros_like(actions)
                noises = self.np_random.normal(size=actions.shape)
                noises = np.clip(noises, -self.clip, self.clip)
                self.noises -= self.theta * self.noises * self.dt
                self.noises += self.scale * np.sqrt(self.dt) * noises
                actions = (actions + self.noises).astype(np.float32)
                actions = np.clip(actions, -1, 1)
        else:
            shape = (len(observations), self.action_size)
            actions = self.np_random.uniform(-1, 1, shape)
        return actions

    def update(self, resets, observations=None):
        # if self.DEP:
        muscle_lengths = observations[:, :self.action_size]
        muscle_forces = observations[:, self.action_size:self.action_size*2] * self.force_scale
        DEP_observations = (muscle_lengths + muscle_forces) 
        
        delta2 = (self.prev_obs - self.prev_obs2)
        delta2 = np.expand_dims(delta2, 2)
        self.prev_obs2 = self.prev_obs
        delta = (DEP_observations - self.prev_obs)
        delta = np.expand_dims(delta, 2)
        delta2 = np.transpose(delta2, axes=(0, 2, 1))

        self.C = (1 - self.alpha) * self.C + self.alpha * (delta @ delta2)

        if self.DEP:
            self.bias = (1 - self.alpha_t) * self.bias - self.alpha_t * self.DEP_actions

        if self.noises is not None:
            self.noises *= (1. - resets)[:, None]