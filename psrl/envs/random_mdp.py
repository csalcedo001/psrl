from gym import spaces
import numpy as np

from psrl.envs import Env

class RandomMDPEnv(Env):
    def __init__(self, n_states, n_actions, max_steps=10000):
        Env.__init__(self)

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_states)

        self.max_steps = max_steps

        # For Diritchlet distribution
        self.alpha = np.ones((n_states)) * 1. / n_states

        self.transitions = None
        self.rewards = None
        self.state = None
        self.steps = None

        self.randomize()
        self.reset()
    

    def reset(self):
        self.state = 0
        self.steps = 0

        return self.state


    def step(self, action):
        # Get next state
        probs = self.transitions[self.state, action]
        next_state = np.random.choice(self.observation_space.n, p=probs)

        # Get reward
        r_mean = self.r_mean[self.state, action, next_state]
        r_var = self.r_var[self.state, action, next_state]
        reward = np.random.normal(r_mean, r_var)

        # Get termination condition
        done = self.steps == self.max_steps - 1

        # Updat state
        self.state = next_state
        self.steps += 1

        return self.state, reward, done, {}


    def randomize(self):
        n_s = self.observation_space.n
        n_a = self.action_space.n

        self.transitions = np.random.dirichlet(self.alpha, size=(n_s, n_a))
        self.r_var = np.random.gamma(1, 1, size=(n_s, n_a, n_s))
        self.r_mean = np.random.normal(1, 1, size=(n_s, n_a, n_s))
    
    def get_p_and_r(self):
        return self.transitions, self.r_mean