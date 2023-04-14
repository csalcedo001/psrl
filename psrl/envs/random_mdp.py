import gym
from gym import spaces
import numpy as np

class RandomMDPEnv(gym.Env):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_states)

        self.n_states = n_states
        self.n_actions = n_actions

        # For Diritchlet distribution
        self.alpha = np.ones((n_states, n_actions, n_states))

        self.transitions = None
        self.randomize()

        self.state = 0

        self.reset()
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state = np.random.randint(0, self.n_states)
        return self.state, 0, False, {}

    def randomize(self):
        transitions = np.random.dirichlet(self.alpha, size=(self.n_states, self.n_actions))
        rewards = np.random.gamma(1, 1, size=(self.n_states, self.n_actions))