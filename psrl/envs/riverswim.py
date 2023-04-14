import gym
from gym import spaces
import numpy as np


class RiverSwimEnv(gym.Env):
    def __init__(self,
            n=6,
            max_steps=20,
            p_swim=[0.35, 0.6],
            rewards=[0.005, 1]
        ):
        
        super().__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(n)

        self.max_steps = max_steps
        self.rewards = rewards

        if type(rewards) in [float, int]:
            self.rewards = [rewards, rewards]
        elif type(rewards) == list and len(rewards) == 2:
            self.rewards = rewards
        else:
            raise ValueError('Invalid reward specification')

        if type(p_swim) in [float, int]:
            self.p_swim = [p_swim, p_swim]
        elif type(p_swim) == list and len(p_swim) == 2:
            self.p_swim = p_swim
        else:
            raise ValueError('Invalid probability of swimming')

        # State
        self.pos = None
        self.steps = None

        self.reset()
    

    def reset(self):
        self.pos = 0
        self.steps = 0

        return self.pos
    

    def step(self, action):
        bottom = 0
        top = self.observation_space.n - 1

        # Get next state
        direction = 0
        if action == 1:
            p_swim = self.p_swim[0] if self.pos == 0 else self.p_swim[1]
            direction = np.random.choice(2, p=[1 - p_swim, p_swim])

        direction = direction * 2 - 1

        next_pos = min(max(self.pos + direction, bottom), top)


        # Get reward
        reward = 0
        if action == 0 and self.pos == bottom and next_pos == bottom:
            reward = self.rewards[0]
        elif action == 1 and self.pos == top and next_pos == top:
            reward = self.rewards[1]


        # Get termination condition
        done = self.steps == self.max_steps - 1


        # Update state
        self.pos = next_pos
        self.steps += 1

        return self.pos, reward, done, {}