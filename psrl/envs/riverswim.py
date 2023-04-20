from gym import spaces
import numpy as np

from psrl.envs import Env


class RiverSwimEnv(Env):
    def __init__(self,
            n=6,
            max_steps=20,
            swim_probs={
                "bottom": [0.0, 0.4, 0.6],
                "middle": [0.05, 0.6, 0.35],
                "top":    [0.4, 0.6, 0.0],
            },
            rewards=[0.005, 1],
        ):
        
        Env.__init__(self)

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

        if type(swim_probs) != dict or swim_probs.keys() != {"bottom", "middle", "top"}:
            raise ValueError('Invalid probability of swimming')
        self.swim_probs = swim_probs


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
        if self.pos == bottom:
            swim_probs = self.swim_probs["bottom"]
        elif self.pos == top:
            swim_probs = self.swim_probs["top"]
        else:
            swim_probs = self.swim_probs["middle"]

        direction = 0
        if action == 1:
            direction = np.random.choice(len(swim_probs), p=swim_probs)

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

    def get_p_and_r(self):
        n_s = self.observation_space.n
        n_a = self.action_space.n

        p = np.zeros((n_s, n_a, n_s))
        r = np.zeros((n_s, n_a, n_s))

        bottom = 0
        top = self.observation_space.n - 1

        for s in range(n_s):
            for a in range(n_a):
                if s == bottom:
                    swim_probs = self.swim_probs["bottom"]
                elif s == top:
                    swim_probs = self.swim_probs["top"]
                else:
                    swim_probs = self.swim_probs["middle"]

                if a == 0:
                    p[s, a, max(s - 1, bottom)] = 1
                    if s == bottom:
                        r[s, a, s] = self.rewards[0]
                else:
                    p[s, a, s] = swim_probs[1]
                    if s != bottom:
                        p[s, a, s - 1] = swim_probs[0]
                    if s != top:
                        p[s, a, s + 1] = swim_probs[2]
                    
                    if s == top:
                        r[s, a, s] = self.rewards[1]
        
        return p, r