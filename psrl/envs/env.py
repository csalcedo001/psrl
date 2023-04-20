import gym

class Env(gym.Env):
    def __init__(self):
        gym.Env.__init__(self)

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def get_p_and_r(self):
        raise NotImplementedError