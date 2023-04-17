from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__()
        
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()

    def observe(self, transition):
        pass

    def update(self):
        pass