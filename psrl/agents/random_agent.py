from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)
        
        self.env = env
        self.config = config

    def act(self, state):
        return self.env.action_space.sample()

    def observe(self, transition):
        pass

    def update(self):
        pass