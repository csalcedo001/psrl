class Agent():
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def reset(self, state):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def observe(self, transition):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError