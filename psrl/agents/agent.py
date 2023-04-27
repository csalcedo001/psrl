class Agent():
    def __init__(self, env, config):
        pass

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