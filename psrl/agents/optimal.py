from .agent import Agent
from .utils import solve_tabular_mdp

class OptimalAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)

        self.env = env
        self.config = config

        self.p, self.r = env.get_p_and_r()
        self.pi, _ = solve_tabular_mdp(
            self.p,
            self.r,
            gamma=config.gamma,
            max_iter=config.max_iter
        )

    def reset(self, staet):
        pass
    
    def act(self, state):
        return self.pi[state]
    
    def observe(aelf, transition):
        pass

    def update(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass