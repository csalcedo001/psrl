from .agent import Agent
from .utils import solve_tabular_mdp

class OptimalAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)

        self.env = env
        self.config = config

        self.p, self.r = env.get_p_and_r()
        self.pi, _ = solve_tabular_mdp(self.p, self.r, config.gamma, config.max_iter)

    def act(self, state):
        return self.pi[state]
    
    def observe(aelf, transition):
        pass

    def update(self):
        pass