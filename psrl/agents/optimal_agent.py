from .agent import Agent
from .utils import solve_tabular_mdp

class OptimalAgent(Agent):
    def __init__(self, env, gamma, max_iter):
        super().__init__()

        self.env = env

        self.p, self.r = env.get_p_and_r()
        self.pi, _ = solve_tabular_mdp(self.p, self.r, gamma, max_iter)

    def act(self, state):
        return self.pi[state]
    
    def observe(aelf, transition):
        pass

    def update(self):
        pass