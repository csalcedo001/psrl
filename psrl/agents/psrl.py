import numpy as np

from .agent import Agent
from .utils import solve_tabular_mdp

class PSRLAgent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)

        self.env = env
        self.config = config

        mu = config.mu
        lambd = config.lambd
        alpha = config.alpha
        beta = config.beta

        n_s = env.observation_space.n
        n_a = env.action_space.n

        # Initialize posterior distributions
        self.p_dist = config.kappa * np.ones((n_s, n_a, n_s))
        self.r_dist = np.tile([mu, lambd, alpha, beta], (n_s, n_a, n_s, 1))

        self.pi = None
        self.buffer = []
        self.steps = 0

        self.update_policy()
    
    def act(self, state):
        self.steps += 1

        # if self.steps % self.config.tau == 0:
        self.update_policy()
        
        return self.pi[state]

    def observe(self, transition):
        self.buffer.append(transition)
    
    def update(self):
        self.update_posterior()
        self.update_policy()
    
    def update_posterior(self):
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        p_count = np.zeros((n_s, n_a, n_s))
        r_sum = np.zeros((n_s, n_a, n_s))

        for s, a, r, s_ in self.buffer:
            p_count[s, a, s_] += 1
            r_sum[s, a, s_] += r

        for s in range(n_s):
            for a in range(n_a):
                self.p_dist[s, a] += p_count[s, a]
        
                for s_ in range(n_s):
                    mu0, lambd, alpha, beta = self.r_dist[s, a, s_]
                    n = p_count[s, a, s_]

                    # Update normal-gamma distribution
                    mu = (lambd * mu0 + r_sum[s, a, s_]) / (lambd + n)
                    lambd += n
                    alpha += n / 2.
                    beta += (r_sum[s, a, s_] ** 2. + lambd * mu0 ** 2. - lambd * mu ** 2.) / 2

                    self.r_dist[s, a, s_] = [mu, lambd, alpha, beta]
        
        
        if self.steps % self.config.tau == 0:
             self.buffer = []

    def update_policy(self):
        # Sample from posterior
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        p = np.zeros((n_s, n_a, n_s))
        r = np.zeros((n_s, n_a, n_s))

        for s in range(n_s):
            for a in range(n_a):
                p[s, a] = np.random.dirichlet(self.p_dist[s, a])
                for s_ in range(n_s):
                    mu0, lambd, alpha, beta = self.r_dist[s, a, s_]

                    # Sample from normal-gamma distribution
                    tau = np.random.gamma(alpha, 1. / beta)
                    mu = np.random.normal(mu0, 1. / np.sqrt(lambd * tau))

                    r[s, a, s_] = mu


        # Solve for optimal policy
        self.pi, _ = solve_tabular_mdp(p, r, self.config.gamma, self.config.max_iter)