import numpy as np
import pickle
import torch
import torch.distributions as dist

from psrl.agents.agent import Agent
from psrl.agents.utils import solve_tabular_mdp

class PSRL_1(Agent):
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
        self.p_dist = config.kappa * torch.ones((n_s, n_a, n_s))
        self.r_dist = torch.tile(torch.Tensor([mu, lambd, alpha, beta]), (n_s, n_a, 1))

        self.pi = None
        self.p_count = np.zeros((n_s, n_a, n_s)).tolist()
        self.r_total = np.zeros((n_s, n_a)).tolist()
        self.steps = 0

        self.update_policy()
    
    def act(self, state):
        self.steps += 1

        # if self.steps % self.config.tau == 0:
        self.update_policy()
        
        return self.pi[state]

    def observe(self, transition):
        s, a, r, s_ = transition

        self.p_count[s][a][s_] += 1
        self.r_total[s][a] += r
    
    def update(self):
        if self.steps % self.config.tau == 0:
            self.update_posterior()
            self.update_policy()
    
    def update_posterior(self):
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        p_count = torch.Tensor(self.p_count)
        r_total = torch.Tensor(self.r_total)

        # Update transition probabilities
        self.p_dist += p_count

        # Update reward function
        mu0, lambd, alpha, beta = torch.moveaxis(self.r_dist, 2, 0)

        r_count = p_count.sum(axis=2)
        mu = (lambd * mu0 + r_total) / (lambd + r_count)
        lambd += r_count
        alpha += r_count / 2.
        beta += (r_total ** 2. + lambd * mu0 ** 2. - lambd * mu ** 2.) / 2

        self.r_dist = torch.stack([mu, lambd, alpha, beta], dim=2)
        
        
        if self.steps % self.config.tau == 0:
            self.p_count = np.zeros((n_s, n_a, n_s)).tolist()
            self.r_total = np.zeros((n_s, n_a)).tolist()

    def update_policy(self):
        ### Sample from posterior
        # Compute transition probabilities
        p = dist.Dirichlet(self.p_dist).sample()

        # Compute reward function
        mu0, lambd, alpha, beta = torch.moveaxis(self.r_dist, 2, 0)

        tau = dist.Gamma(alpha, 1. / beta).sample()
        mu = dist.Normal(mu0, 1. / torch.sqrt(lambd * tau)).sample()

        r = mu


        ### Solve for optimal policy
        self.pi, _ = solve_tabular_mdp(
            p,
            r,
            gamma=self.config.gamma,
            max_iter=self.config.max_iter
        )
    
    def save(self, path):
        data = {
            'p_dist': self.p_dist,
            'r_dist': self.r_dist,
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data, out_file)
    
    def load(self, path):
        with open(path, 'rb') as in_file:
            data = pickle.load(in_file)

        self.p_dist = data['p_dist']
        self.r_dist = data['r_dist']