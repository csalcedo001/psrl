import numpy as np

class PSRLAgent():
    def __init__(self, env, gamma, kappa, mu, lambd, alpha, beta, max_iter):
        self.env = env
        self.gamma = gamma
        self.max_iter = max_iter

        n_s = env.observation_space.n
        n_a = env.action_space.n

        # Initialize posterior distributions
        self.p_dist = kappa * np.ones((n_s, n_a, n_s))
        self.r_dist = np.tile([mu, lambd, alpha, beta], (n_s, n_a, n_s, 1))

        self.pi = None
        self.buffer = []

        self.update_policy()
    
    def act(self, state):
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
        s_idx = np.arange(n_s)
        
        ones = np.eye(n_s)
        pi = np.zeros(n_s, dtype=int)
        
        p_r = np.einsum('ijk, ijk -> ij', p, r)
        
        for i in range(self.max_iter):
            # Solve for Q values
            v = np.linalg.solve(ones - self.gamma * p[s_idx, pi, :], p_r[s_idx, pi])
            q = p_r + self.gamma * np.einsum('ijk, k -> ij', p, v)

            # Get greedy policy - break ties at random
            pi_ = np.array([np.random.choice(np.argwhere(qs == np.amax(qs))[0]) \
                            for qs in q])
            
            if np.prod(pi_ == pi) == 1:
                break
            else:
                pi = pi_

        self.pi = pi