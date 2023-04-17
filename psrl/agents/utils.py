import numpy as np


def solve_tabular_mdp(p, r, gamma, max_iter):
    n_s = p.shape[0]

    s_idx = np.arange(n_s)

    ones = np.eye(n_s)
    pi = np.zeros(n_s, dtype=int)

    p_r = np.einsum('ijk, ijk -> ij', p, r)
    q = None

    for i in range(max_iter):
        # Solve for Q values
        v = np.linalg.solve(ones - gamma * p[s_idx, pi, :], p_r[s_idx, pi])
        q = p_r + gamma * np.einsum('ijk, k -> ij', p, v)

        # Get greedy policy - break ties at random
        pi_ = np.array([np.random.choice(np.argwhere(qs == np.amax(qs))[0]) \
                        for qs in q])
        
        if np.prod(pi_ == pi) == 1:
            break
        else:
            pi = pi_

    return pi, q