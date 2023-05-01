import numpy as np
import torch


def solve_tabular_mdp(p, r, gamma, max_iter, device='cpu'):
    p = torch.Tensor(p)
    r = torch.Tensor(r)

    n_s = p.shape[0]

    s_idx = torch.arange(n_s)

    ones = torch.eye(n_s)
    pi = torch.zeros(n_s, dtype=int)

    p_r = torch.einsum('ijk, ijk -> ij', p, r)
    q = None

    for i in range(max_iter):
        # Solve for Q values
        pi_idx = pi.numpy()
        v = torch.linalg.solve(ones - gamma * p[s_idx, pi_idx, :], p_r[s_idx, pi_idx])
        q = p_r + gamma * torch.einsum('ijk, k -> ij', p, v)
        pi_ = torch.argmax(q, axis=1)
        
        if torch.all(pi_ == pi):
            break
        else:
            pi = pi_

    pi = pi.numpy()
    q = q.numpy()

    return pi, q