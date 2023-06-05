import torch
import numpy as np

def solve_tabular_mdp(p, r, max_iter, gamma=1):
    return value_iteration(p, r, max_iter, gamma)


def value_iteration(p, r, max_iter, gamma=1):
    pi, (_, q, _, _) = extended_value_iteration(p, r, max_iter, 0, 0, gamma)

    return pi, q

def extended_value_iteration(p, r, max_iter, cb_p=0, cb_r=0, gamma=1, epsilon=1e-3):
    p = torch.Tensor(p)
    r = torch.Tensor(r)

    n_s, n_a = r.shape[:2]

    s_idx = torch.arange(n_s)

    ones = torch.eye(n_s)
    v = torch.zeros(n_s)
    pi = torch.zeros(n_s, dtype=int)

    if len(r.shape) == 3:
        r = r.sum(axis=2)
    
    r_tilde = r + cb_r
    r_tilde = r_tilde.float()

    
    q = None

    for i in range(int(max_iter)):
        # Compute p tilde
        if cb_p is 0 and cb_r is 0:
            p_tilde = p[:]
        else:
            p_s_order = np.argsort(-v)

            p = p.numpy()
            p_tilde = np.zeros_like(p)
            for s in range(n_s):
                for a in range(n_a):
                    p_tilde[s, a] = inner_maximization(p[s, a], cb_p[s, a], p_s_order)
            p = torch.Tensor(p)
            p_tilde = torch.Tensor(p_tilde)

        # Compute policy pi
        p_r = torch.einsum('ijk, ij -> ij', p_tilde, r_tilde)
        q = p_r + gamma * torch.einsum('ijk, k -> ij', p_tilde, v)
        v_ = torch.max(q, axis=1).values

        dists = torch.abs(v_ - v)
        diff = dists.max() - dists.min()
        
        # Check if value function is epsilon-optimal
        if diff < epsilon:
            break
            
        v = v_

    # Get policy from Q function
    pi = torch.argmax(q, axis=1)

    pi = pi.numpy()

    return pi, (v, q, p_tilde, r_tilde)




def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank):
    '''
    Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
    Arg:
        p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
        confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
        rank : (n_states)-shaped int array. The sorted list of states in descending order of value.
    Return:
        (n_states)-shaped float array. The optimistic transition p(.|s, a).
    '''
    
    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = np.minimum(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()

    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])

        if len(rank_dup) == 0:
            break

        last = rank_dup.pop()
    
    return p_sa