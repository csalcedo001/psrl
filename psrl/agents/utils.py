import torch
import numpy as np
import math


def solve_tabular_mdp(p, r, gamma, max_iter):
    p = torch.Tensor(p)
    r = torch.Tensor(r)

    n_s = p.shape[0]

    s_idx = torch.arange(n_s)

    ones = torch.eye(n_s)
    pi = torch.zeros(n_s, dtype=int)

    if len(r.shape) == 2:
        p_r = torch.einsum('ijk, ij -> ij', p, r)
    elif len(r.shape) == 3:
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
    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa


def extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, epsilon):
    '''
    The extended value iteration which finds an optimistic MDP within the plausible set of MDPs and solves for its near-optimal policy.
    '''
    # Initial values (an optimal 0-step non-stationary policy's values)
    state_value_hat = np.zeros(n_states)
    next_state_value_hat = np.zeros(n_states)
    du = np.zeros(n_states)
    du[0], du[-1] = math.inf, -math.inf
    # Optimistic MDP and its epsilon-optimal policy
    p_tilde = np.zeros((n_states, n_actions, n_states))
    r_tilde = r_hat + confidence_bound_r
    pi_tilde = np.zeros(n_states, dtype='int')
    while not du.max() - du.min() < epsilon:
        # Sort the states by their values in descending order
        rank = np.argsort(-state_value_hat)
        for st in range(n_states):
            best_ac, best_q = None, -math.inf
            for ac in range(n_actions):
                # print('opt', st, ac)
                # print(state_value_hat)
                # Optimistic transitions
                p_sa_tilde = inner_maximization(p_hat[st, ac], confidence_bound_p[st, ac], rank)
                q_sa = r_tilde[st, ac] + (p_sa_tilde * state_value_hat).sum()
                p_tilde[st, ac] = p_sa_tilde
                if best_q < q_sa:
                    best_q = q_sa
                    best_ac = ac
                    pi_tilde[st] = best_ac
            next_state_value_hat[st] = best_q
            # print(state_value_hat)
        du = next_state_value_hat - state_value_hat
        state_value_hat = next_state_value_hat
        next_state_value_hat = np.zeros(n_states)
        # print('u', state_value_hat, du.max() - du.min(), epsilon)
    return pi_tilde, (p_tilde, r_tilde)