import numpy as np
from psrl.agents.utils import get_policy_average_reward

def brute_force_policy_evaluation(p, r, pi, gamma, epsilon, max_iter):
    n_s, n_a, _ = p.shape

    if len(pi.shape) == 1:
        pi_idx = pi
        pi = np.zeros((n_s, n_a))
        pi[np.arange(n_s), pi_idx] = 1

    # Manually executing policy evaluation
    v = np.zeros(n_s)
    for i in range(max_iter):
        # Initialize to 0
        delta = 0
        v_next = np.zeros(n_s)

        avg_r = get_policy_average_reward(p, r, pi)

        for s in range(n_s):
            for a in range(n_a):
                if gamma < 1:
                    v_next[s] += pi[s, a] * np.sum([
                        p[s, a, s_] * (r[s, a] + gamma * v[s_])
                        for s_ in range(n_s)
                    ])
                elif gamma == 1:
                    v_next[s] += pi[s, a] * np.sum([
                        p[s, a, s_] * (r[s, a] - avg_r + v[s_]) 
                        for s_ in range(n_s)
                    ])
            
            diff = abs(v_next[s] - v[s])
            delta = max(diff, delta)
        
        v = v_next
        
        if delta < epsilon:
            break
    
    return v


def brute_force_policy_average_reward(p, r, pi, epsilon=1e-2, max_iter=100):
    n_s, n_a = p.shape[:2]
    if len(pi.shape) == 1:
        pi_idx = pi
        pi = np.zeros((len(pi_idx), len(pi_idx)))
        pi[np.arange(len(pi_idx)), pi_idx] = 1
    
    mu_pi = brute_force_steady_state_distribution(p, pi, epsilon=epsilon, max_iter=max_iter)
    avg_reward = np.sum([
        mu_pi[s] * np.sum([
            pi[s, a] * r[s, a]
            for a in range(n_a)
        ])
        for s in range(n_s)
    ])

    return avg_reward



def brute_force_steady_state_distribution(p, pi, epsilon=1e-2, max_iter=100):
    n_s, n_a = p.shape[:2]
    mu_pi = np.ones(n_s) / n_s      # Initialize as uniform distribution

    for _ in range(int(max_iter)):
        next_mu_pi = np.zeros(n_s)
        for next_s in range(n_s):
            next_mu_pi[next_s] = np.sum([
                mu_pi[s] * np.sum([
                    pi[s, a] * p[s, a, next_s]
                    for a in range(n_a)
                ])
                for s in range(n_s)
            ])

        diff = np.abs(next_mu_pi - mu_pi).max()
        mu_pi = next_mu_pi

        if diff < epsilon:
            break

    # Normalize in case sum is not exactly 1
    mu_pi /= mu_pi.sum()  
    
    return mu_pi

def stationary_transition_matrix(P_pi, epsilon, max_iter):
    n_s = P_pi.shape[0]

    P_pi_prod = np.eye(n_s, n_s)
    P_pi_sum = np.zeros((n_s, n_s))
    P_pi_star = np.zeros((n_s, n_s))

    for n in range(1, max_iter):
        P_pi_prod = np.dot(P_pi_prod, P_pi)
        P_pi_sum += P_pi_prod
        P_pi_star_ = P_pi_sum / n

        if np.abs(P_pi_star_ - P_pi_star).max() < epsilon:
            break
            
        P_pi_star = P_pi_star_
    
    return P_pi_star

def average_reward_policy_evaluation(p, r, pi, epsilon, max_iter):
    P_pi = np.einsum('ijk,ij->ik', p, pi)
    R_pi = np.einsum('ij,ij->i', r, pi)

    P_pi_star = stationary_transition_matrix(P_pi, epsilon, max_iter)

    v_pi_star = (np.linalg.inv(np.eye(len(P_pi_star)) - P_pi + P_pi_star) - P_pi_star) @ R_pi

    return v_pi_star

grids_and_policies = {
    "simple": {
        "grid": [
            ['S', 'G']
        ],
        "optimal_policies": [       # Always goes right
            np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]),
            np.array([
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]),
            np.array([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]),
            np.array([
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]),
        ]
    },
    "square": {
        "grid": [
            ['S', ' '],
            [' ', 'G']
        ],
        "optimal_policies": [
            np.array([              # Always goes right
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0]
            ]),
            np.array([              # Toss a fair coin to go right or down
                [0, 0.5, 0.5, 0],
                [0,   0,   1, 0],
                [0,   1,   0, 0],
                [0,   0,   0, 1]
            ]),
            np.array([              # Biased towards going down
                [0, 0.1, 0.9, 0],
                [0,   0,   1, 0],
                [0,   1,   0, 0],
                [0,   0,   1, 0]
            ])
        ]
    }
}

