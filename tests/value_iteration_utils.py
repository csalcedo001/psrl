import numpy as np

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

        for s in range(n_s):
            for a in range(n_a):
                v_next[s] += pi[s, a] * np.sum([p[s, a, s_] * (r[s, a] + gamma * v[s_]) for s_ in range(n_s)])
            
            diff = abs(v_next[s] - v[s])
            delta = max(diff, delta)
        
        # For most cases where gamma = 1 and reward = 1, this stop
        # condition won't go through
        if delta < epsilon:
            break
        
        v = v_next
    
    if gamma == 1:
        v = v / i
    
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



grids_and_policies = {
    "square": {
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

