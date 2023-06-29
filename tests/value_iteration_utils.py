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
            
            if gamma < 1:
                diff = abs(v_next[s] - v[s])
            elif gamma == 1:
                diff = abs(v_next[s] / max(i, 1) - v[s] / (i + 1))
            delta = max(diff, delta)
        
        # For most cases where gamma = 1 and reward = 1, this stop
        # condition won't go through
        if delta < epsilon:
            break
        
        v = v_next
    
    if gamma == 1:
        v = v / i
    
    return v



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

