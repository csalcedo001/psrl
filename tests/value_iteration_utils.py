import numpy as np

def brute_force_value_iteration(p, r, pi, gamma, epsilon, max_iter):
    n_s, n_a, _ = p.shape

    # Manually executing policy evaluation
    v = np.zeros(n_s)
    for _ in range(max_iter):
        # Initialize to 0
        delta = 0
        v_next = np.zeros(n_s)

        for s in range(n_s):
            for a in range(n_a):
                v_next[s] += pi[s, a] * np.sum([p[s, a, s_] * (r[s, a] + gamma * v[s_]) for s_ in range(n_s)])
            
            delta = max(delta, abs(v_next[s] - v[s]))
        
        # For most cases where gamma = 1 and reward = 1, this stop
        # condition won't go through
        if delta < epsilon:
            break
        
        v = v_next
    
    return v