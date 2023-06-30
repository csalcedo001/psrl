import numpy as np

def get_steady_state_distribution(p, pi, epsilon=1e-2, max_iter=100):
    n_s, _ = p.shape[:2]
    mu_pi = np.ones(n_s) / n_s      # Initialize as uniform distribution

    for i in range(int(max_iter)):
        next_mu_pi = np.einsum("i,ij,ijk->k", mu_pi, pi, p)

        diff = np.abs(next_mu_pi - mu_pi).max()
        mu_pi = next_mu_pi

        if diff < epsilon:
            break    
    
    return mu_pi

def get_policy_average_reward(p, r, pi):
    mu_pi = get_steady_state_distribution(p, pi)
    return np.einsum("i,ij,ij", mu_pi, pi, r)



# MDP: 2 states, 2 actions. Action 0 transitions deterministically
# to state 0 on any state, likewise for action 1 and state 1. A reward
# of +1 is given only for transitions that keep the agent in the same
# state, i.e. (s, a) = (0, 0) or (s, a) = (1, 1).
p = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
r = np.array([[1, 0], [0, 1]])


policies_data = [
    {
        'pi': np.array([[0.5, 0.5], [0.5, 0.5]]),
        'e_avg_r': 0.5
    },
    {
        'pi': np.array([[0, 1], [0, 1]]),
        'e_avg_r': 1.0
    },
    {
        'pi': np.array([[1, 0], [0, 1]]),
        'e_avg_r': 1.0
    },
    {
        'pi': np.array([[0, 1], [1, 0]]),
        'e_avg_r': 0.0
    },
]

for policy_data in policies_data:
    pi = policy_data['pi']
    e_avg_r = policy_data['e_avg_r']

    avg_r = get_policy_average_reward(p, r, pi)

    print(f"Expected average reward: {e_avg_r}")
    print(f"Computed average reward: {avg_r}")
