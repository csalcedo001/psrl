import itertools
import numpy as np

from .agent import Agent
from .utils import extended_value_iteration


class UCRL2Agent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)

        self.env = env
        self.config = config

        self.pi = None

        self.setup()

    def setup(self):
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        self.t = 1
        self.vi = np.zeros((n_s, n_a))

        # Model state
        self.total_visitations = np.zeros((n_s, n_a))
        self.total_rewards = np.zeros((n_s, n_a))
        self.total_transitions = np.zeros((n_s, n_a, n_s))

    def act(self, state):
        if self.pi is None:
            self.update_policy()
        
        action = self.pi[state]
        return action

    def observe(self, transition):
        st, ac, reward, next_st = transition

        # Update statistics
        self.vi[st, ac] += 1
        self.total_rewards[st, ac] += reward
        self.total_transitions[st, ac, next_st] += 1

        # Next tick
        self.t += 1
        st = next_st

        if self.vi[st, ac] > max(1, self.total_visitations[st, ac]):
            self.update_policy()
    
    def update(self):
        pass
    
    def update_policy(self):
        n_s = self.env.observation_space.n
        n_a = self.env.action_space.n

        # Initialize episode k
        t_k = self.t
        # Per-episode visitations
        self.vi = np.zeros((n_s, n_a))
        # MLE estimates
        p_hat = self.total_transitions / np.clip(self.total_visitations.reshape((n_s, n_a, 1)), 1, None)
        # print('p_hat', p_hat)
        r_hat = self.total_rewards / np.clip(self.total_visitations, 1, None)
        # print('r_hat', r_hat)

        # Compute near-optimal policy for the optimistic MDP
        confidence_bound_r = np.sqrt(7 * np.log(2 * n_s * n_a * t_k / self.config.delta) / (2 * np.clip(self.total_visitations, 1, None)))
        confidence_bound_p = np.sqrt(14 * np.log(2 * n_a * t_k / self.config.delta) / np.clip(self.total_visitations, 1, None))
        # print('cb_p', confidence_bound_p)
        # print('cb_r', confidence_bound_r)
        pi_k, mdp_k = extended_value_iteration(n_s, n_a, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
        # print(pi_k, mdp_k)

        self.pi = pi_k
        

    

def ucrl2(mdp, delta, initial_state=None):
    '''
    UCRL2 algorithm
    See _Near-optimal Regret Bounds for Reinforcement Learning_. Jaksch, Ortner, Auer. 2010.
    '''
    n_states, n_actions = mdp.n_states, mdp.n_actions
    t = 1
    # Initial state
    st = mdp.reset(initial_state)
    # Model estimates
    total_visitations = np.zeros((n_states, n_actions))
    total_rewards = np.zeros((n_states, n_actions))
    total_transitions = np.zeros((n_states, n_actions, n_states))
    vi = np.zeros((n_states, n_actions))
    for k in itertools.count():
        # Initialize episode k
        t_k = t
        # Per-episode visitations
        vi = np.zeros((n_states, n_actions))
        # MLE estimates
        p_hat = total_transitions / np.clip(total_visitations.reshape((n_states, n_actions, 1)), 1, None)
        # print('p_hat', p_hat)
        r_hat = total_rewards / np.clip(total_visitations, 1, None)
        # print('r_hat', r_hat)

        # Compute near-optimal policy for the optimistic MDP
        confidence_bound_r = np.sqrt(7 * np.log(2 * n_states * n_actions * t_k / delta) / (2 * np.clip(total_visitations, 1, None)))
        confidence_bound_p = np.sqrt(14 * np.log(2 * n_actions * t_k / delta) / np.clip(total_visitations, 1, None))
        # print('cb_p', confidence_bound_p)
        # print('cb_r', confidence_bound_r)
        pi_k, mdp_k = extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
        # print(pi_k, mdp_k)

        # Execute policy
        ac = pi_k[st]
        # End episode when we visit one of the state-action pairs "often enough"
        while vi[st, ac] < max(1, total_visitations[st, ac]):
            next_st, reward = mdp.step(ac)
            # print('step', t, st, ac, next_st, reward)
            yield (t, st, ac, next_st, reward)
            # Update statistics
            vi[st, ac] += 1
            total_rewards[st, ac] += reward
            total_transitions[st, ac, next_st] += 1
            # Next tick
            t += 1
            st = next_st
            ac = pi_k[st]

        total_visitations += vi