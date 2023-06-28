''' Test extended value iteration and related functions

This script serves to purpose of testing whether extended value iteration
(EVI) is working correctly. It does so by testing each of its individual
components independently to make sure they are working correctly from the
bottom up to the main function. The script can be categorized in the
following parts:

1.  Test the inner maximization function:
    Computes the optimistic MDP on EVI, so it is called frequently.

2.  Test the value iteration function:
    A special case of EVI, value iteration uses EVI on the background
    to compute the value function.

3.  Test the extended value iteration function as a whole.
'''



import unittest
from parameterized import parameterized
import numpy as np
from dotmap import DotMap
import gym

from psrl import agents
from psrl.agents import UCRL2Agent
from psrl.envs import GridworldEnv, TwoRoomGridworldEnv
from psrl.config import get_env_config

from testcase import TestCase
from value_iteration_utils import brute_force_value_iteration, grids_and_policies


def inner_maximization(p_sa, cb_p_sa, p_s_order):
    return agents.utils.inner_maximization(p_sa, cb_p_sa, p_s_order)

# def inner_maximization(p_sa, cb_p_sa, p_s_order):
#     env_config = get_env_config('riverswim')
#     env_config.n = p_sa.shape[0]
#     env = RiverSwimEnv(env_config)

#     agent_config = get_agent_config('ucrl2')
#     agent = UCRL2Agent(env, agent_config)

#     n_s = env.observation_space.n
#     n_a = env.action_space.n

#     p = np.zeros((n_a, n_a, n_s))
#     p[0, 0] = p_sa

#     agent.p_distances = np.zeros((n_a, n_a))
#     agent.p_distances[0, 0] = cb_p_sa

#     p_s_order = p_s_order[::-1]

#     return agent.max_proba(p, p_s_order, 0, 0)


class TestInnerMaximization(TestCase):
    n = 10
    
    def test_sorted_zero_cb(self):
        n = self.n

        alpha = np.ones(n)

        p_sa = np.random.dirichlet(alpha)
        cb_p_sa = 0
        p_s_order = np.arange(n)

        self.assertRoundEqual(p_sa.sum(), 1)

        p_sa_hat = inner_maximization(p_sa, cb_p_sa, p_s_order)

        self.assertEqual(p_sa_hat.shape, (n,), msg=p_sa_hat.shape)
        self.assertRoundEqual(p_sa_hat.sum(), 1)
        self.assertNumpyEqual(p_sa_hat, p_sa)
    
    def test_small_cb_sorted(self):
        n = self.n

        alpha = np.ones(n)

        p_sa = np.random.dirichlet(alpha)
        p_sa.sort()
        cb_p_sa = p_sa[0]
        p_s_order = n - 1 - np.arange(n)

        self.assertRoundEqual(p_sa.sum(), 1)

        p_sa_hat = inner_maximization(p_sa, cb_p_sa, p_s_order)

        self.assertEqual(p_sa_hat.shape, (n,), msg=p_sa_hat.shape)
        self.assertRoundEqual(p_sa_hat.sum(), 1)
        
        true_p_sa_hat = p_sa.copy()
        true_p_sa_hat[p_s_order[0]] += cb_p_sa / 2.
        true_p_sa_hat[p_s_order[-1]] -= cb_p_sa / 2.

        self.assertNumpyEqual(p_sa_hat, true_p_sa_hat)
    
    def test_dirac_result_distr(self):
        n = self.n

        alpha = np.ones(n)

        p_sa = np.random.dirichlet(alpha)
        cb_p_sa = 2 # confidence bound larger than total probability, hence result is a dirac distribution
        p_s_order = n - 1 - np.arange(n)

        self.assertRoundEqual(p_sa.sum(), 1)

        p_sa_hat = inner_maximization(p_sa, cb_p_sa, p_s_order)

        self.assertEqual(p_sa_hat.shape, (n,), msg=p_sa_hat.shape)
        self.assertRoundEqual(p_sa_hat.sum(), 1)
        
        true_p_sa_hat = np.zeros_like(p_sa)
        true_p_sa_hat[p_s_order[0]] = 1

        self.assertNumpyEqual(p_sa_hat, true_p_sa_hat)
    
    def test_test_full_loop(self):
        n = self.n

        p_sa = np.arange(n)
        p_sa = p_sa / p_sa.sum()
        cb_p_sa = 2 * (1. - p_sa[-1]- p_sa[1])
        p_s_order = n - 1 - np.arange(n)

        self.assertRoundEqual(p_sa.sum(), 1)

        p_sa_hat = inner_maximization(p_sa, cb_p_sa, p_s_order)

        self.assertEqual(p_sa_hat.shape, (n,), msg=p_sa_hat.shape)
        self.assertRoundEqual(p_sa_hat.sum(), 1)
        
        true_p_sa_hat = np.zeros_like(p_sa)
        true_p_sa_hat[-2] = p_sa[1]
        true_p_sa_hat[-1] = 1 - p_sa[1]

        self.assertNumpyEqual(p_sa_hat, true_p_sa_hat)




policy_evaluation_parameters = [
    (name, pi, gamma)
    for name in grids_and_policies
    for pi in grids_and_policies[name]['optimal_policies']
    for gamma in [1.0, 0.9]
]

class TestValueIteration(TestCase):
    epsilon = 1e-3      # Threshold of absolute maximum difference
    max_iter = 1000     # Maximum number of iterations

    
    @parameterized.expand(policy_evaluation_parameters)
    def test_policy_evaluation(self, name, pi, gamma):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_hat = agents.utils.policy_evaluation(
            p,
            r,
            pi,
            gamma=gamma,
            epsilon=self.epsilon,
            max_iter=self.max_iter
        )

        v = brute_force_value_iteration(
            p,
            r,
            pi,
            gamma,
            self.epsilon,
            self.max_iter
        )
        
        self.assertNumpyEqual(v_hat, v)

    
    def test_value_iteration(self):
        pass





if __name__ == '__main__':
    unittest.main()