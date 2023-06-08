import unittest
import numpy as np
from dotmap import DotMap

from psrl import agents
from psrl.agents import UCRL2Agent
from psrl.envs import RiverSwimEnv
from psrl.config import get_agent_config, get_env_config

from .testcase import TestCase


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

if __name__ == '__main__':
    unittest.main()