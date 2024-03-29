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
from value_iteration_utils import (
    brute_force_policy_evaluation,
    brute_force_policy_average_reward,
    brute_force_steady_state_distribution,
    stationary_transition_matrix,
    average_reward_policy_evaluation,
    compute_gain,
    average_reward_value_iteration_on_s,
    grids_and_policies,
)


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




disc_policy_evaluation_parameters = [
    (name, pi, gamma)
    for name in grids_and_policies
    for pi in grids_and_policies[name]['optimal_policies']
    for gamma in [
        # 1,       # In general, this shouldn't be a valid value for gamma
        0.99, 
        0.9,
    ]
]
disc_value_iteration_parameters = [
    (name, gamma)
    for name in grids_and_policies
    for gamma in [
        # 1,       # In general, this shouldn't be a valid value for gamma
        0.99,
        0.9,
    ]
]

class TestDiscountedValueIteration(TestCase):
    epsilon = 1e-3      # Threshold of absolute maximum difference
    max_iter = 1000     # Maximum number of iterations

    
    @parameterized.expand(disc_policy_evaluation_parameters)
    def test_policy_evaluation(self, name, pi, gamma):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        args = {
            'p': p,
            'r': r,
            'pi': pi,
            'gamma': gamma,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter
        }

        v_hat = agents.utils.policy_evaluation(**args)
        v = brute_force_policy_evaluation(**args)

        diff = np.abs(v_hat - v).max()
        self.assertTrue(diff < 2 * self.epsilon, msg=(diff, self.epsilon))

    
    @parameterized.expand(disc_value_iteration_parameters)
    def test_value_iter_v_star_hat_epsilon_close(self, name, gamma):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        args = {
            'p': p,
            'r': r,
            'gamma': gamma,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter
        }

        # Get an optimal policy and compute its value function
        pi_star = grids_and_policies[name]['optimal_policies'][0]
        v_star = brute_force_policy_evaluation(pi=pi_star, **args)


        # Make sure the value function returned directly from value iteration
        # is the same as the one obtained by the optimal policy through
        # policy evaluation
        _, q_star_hat = agents.utils.value_iteration(tau=1, **args)
        v_star_hat = q_star_hat.max(axis=1)

        diff = np.abs(v_star - v_star_hat).max()
        self.assertTrue(diff < 2 * self.epsilon, msg={
            'pi_star': pi_star,
            'v_star': v_star,
            'v_star_hat': v_star_hat,
        })


    
    @parameterized.expand(disc_value_iteration_parameters)
    def test_value_iteration_policy(self, name, gamma):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        args = {
            'p': p,
            'r': r,
            'gamma': gamma,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter
        }

        # Get an optimal policy and compute its value function
        pi = grids_and_policies[name]['optimal_policies'][0]
        v = brute_force_policy_evaluation(pi=pi, **args)


        # Make sure that the policy returned by policy iteration results in
        # the same value function as the one obtained by policy iteration
        # on the optimal policy
        pi_hat, _ = agents.utils.value_iteration(**args)
        v_hat = brute_force_policy_evaluation(pi=pi_hat, **args)

        diff = np.abs(v_hat - v).max()
        self.assertTrue(diff < 2 * self.epsilon, msg={
            'pi': pi,
            'v': v,
            'pi_hat': pi_hat,
            'v_hat': v_hat,
        })



avg_rew_policy_evaluation_parameters = [
    (name, pi)
    for name in grids_and_policies
    for pi in grids_and_policies[name]['optimal_policies']
]

class TestAverageRewardValueIteration(TestCase):
    epsilon = 1e-3      # Threshold of absolute maximum difference
    max_iter = 1000     # Maximum number of iterations
    
    @parameterized.expand([(name,) for name in grids_and_policies])
    def test_stat_mat_zero_error_all_policies(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1

            P_pi = np.einsum('ijk,ij->ik', p, pi)

            # Computes exact solution given enough iterations
            P_star_pi = stationary_transition_matrix(P_pi, epsilon=0, max_iter=10000)

            self.assertNumpyEqual(P_star_pi.sum(axis=1), np.ones(n_s))
            self.assertNumpyEqual(P_star_pi, P_star_pi @ P_pi)
            self.assertNumpyEqual(P_star_pi, P_star_pi @ P_star_pi)
    
    @parameterized.expand([(name,) for name in grids_and_policies])
    @unittest.skip('Not correctly implemented yet')
    def test_stat_mat_epsilon_close_all_policies(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1

            P_pi = np.einsum('ijk,ij->ik', p, pi)

            # Computes exact solution given enough iterations
            P_star_pi = stationary_transition_matrix(P_pi, epsilon=0, max_iter=10000)
            P_star_pi_hat = stationary_transition_matrix(P_pi, epsilon=1/self.max_iter, max_iter=self.max_iter)

            self.assertNumpyEqual(P_star_pi_hat.sum(axis=1), np.ones(n_s))

            error = np.abs(P_star_pi_hat - P_star_pi).max()
            self.assertLessEqual(error, self.epsilon)
    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    def test_gain_pi_star_vs_all_policies(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        rho_pi_star = compute_gain(p, r, pi_star, epsilon=self.epsilon, max_iter=self.max_iter)

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1

            rho_pi = compute_gain(p, r, pi, epsilon=self.epsilon, max_iter=self.max_iter)

            self.assertLessEqual(rho_pi.max(), rho_pi_star.max(), msg={
                'pi': pi,
                'rho_pi': rho_pi,
                'pi_star': pi_star,
                'rho_pi_star': rho_pi_star
            })
    

    @parameterized.expand([(name,) for name in grids_and_policies])
    def test_stat_mat_prod_val_func_eq_zero_all_policies(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)


        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1
            v_pi = average_reward_policy_evaluation(p, r, pi, epsilon=0, max_iter=10000)

            P_pi = np.einsum('ijk,ij->ik', p, pi)
            P_star_pi = stationary_transition_matrix(P_pi, epsilon=0, max_iter=10000)

            self.assertNumpyEqual(np.abs(P_star_pi @ v_pi), np.zeros_like(v_pi))

    @parameterized.expand([(name,) for name in grids_and_policies])
    def test_p_star_pi_prod_v_pi_zero_all_policies(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1
            P_pi = np.einsum('ijk,ij->ik', p, pi)
            P_star_pi = stationary_transition_matrix(P_pi, epsilon=0, max_iter=10000)
            v_pi = average_reward_policy_evaluation(p, r, pi, epsilon=self.epsilon, max_iter=self.max_iter)

            self.assertNumpyEqual(P_star_pi @ v_pi, np.zeros(n_s))
    
    @parameterized.expand([(name,) for name in grids_and_policies])
    def test_v_pi_on_bellman_eq_vs_all_policies(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1
            P_pi = np.einsum('ijk,ij->ik', p, pi)
            R_pi = np.einsum('ij,ij->i', r, pi)
            rho_pi = compute_gain(p, r, pi, epsilon=0, max_iter=10000)
            v_pi = average_reward_policy_evaluation(p, r, pi, epsilon=0, max_iter=10000)

            self.assertNumpyEqual(rho_pi + v_pi, R_pi + P_pi @ v_pi)
    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    def test_pi_star_bellman_opt_eq(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_pi_star = average_reward_policy_evaluation(p, r, pi_star, epsilon=self.epsilon, max_iter=self.max_iter)
        rho_pi_star = compute_gain(p, r, pi_star, epsilon=self.epsilon, max_iter=self.max_iter)

        self.assertNumpyEqual(rho_pi_star + v_pi_star, np.max(r + np.einsum('ijk,k->ij', p, v_pi_star), axis=1))
    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    def test_value_iter_s_tilde_is_v_star(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_star = average_reward_policy_evaluation(p, r, pi_star, epsilon=0, max_iter=10000)
        rho_star = compute_gain(p, r, pi_star, epsilon=0, max_iter=10000)[0]

        n_s = p.shape[0]

        v_star_s_tilde = None
        for s_tilde in range(n_s):
            v = average_reward_value_iteration_on_s(p, r, rho_star, s_tilde, epsilon=0, max_iter=10000)

            if np.all(np.abs(rho_star + v - np.max(r + np.einsum('ijk,k->ij', p, v), axis=1)) < 1e-9):
                v_star_s_tilde = v
                break
        
        self.assertTrue(v_star_s_tilde is not None, msg={
            'p': p,
            'r': r,
            'pi_star': pi_star,
            'v_star': v_star,
            'rho_star': rho_star,
            's_tilde': s_tilde,
            'v': v,
        })

        v_star_s_tilde -= v_star_s_tilde[0]
        v_star -= v_star[0]

        self.assertNumpyEqual(v_star_s_tilde, v_star)
    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    def test_value_iter_v_star_hat_is_v_star(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_star = average_reward_policy_evaluation(p, r, pi_star, epsilon=0, max_iter=10000)
        rho_star = compute_gain(p, r, pi_star, epsilon=0, max_iter=10000)[0]
        
        r_hat = r - rho_star
        pi_star_hat, q_star_hat = agents.utils.value_iteration(p, r_hat, gamma=1, epsilon=0, max_iter=10000)
        v_star_hat = np.max(q_star_hat, axis=1)

        self.assertLessEqual(np.abs(v_star_hat - v_star).max(), 2 * self.epsilon, msg={
            'epsilon': self.epsilon,
            'pi_star': pi_star,
            'pi_star_hat': pi_star_hat,
            'v_star': v_star,
            'v_star_hat': v_star_hat,
            'p': p,
            'r': r,
        })
    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    def test_value_iter_pi_star_hat_is_pi_star(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_star = average_reward_policy_evaluation(p, r, pi_star, epsilon=0, max_iter=10000)

        n_s, n_a = p.shape[:2]

        pi_star_hat_idx, _ = agents.utils.value_iteration(p, r, gamma=1, epsilon=0, max_iter=10000)
        pi_star_hat = np.zeros((n_s, n_a))
        pi_star_hat[np.arange(n_s), pi_star_hat_idx] = 1
        v_star_hat = average_reward_policy_evaluation(p, r, pi_star_hat, epsilon=0, max_iter=10000)

        self.assertLessEqual(np.abs(v_star_hat - v_star).max(), 1e-9, msg={
            'pi_star': pi_star,
            'pi_star_hat': pi_star_hat,
            'v_star': v_star,
            'v_star_hat': v_star_hat,
            'p': p,
            'r': r,
        })

    
    @parameterized.expand(avg_rew_policy_evaluation_parameters)
    @unittest.skip("Not implemented correctly yet")
    def test_v_star_gr_eq_v_pi_all_policies(self, name, pi_star):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()

        v_star, _ = agents.utils.value_iteration(p, r, pi_star, gamma=1, epsilon=self.epsilon, max_iter=self.max_iter)
        rho_star = compute_gain(p, r, pi_star, epsilon=0, max_iter=10000)

        n_s, n_a = p.shape[:2]

        # Total of |A| ^ |S| greedy policies, test feasible only on small MDPs
        all_policies = np.array(np.meshgrid(*np.tile(np.arange(n_a), (n_s, 1)))).T.reshape(-1, n_s)

        for pi_idx in all_policies:
            pi = np.zeros((n_s, n_a))
            pi[np.arange(n_s), pi_idx] = 1

            v_pi, _ = agents.utils.value_iteration(p, r, pi, gamma=1, epsilon=0, max_iter=10000)
            rho_pi = compute_gain(p, r, pi, epsilon=0, max_iter=10000)

            self.assertTrue(np.all(rho_star + v_star - v_pi - rho_pi > 0), msg=({
                'r': r,
                'p': p,
                'v_star': v_star,
                'v_pi': v_pi,
                'pi_star': pi_star,
                'pi': pi
            }))


        
    @parameterized.expand([(name,) for name in grids_and_policies])
    @unittest.skip('Unused function')
    def test_steady_state_distribution(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, _ = env.get_p_and_r()
        # TODO: parameterize policy
        pi = grids_and_policies[name]['optimal_policies'][0]

        mu_pi = brute_force_steady_state_distribution(p, pi)
        mu_pi_hat = agents.utils.get_steady_state_distribution(p, pi)

        self.assertRoundEqual(mu_pi_hat.sum(), 1)
        self.assertNumpyEqual(mu_pi_hat, mu_pi)


    @parameterized.expand([(name,) for name in grids_and_policies])
    def test_policy_average_reward(self, name):
        env_config = get_env_config('gridworld')
        env_config.grid = grids_and_policies[name]['grid']
        env = GridworldEnv(env_config)

        p, r = env.get_p_and_r()
        # TODO: parameterize policy
        pi = grids_and_policies[name]['optimal_policies'][0]

        avg_r = brute_force_policy_average_reward(p, r, pi)
        avg_r_hat = agents.utils.get_policy_average_reward(p, r, pi)

        self.assertRoundEqual(avg_r_hat, avg_r)




if __name__ == '__main__':
    unittest.main(verbosity=2)