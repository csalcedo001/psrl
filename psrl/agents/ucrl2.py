import numpy as np
import copy as cp
import pickle

from .agent import Agent
from .utils import extended_value_iteration


class UCRL2Agent(Agent):
    def __init__(self, env, config):
        Agent.__init__(self, env, config)

        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.t = 1
        self.delta = config.delta
        
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS,), dtype=int)
        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.q = np.zeros((self.nS, self.nA))

    def name(self):
        return "UCRL2"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]
    
    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]
    
    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1

    #Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
                                                / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.nS * np.log(2 * self.nA * self.t / self.delta))
                                                / (max([1, self.Nk[s, a]])))
        
    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.nS)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            max_p = np.clip(max_p, None, 1)
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])# Error?
                l += 1

        return max_p
    
    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
        action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
        # action_noise = np.zeros_like(action_noise)

        u0 = self.u # np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.nS)
        niter = 0
        while True:
            sorted_indices = np.argsort(u0)
            for s in range(self.nS):
                for a in range(self.nA):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)

                    r_tilde = min((1, r_estimate[s, a] + self.r_distances[s, a]))
                    temp = r_tilde + sum([u * p for (u, p) in zip(u0, max_p)])
                    if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
                        u1[s] = temp
                        self.policy[s] = a
            
            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                break

            u0 = u1
            u1 = np.zeros(self.nS)

            if niter > max_iter:
                break

            niter += 1
        
        self.u = u0

    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.updateN() # Don't run it after the reinitialization of self.vk
        self.vk = np.zeros((self.nS, self.nA))
        r_estimate = np.zeros((self.nS, self.nA))
        p_estimate = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.nS):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.policy, (self.u, self.q, _, _) = extended_value_iteration(p_estimate, r_estimate, self.p_distances, self.r_distances, max_iter=100)

    # To reinitialize the learner with a given initial state inistate.
    def reset(self,inistate):
        self.observations = [[inistate], [], []]
        self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def act(self,state):
        action = self.policy[state]
        if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
            self.new_episode()
            action  = self.policy[state]
        return action

    # To update the learner after one step of the current policy.
    def observe(self, transition):
        state, action, reward, observation = transition
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.updateP()
        self.updateR()
        self.t += 1
    
    def update(self):
        pass

    def save(self, path):
        data = {
            "observations": self.observations,
            "vk": self.vk,
            "Nk": self.Nk,
            "policy": self.policy,
            "r_distances": self.r_distances,
            "p_distances": self.p_distances,
            "Rk": self.Rk,
            "Pk": self.Pk,
            "u": self.u,    
            "nS": self.nS,
            "nA": self.nA,
            "delta": self.delta,
            "t": self.t,
            "q": self.q,
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data, out_file)

    def load(self, path):
        with open(path, 'rb') as in_file:
            data = pickle.load(in_file)
        
        self.observations = data["observations"]
        self.vk = data["vk"]
        self.Nk = data["Nk"]
        self.policy = data["policy"]
        self.r_distances = data["r_distances"]
        self.p_distances = data["p_distances"]
        self.Rk = data["Rk"]
        self.Pk = data["Pk"]
        self.u = data["u"]
        self.nS = data["nS"]
        self.nA = data["nA"]
        self.delta = data["delta"]
        self.t = data["t"]
        self.q = data["q"]


class KLUCRLAgent(UCRL2Agent):
    def __init__(self, env, config):
        UCRL2Agent.__init__(self, env, config)
        
        self.env = env
        self.config = config

        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.t = 1
        self.delta = config.delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS,), dtype=int)
        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
    
    def name(self):
        return "KL-UCRL"
    
    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    # KL-UCRL variant (Cp and Cr are difined as constrained constants in the paper of Filippi et al 2011, here we use the one used on the proofs
    # provided by the paper (with 2 instead of T at the initialization to prevent div by 0).
    def distances(self):
        B = np.log((2 * np.exp(1) * (self.nS)**2 * self.nA * np.log(max([2, self.t]))) / self.delta)
        Cp = self.nS * (B + np.log(B + 1 / np.log(max([2, self.t]))) * (1 + 1 / (B + 1 / np.log(max([2, self.t])))))
        Cr = np.sqrt((np.log(4 * self.nS * self.nA * np.log(max([2, self.t])) / self.delta)) / 1.99)
        for s in range(self.nS):
            for a in range(self.nA):
                self.r_distances[s, a] = Cr / np.sqrt(max([1, self.Nk[s, a]]))
                self.p_distances[s, a] = Cp / (max([1, self.Nk[s, a]]))

    # Key function of the problem -> solving the maximization problem is essentially based on finding roots of this function.
    def f(self, nu, p, V, Z_): # notations of the paper
        sum1 = 0
        sum2 = 0
        for i in Z_:
            if nu == V[i]:
                return - 10**10
            sum1 += p[i] * np.log(nu - V[i])
            sum2 += p[i] / (nu - V[i])
        if sum2 <= 0:
            return - 10**10
        return sum1 + np.log(sum2)
    
    # Derivative of f, used in newton optimization.
    def diff_f(self, nu, p, V, Z_, epsilon = 0):
        sum1 = 0
        sum2 = 0
        for i in range(len(p)):
            if i in Z_:
                sum1 += p[i] / (nu - V[i])
                sum2 += p[i] / (nu - V[i])**2
        return sum1 - sum2 / sum1

    # The maximization algorithm proposed by Filippi et al. 2011.
    # Inspired (for error preventing) from a Matlab Code provided by Mohammad Sadegh Talebi.
    # Exotics inputs:
    #    tau our approximation of 0
    #    max_iter maximmum number of iterations on newton optimization
    #    tol precision required in newton optimization
    def MaxKL(self, p_estimate, u0, s, a, tau = 10**(-8), max_iter = 10, tol = 10**(-5)):
        degenerate = False # used to catch some errors
        Z, Z_, argmax = [], [], []
        maxV = max(u0)
        q = np.zeros(self.nS)
        for i in range(self.nS):
            if u0[i] == maxV:
                argmax.append(i)
            if p_estimate[s, a, i] > tau:
                Z_.append(i)
            else:
                Z.append(i)
        I = []
        test0 = False
        for i in argmax:
            if i in Z:
                I.append(i)
                test0 = True
        if test0:
            test = [(self.f(u0[i], p_estimate[s, a], u0, Z_) < self.p_distances[s, a]) for i in I]
        else:
            test = [False]
        if (True in test) and (maxV > 0): # I must not and cannot be empty if this is true.
            for i in range(len(test)):
                if test[i]: # it has to happen because of previous if
                    nu = u0[I[i]]
                    break
            r = 1 - np.exp(self.f(nu, p_estimate[s, a], u0, Z_) - self.p_distances[s, a])
            for i in I: # We want sum(q[i]) for i in I = r.
                q[i] = r / len(I)
        else:
            if len(Z) >= self.nS - 1: # To prevent the algorithm from running the Newton optimization on a constant or undefined function.
                degenerate = True
                q = p_estimate[s, a]
            else:
                VZ_ = []
                for i in range(len(u0)):
                    if p_estimate[s, a, i] > tau:
                        VZ_.append(u0[i])
                nu0 = 1.1 * max(VZ_)  # This choice of initialization is based on the Matlab Code provided by Mohammad Sadegh Talebi, the one
                # provided by the paper leads to many errors while T is small.
                # about the following (unused) definition of nu0 see apendix B of Filippi et al 2011
                #nu0 = np.sqrt((sum([p_estimate[s, a, i] * u0[i]**2 for i in range(self.nS)]) -
                #              (sum([p_estimate[s, a, i] * u0[i] for i in range(self.nS)]))**2) / (2 * self.p_distances[s, a]))
                r = 0
                nu1 = 0
                err_nu = 10**10
                k = 1
                while (err_nu >= tol) and (k < max_iter):
                    nu1 = nu0 - (self.f(nu0, p_estimate[s, a], u0, Z_) - self.p_distances[s, a]) / (self.diff_f(nu0, p_estimate[s, a], u0, Z_))
                    if nu1 < max(VZ_):# f defined on ]max(VZ_); +inf[ we have to prevent newton optimization from going out from the definition interval
                        nu1 = max(VZ_) + tol
                        nu0 = nu1
                        k += 1
                        break
                    else:
                        err_nu = np.abs(nu1 - nu0)
                        k += 1
                        nu0 = nu1
                nu = nu0
        if not degenerate:
            q_tilde = np.zeros(self.nS)
            for i in Z_:
                if nu == u0[i]:
                    q_tilde[i] = p_estimate[s, a, i] * 10**10
                else:
                    q_tilde[i] = p_estimate[s, a, i] / (nu - u0[i])
            sum_q_tilde = sum(q_tilde)
            for i in Z_:
                q[i] = ((1 - r) * q_tilde[i]) / sum_q_tilde
        return q

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    # In KL-UCRL MaxKL used instead of max_proba (see Filippi et al. 2011).
    def EVI(self, r_estimate, p_estimate, epsilon = 0.1):
        tau = 10**(-6)
        maxiter = 1000

        u0 = np.zeros(self.nS)
        u1 = np.zeros(self.nS)
        niter = 0
        while True:
            for s in range(self.nS):
                test0 = np.any(u0 >= tau) # Test u0 != [0,..., 0]
                for a in range(self.nA):
                    if not test0: # MaxKL cannot run with V = [0, 0,..., 0, 0] because function f undifined in this case.
                        max_p = p_estimate[s, a]
                    else:
                        max_p = self.MaxKL(p_estimate, u0, s, a)
                    
                    q_sa = r_estimate[s, a] + self.r_distances[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])
                    self.q[s, a] = q_sa
                    if (a == 0) or (q_sa > u1[s]):
                        u1[s] = q_sa
                        self.policy[s] = a
            
            diff  = [x - y for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                break
        
            u0 = u1
            u1 = np.zeros(self.nS)

            if niter > maxiter:
                break

            niter += 1
        
        self.u = u0

    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.updateN() # Don't run it after the reinitialization of self.vk
        self.vk = np.zeros((self.nS, self.nA))
        r_estimate = np.zeros((self.nS, self.nA))
        p_estimate = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.nS):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.EVI(r_estimate, p_estimate)



        # self.nS = env.observation_space.n
        # self.nA = env.action_space.n
        # self.t = 1
        # self.delta = config.delta
        # self.observations = [[], [], []]
        # self.vk = np.zeros((self.nS, self.nA))
        # self.Nk = np.zeros((self.nS, self.nA))
        # self.policy = np.zeros((self.nS,), dtype=int)
        # self.r_distances = np.zeros((self.nS, self.nA))
        # self.p_distances = np.zeros((self.nS, self.nA))

    def save(self, path):
        data = {
            "nS": self.nS,
            "nA": self.nA,
            "t": self.t,
            "delta": self.delta,
            "observations": self.observations,
            "vk": self.vk,
            "Nk": self.Nk,
            "policy": self.policy,
            "r_distances": self.r_distances,
            "p_distances": self.p_distances,
            "Rk": self.Rk,
            "Pk": self.Pk,
            "u": self.u,
            "q": self.q,
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data, out_file)

    def load(self, path):
        with open(path, 'rb') as in_file:
            data = pickle.load(in_file)
        
        self.observations = data["observations"]
        self.vk = data["vk"]
        self.Nk = data["Nk"]
        self.policy = data["policy"]
        self.r_distances = data["r_distances"]
        self.p_distances = data["p_distances"]
        self.Rk = data["Rk"]
        self.Pk = data["Pk"]
        self.u = data["u"]
        self.nS = data["nS"]
        self.nA = data["nA"]
        self.delta = data["delta"]
        self.t = data["t"]
        self.q = data["q"]




# class UCRL2Agent(Agent):
#     def __init__(self, env, config):
#         Agent.__init__(self, env, config)

#         self.env = env
#         self.config = config

#         self.pi = None

#         self.setup()

#     def setup(self):
#         n_s = self.env.observation_space.n
#         n_a = self.env.action_space.n

#         self.t = 1
#         self.vi = np.zeros((n_s, n_a))

#         # Model state
#         self.total_visitations = np.zeros((n_s, n_a))
#         self.total_rewards = np.zeros((n_s, n_a))
#         self.total_transitions = np.zeros((n_s, n_a, n_s))

#     def act(self, state):
#         if self.pi is None:
#             self.update_policy()
        
#         action = self.pi[state]
#         return action

#     def observe(self, transition):
#         st, ac, reward, next_st = transition

#         # Update statistics
#         self.vi[st, ac] += 1
#         self.total_rewards[st, ac] += reward
#         self.total_transitions[st, ac, next_st] += 1

#         # Next tick
#         self.t += 1
#         st = next_st

#         if self.vi[st, ac] > max(1, self.total_visitations[st, ac]):
#             self.total_visitations += self.vi

#             self.update_policy()
    
#     def update(self):
#         pass
    
#     def update_policy(self):
#         n_s = self.env.observation_space.n
#         n_a = self.env.action_space.n

#         # Initialize episode k
#         t_k = self.t
#         # Per-episode visitations
#         self.vi = np.zeros((n_s, n_a))
#         # MLE estimates
#         p_hat = self.total_transitions / np.clip(self.total_visitations.reshape((n_s, n_a, 1)), 1, None)
#         # print('p_hat', p_hat)
#         r_hat = self.total_rewards / np.clip(self.total_visitations, 1, None)
#         # print('r_hat', r_hat)

#         # Compute near-optimal policy for the optimistic MDP
#         confidence_bound_r = np.sqrt(7 * np.log(2 * n_s * n_a * t_k / self.config.delta) / (2 * np.clip(self.total_visitations, 1, None)))
#         confidence_bound_p = np.sqrt(14 * np.log(2 * n_a * t_k / self.config.delta) / np.clip(self.total_visitations, 1, None))
#         # print('cb_p', confidence_bound_p)
#         # print('cb_r', confidence_bound_r)
#         pi_k, mdp_k = extended_value_iteration(n_s, n_a, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
#         # print(pi_k, mdp_k)

#         self.pi = pi_k
        

    

# def ucrl2(mdp, delta, initial_state=None):
#     '''
#     UCRL2 algorithm
#     See _Near-optimal Regret Bounds for Reinforcement Learning_. Jaksch, Ortner, Auer. 2010.
#     '''
#     n_states, n_actions = mdp.n_states, mdp.n_actions
#     t = 1
#     # Initial state
#     st = mdp.reset(initial_state)
#     # Model estimates
#     total_visitations = np.zeros((n_states, n_actions))
#     total_rewards = np.zeros((n_states, n_actions))
#     total_transitions = np.zeros((n_states, n_actions, n_states))
#     vi = np.zeros((n_states, n_actions))
#     for k in itertools.count():
#         # Initialize episode k
#         t_k = t
#         # Per-episode visitations
#         vi = np.zeros((n_states, n_actions))
#         # MLE estimates
#         p_hat = total_transitions / np.clip(total_visitations.reshape((n_states, n_actions, 1)), 1, None)
#         # print('p_hat', p_hat)
#         r_hat = total_rewards / np.clip(total_visitations, 1, None)
#         # print('r_hat', r_hat)

#         # Compute near-optimal policy for the optimistic MDP
#         confidence_bound_r = np.sqrt(7 * np.log(2 * n_states * n_actions * t_k / delta) / (2 * np.clip(total_visitations, 1, None)))
#         confidence_bound_p = np.sqrt(14 * np.log(2 * n_actions * t_k / delta) / np.clip(total_visitations, 1, None))
#         # print('cb_p', confidence_bound_p)
#         # print('cb_r', confidence_bound_r)
#         pi_k, mdp_k = extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
#         # print(pi_k, mdp_k)

#         # Execute policy
#         ac = pi_k[st]
#         # End episode when we visit one of the state-action pairs "often enough"
#         while vi[st, ac] < max(1, total_visitations[st, ac]):
#             next_st, reward = mdp.step(ac)
#             # print('step', t, st, ac, next_st, reward)
#             yield (t, st, ac, next_st, reward)
#             # Update statistics
#             vi[st, ac] += 1
#             total_rewards[st, ac] += reward
#             total_transitions[st, ac, next_st] += 1
#             # Next tick
#             t += 1
#             st = next_st
#             ac = pi_k[st]

#         total_visitations += vi