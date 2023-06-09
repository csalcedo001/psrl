import numpy as np
import copy as cp

from psrl.agents.agent import Agent


class UCRL2_v1(Agent):
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

		self.EVI(r_estimate, p_estimate)

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