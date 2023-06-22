import os
import numpy as np

from psrl.envs.gridworld import TwoRoomGridworldEnv
from psrl.config import get_env_config, get_agent_config

from ucrl2 import UCRL2Agent
from utils import load_experiment_config



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')

policy_path = os.path.join(exp_config.plot_path, 'policy')


env_config = get_env_config('tworoom')
env = TwoRoomGridworldEnv(env_config)



agent_config = get_agent_config('ucrl2')
agent = UCRL2Agent(env, agent_config)



agent = UCRL2Agent(env, agent_config)
step = exp_config.train_debug_episodes - 1

checkpoint_path = os.path.join(exp_config.save_path, 'checkpoint_{}.pkl'.format(str(step).zfill(4)))
agent.load(checkpoint_path)
agent.new_episode()

p_hat = agent.Pk / np.clip(agent.Pk.sum(axis=2, keepdims=True), 1, None) + np.expand_dims(agent.p_distances, axis=2)
r_hat = agent.Rk / np.clip(agent.Nk, 1, None) + agent.r_distances
p_count = agent.Pk
r_count = agent.Rk
v = agent.u
q = agent.q

print('agent.Rk: [{}, {}]'.format(agent.Rk.min(), agent.Rk.max()))
print('agent.Nk: [{}, {}]'.format(agent.Nk.min(), agent.Nk.max()))
print('agent.Pk: [{}, {}]'.format(agent.Pk.min(), agent.Pk.max()))
print('agent.u: [{}, {}]'.format(agent.u.min(), agent.u.max()))
print('agent.q: [{}, {}]'.format(agent.q.min(), agent.q.max()))
print('agent.p_distances: [{}, {}]'.format(agent.p_distances.min(), agent.p_distances.max()))
print('agent.r_distances: [{}, {}]'.format(agent.r_distances.min(), agent.r_distances.max()))
print('p_hat: [{}, {}]'.format(p_hat.min(), p_hat.max()))
print('r_hat: [{}, {}]'.format(r_hat.min(), r_hat.max()))
print('p_count: [{}, {}]'.format(p_count.min(), p_count.max()))
print('r_count: [{}, {}]'.format(r_count.min(), r_count.max()))
print('v: [{}, {}]'.format(v.min(), v.max()))
print('q: [{}, {}]'.format(q.min(), q.max()))