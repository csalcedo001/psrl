import os
import numpy as np

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map

from plotting import (
    save_policy_plot,
    save_expected_reward_heatmap_plot,
    save_state_value_heatmap_plot,
    save_action_value_heatmap_plot,
    save_empirical_state_visitation_heatmap_plot,
    save_reward_count_heatmap_plot,
)
from arg_utils import get_experiment_parser
from utils import load_experiment_config, set_seed, get_file_path_from_config




# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)



# Setup experiment
seed = exp_config.seed
if args.seed is not None:
    seed = args.seed
set_seed(seed)
print("*** SEED:", seed)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)



# Get agent
agent_class = agent_name_map[exp_config.agent]
agent_config = get_agent_config(exp_config.agent)
agent = agent_class(env, agent_config)
agent_path = get_file_path_from_config('agent.pkl', exp_config)
agent.load(agent_path)



# Get data from agent
if exp_config.agent in ['ucrl2', 'kl_ucrl']:
    p_hat = agent.Pk / np.clip(agent.Pk.sum(axis=2, keepdims=True), 1, None) + np.expand_dims(agent.p_distances, axis=2)
    r_hat = agent.Rk / np.clip(agent.Nk, 1, None) + agent.r_distances
    p_count = agent.Pk
    r_count = agent.Rk
    v = agent.u
    q = agent.q
    state_visitation = np.sum(agent.Nk + agent.vk, axis=1)
else:
    raise Exception(f'Agent {exp_config.agent} not supported.')



# Save plots
print('Saving plots...')

save_policy_plot(
    env,
    agent,
    get_file_path_from_config('policy.png', exp_config, mkdir=True, root_type='plots'),
    title='Policy',
)

save_expected_reward_heatmap_plot(
    env,
    r_hat,
    get_file_path_from_config('expected_reward.png', exp_config, root_type='plots'),
    title='Expected Reward',
)
save_action_value_heatmap_plot(
    env,
    q,
    get_file_path_from_config('action_value_function.png', exp_config, root_type='plots'),
    title='Action Value Function',
)
save_state_value_heatmap_plot(
    env,
    v,
    get_file_path_from_config('state_value.png', exp_config, root_type='plots'),
    title='State Value Function',
)
save_empirical_state_visitation_heatmap_plot(
    env,
    state_visitation,
    get_file_path_from_config('empirical_state_visitation.png', exp_config, root_type='plots'),
    title='Empirical State Visitation',
)
save_reward_count_heatmap_plot(
    env,
    r_count,
    get_file_path_from_config('empirical_total_reward.png', exp_config, root_type='plots'),
    title='Empirical Total Reward',
)
