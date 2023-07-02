import os
import torch
import torch.distributions as dist
import numpy as np
import pickle

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

no_goal = exp_config.no_goal
if args.goal_reward is not None:
    no_goal = args.goal_reward == 0
    
if not no_goal:
    exp_config.save_path = os.path.join(exp_config.save_path, 'regret_plot')
    exp_config.plots_path = os.path.join(exp_config.plots_path, 'regret_plot')



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = no_goal
env = env_class(env_config)



# Load trajectories
trajectories_path = get_file_path_from_config('option_trajectories.pkl', exp_config)
with open(trajectories_path, 'rb') as f:
    option_trajectory = pickle.load(f)

option_state_visitation = np.zeros((env.observation_space.n,))
env_state_visitation = np.zeros((env.observation_space.n,))

for option_transition in option_trajectory:
    op_state, option, reward, op_next_state, info = option_transition

    option_state_visitation[op_state] += 1
    for env_next_state in info['next_states']:
        env_state_visitation[env_next_state] += 1

option_state_visitation = np.array(option_state_visitation)



# Save plots
print('Saving plots...')

save_empirical_state_visitation_heatmap_plot(
    env,
    option_state_visitation,
    get_file_path_from_config('environment_state_visitation.png', exp_config, root_type='plots'),
    title='Option-level State Visitation',
)
save_empirical_state_visitation_heatmap_plot(
    env,
    env_state_visitation,
    get_file_path_from_config('option_state_visitation.png', exp_config, root_type='plots'),
    title='Environment-level State Visitation',
)