import numpy as np


from setup_script import setup_script
from file_system import load_pickle
from plotting import (
    save_empirical_state_visitation_heatmap_plot,
)
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()



# Load trajectories
trajectories_path = get_file_path_from_config('option_trajectories.pkl', exp_config)
option_trajectory = load_pickle(trajectories_path)

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
    get_file_path_from_config('option_state_visitation.png', exp_config, root_type='plots'),
    title='Option-level State Visitation',
)
save_empirical_state_visitation_heatmap_plot(
    env,
    env_state_visitation,
    get_file_path_from_config('environment_state_visitation.png', exp_config, root_type='plots'),
    title='Environment-level State Visitation',
)